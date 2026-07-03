"""P3'-backend: HF download progress + installed-model management + config.

All HF interaction (snapshot_download, scan_cache_dir, HfApi) is monkeypatched — no real
network or cache is touched. The download seam is `api._download_model`; the cache seams
are `api._installed_models` / `api._delete_cached`.
"""
import json

import torch
from fastapi.testclient import TestClient
from transformers import LlamaConfig, LlamaForCausalLM

import parametic_studio.api as api
from parametic_studio.kernel.model_session import ModelSession


class _Tok:
    eos_token_id = -1

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return f"<{ids[0]}>"


def _tiny_session():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return ModelSession(LlamaForCausalLM(cfg).eval(), _Tok(), torch.device("cpu"))


def test_catalog_carries_installed_and_size(monkeypatch):
    # one catalog model cached, plus a cache-only model not in the static catalog
    monkeypatch.setattr(api, "_installed_models", lambda: [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "size_mb": 990.0},
        {"id": "some/other-model", "size_mb": 123.0},
    ])
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "catalog"})
        msg = ws.receive_json()
    assert msg["type"] == "catalog"
    by_id = {m["id"]: m for m in msg["models"]}
    assert by_id["Qwen/Qwen2.5-0.5B-Instruct"]["installed"] is True
    assert by_id["Qwen/Qwen2.5-0.5B-Instruct"]["size_mb"] == 990.0
    assert by_id["Qwen/Qwen2.5-1.5B-Instruct"]["installed"] is False    # catalog model not cached
    assert by_id["Qwen/Qwen2.5-1.5B-Instruct"]["size_mb"] is None
    assert "some/other-model" in by_id                                  # cache-only surfaced
    assert by_id["some/other-model"]["label"] == "other-model"


def test_installed_models_lists_cached(monkeypatch):
    monkeypatch.setattr(api, "_installed_models",
                        lambda: [{"id": "a/b", "size_mb": 10.0}, {"id": "c/d", "size_mb": 20.0}])
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "installed_models"})
        msg = ws.receive_json()
    assert msg["type"] == "installed_models"
    assert msg["items"] == [{"id": "a/b", "size_mb": 10.0}, {"id": "c/d", "size_mb": 20.0}]


def test_open_streams_download_progress_then_opened(monkeypatch):
    api.SESSIONS = {}       # not resident → download path runs
    api._locks = {}
    sess = _tiny_session()

    async def fake_download(model_id, on_progress=None):
        for i in (1, 2, 3):
            await on_progress(float(i * 100), 300.0, i * 100 / 3)

    monkeypatch.setattr(api, "_download_model", fake_download)
    monkeypatch.setattr("parametic_studio.kernel.model_session.ModelSession.from_pretrained",
                        classmethod(lambda cls, mid, device="auto": sess))

    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "open", "model": "m/new"})
        msgs = [ws.receive_json() for _ in range(5)]  # loading + 3 progress + opened
    types = [m["type"] for m in msgs]
    assert types == ["loading", "download_progress", "download_progress", "download_progress", "opened"]
    progress = [m for m in msgs if m["type"] == "download_progress"]
    assert [p["done_mb"] for p in progress] == [100.0, 200.0, 300.0]
    assert all(p["total_mb"] == 300.0 and p["model"] == "m/new" for p in progress)
    assert "m/new" in api.SESSIONS


def test_open_already_loaded_skips_download(monkeypatch):
    api.SESSIONS = {"m1": _tiny_session()}

    async def boom(*a, **k):
        raise AssertionError("resident model must not re-download")

    monkeypatch.setattr(api, "_download_model", boom)
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "open", "model": "m1"})
        a, b = ws.receive_json(), ws.receive_json()
    assert a["type"] == "loading" and b["type"] == "opened"


def test_open_bad_repo_emits_load_failed_then_error(monkeypatch):
    api.SESSIONS = {}
    api._locks = {}

    async def fail(model_id, on_progress=None):
        raise RuntimeError("Repository Not Found")

    monkeypatch.setattr(api, "_download_model", fail)
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "open", "model": "no/such-model"})
        a = ws.receive_json()
        b = ws.receive_json()
        c = ws.receive_json()
    assert a["type"] == "loading"
    assert b["type"] == "load_failed" and b["model"] == "no/such-model"
    assert c["type"] == "error" and "Not Found" in c["reason"]
    assert "no/such-model" not in api.SESSIONS


def test_delete_cached_acks(monkeypatch):
    called = {}
    monkeypatch.setattr(api, "_delete_cached", lambda mid: called.setdefault("mid", mid))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "delete_cached", "model": "a/b"})
        msg = ws.receive_json()
    assert msg["type"] == "cached_deleted" and msg["model"] == "a/b"
    assert called["mid"] == "a/b"


def test_delete_cached_leaves_loaded_session(monkeypatch):
    # deleting disk cache must not evict a live session
    api.SESSIONS = {"a/b": _tiny_session()}
    monkeypatch.setattr(api, "_delete_cached", lambda mid: None)
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "delete_cached", "model": "a/b"})
        ws.receive_json()
    assert "a/b" in api.SESSIONS


def test_config_get_set_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "get_config"})
        assert ws.receive_json() == {"type": "config", "config": {}}   # missing file → {}
        ws.send_json({"type": "set_config", "config": {"theme": "dark"}})
        assert ws.receive_json()["config"] == {"theme": "dark"}
        ws.send_json({"type": "set_config", "config": {"remote_url": "x"}})   # merge, not replace
        assert ws.receive_json()["config"] == {"theme": "dark", "remote_url": "x"}
        ws.send_json({"type": "get_config"})
        assert ws.receive_json()["config"] == {"theme": "dark", "remote_url": "x"}
    assert json.loads((tmp_path / "config.json").read_text()) == {"theme": "dark", "remote_url": "x"}


# ---- download filter + progress accuracy (lead-review regression: pct 144%, done>total) ----

class _Sib:
    def __init__(self, rfilename, size):
        self.rfilename, self.size = rfilename, size


def test_needed_files_one_filter_for_patterns_and_total():
    # regression: total summed ALL siblings while the download had no filter → done/total desync.
    sibs = [_Sib("model.safetensors", 200_000_000), _Sib("config.json", 1_000),
            _Sib("tokenizer.json", 2_000_000), _Sib("onnx/model_q4.onnx", 500_000_000),
            _Sib("onnx/model.onnx", 500_000_000), _Sib("pytorch_model.bin", 300_000_000)]
    patterns, total = api._needed_files(sibs)
    assert patterns == api._ALLOW_PATTERNS                       # safetensors present → no .bin
    assert total == 200_000_000 + 1_000 + 2_000_000              # onnx + bin excluded from total too


def test_needed_files_bin_fallback_when_no_safetensors():
    sibs = [_Sib("pytorch_model.bin", 300_000_000), _Sib("config.json", 1_000)]
    patterns, total = api._needed_files(sibs)
    assert "*.bin" in patterns and "*.safetensors" not in patterns
    assert total == 300_000_000 + 1_000


def test_download_model_passes_filter_and_clamps_pct(monkeypatch, tmp_path):
    # blobs dir pre-seeded LARGER than the filtered total → pct must clamp at 100, never 144.
    import asyncio
    import threading

    import huggingface_hub
    import huggingface_hub.constants as hub_constants

    blobs = tmp_path / "models--a--b" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "blob1").write_bytes(b"x" * 3000)          # 3000B on disk vs total 2000B

    class _Api:
        def model_info(self, mid, files_metadata=True):
            import types
            return types.SimpleNamespace(siblings=[_Sib("model.safetensors", 2000),
                                                   _Sib("onnx/big.onnx", 999_999)])

    released = threading.Event()
    captured = {}

    def fake_snapshot(model_id, allow_patterns=None):
        captured["patterns"] = allow_patterns
        released.wait(timeout=10)                        # stay "downloading" until one poll fires

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))

    progress = []

    async def on_progress(done_mb, total_mb, pct):
        progress.append((done_mb, total_mb, pct))
        released.set()                                   # let the fake download finish

    asyncio.run(api._download_model("a/b", on_progress))
    assert captured["patterns"] == api._ALLOW_PATTERNS   # the same filter reached snapshot_download
    assert progress and progress[0][2] == 100.0          # clamped (raw would be 150%)
    assert progress[0][1] == 2000 / 1e6                  # total from the FILTERED siblings (onnx excluded)
