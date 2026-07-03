import json
import os

import pytest
import torch
from fastapi import WebSocketDisconnect
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


def test_generate_streams_tokens_then_done():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 5})
        toks, attns = [], []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            (toks if msg["type"] == "token" else attns).append(msg)
    assert [t["step"] for t in toks] == [0, 1, 2, 3, 4]
    assert all("model" in t for t in toks)
    assert "text" in toks[0] and "attn" not in toks[0]  # token event carries no raw tensor
    # P=3 prompt → 3 prefill rows + 4 decode rows = full causal square (row i → kv i+1)
    assert len(attns) == 7
    assert [a["shape"] for a in attns] == [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]
    assert msg["reason"] == "max_tokens"


def test_ws_streams_activation_when_subscribed():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 4, "probes": ["activation"]})
        acts = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            if msg["type"] == "activation":
                acts.append(msg)
    assert [a["step"] for a in acts] == [0, 1, 2, 3]
    assert acts[0]["shape"] == [2, 2]  # [L, modules]
    assert len(acts[0]["data"]) == 2


def test_ws_streams_logitlens_when_subscribed():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 3, "probes": ["logitlens"]})
        lens = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            if msg["type"] == "logitlens":
                lens.append(msg)
    assert [m["step"] for m in lens] == [0, 1, 2]
    assert len(lens[0]["layers"]) == 2  # L layers
    assert "token" in lens[0]["layers"][0] and "prob" in lens[0]["layers"][0]


def test_ws_close_unloads_model():
    api.SESSIONS = {"m1": _tiny_session(), "m2": _tiny_session()}
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "close", "model": "m1"})
        msg = ws.receive_json()
    assert msg["type"] == "closed" and msg["model"] == "m1"
    assert "m1" not in api.SESSIONS and "m2" in api.SESSIONS


def test_ws_close_default_model_drops_session_ref():
    # regression: closing the boot-default model must clear the SESSION global, or its weights stay in memory forever.
    s = _tiny_session()
    api.SESSION = s
    api.SESSIONS = {"m1": s}
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "close", "model": "m1"})
        assert ws.receive_json()["type"] == "closed"
    assert api.SESSION is None and "m1" not in api.SESSIONS


def test_ws_open_emits_loading_then_opened():
    api.SESSIONS = {"m1": _tiny_session()}  # pre-loaded → no download
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "open", "model": "m1"})
        a = ws.receive_json()
        b = ws.receive_json()
    assert a["type"] == "loading" and a["model"] == "m1"
    assert b["type"] == "opened" and b["model"] == "m1"


def test_ws_routes_and_tags_by_model():
    api.SESSIONS = {"m1": _tiny_session(), "m2": _tiny_session()}
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 2, "model": "m1", "probes": []})
        toks = []
        while True:
            m = ws.receive_json()
            if m["type"] == "done":
                break
            if m["type"] == "token":
                toks.append(m)
    assert toks and all(t["model"] == "m1" for t in toks)
    assert m["model"] == "m1"  # done is tagged too


def test_ws_spot_streams_progress_then_spotmap():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "spot", "examples": ["alpha", "beta", "gamma"]})
        progress, final = [], None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "spotmap":
                final = msg
                break
            progress.append(msg)
    assert [p["type"] for p in progress] == ["spot_progress"] * 3
    assert [p["i"] for p in progress] == [1, 2, 3]
    assert all(p["total"] == 3 for p in progress)
    assert all(len(p["grid"]) == p["layers"] for p in progress)   # live grid each step
    assert final["layers"] == 2 and len(final["grid"]) == 2       # spotmap unchanged
    assert final["grid"] == progress[-1]["grid"]                  # final == last progress


def test_drilldown_after_generation():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 5})
        while ws.receive_json()["type"] != "done":
            pass
        ws.send_json({"type": "drilldown", "layer": 0})
        msg = ws.receive_json()
    assert msg["type"] == "perhead"
    assert msg["layer"] == 0
    assert msg["shape"][0] == 2  # heads
    assert len(msg["data"]) == 2


def test_ws_ppl_intervene_clear_cycle():
    api.SESSION = _tiny_session()
    ex = ["alpha", "beta"]
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "ppl", "examples": ex})
        base = ws.receive_json()
        assert base["type"] == "ppl" and isinstance(base["value"], float)
        ws.send_json({"type": "intervene", "region": {"kind": "spot", "examples": ex, "topk": 0.5}, "op": "zero"})
        while (m := ws.receive_json())["type"] == "locate_progress":  # spot locate streams progress first
            pass
        assert m["type"] == "intervened"
        ws.send_json({"type": "ppl", "examples": ex})
        assert ws.receive_json()["value"] != base["value"]      # intervention changed ppl
        ws.send_json({"type": "clear"})
        assert ws.receive_json()["type"] == "cleared"
        ws.send_json({"type": "ppl", "examples": ex})
        assert abs(ws.receive_json()["value"] - base["value"]) < 1e-4  # restored


def test_ws_ops_route_to_named_model_not_default():
    # regression: knob/ppl ops must hit the requested model, never silently fall back to SESSION.
    api.SESSION = None  # a fallback would crash on None
    api.SESSIONS = {"m1": _tiny_session()}
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "ppl", "model": "m1", "examples": ["alpha"]})
        msg = ws.receive_json()
    assert msg["type"] == "ppl" and isinstance(msg["value"], float)


def test_ws_tensors_lists_metadata():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "tensors"})
        msg = ws.receive_json()
    assert msg["type"] == "tensors"
    assert any(t["name"] == "model.layers.0.mlp.gate_proj.weight" for t in msg["tensors"])
    assert all("shape" in t and "dtype" in t for t in msg["tensors"])


def test_ws_save_region_then_named_intervene():
    api.SESSION = _tiny_session()
    ex = ["alpha", "beta"]
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "save_region", "name": "r1", "region": {"kind": "spot", "examples": ex, "topk": 0.5}})
        while (saved := ws.receive_json())["type"] == "locate_progress":  # spot locate streams progress first
            pass
        assert saved["type"] == "region_saved" and saved["name"] == "r1" and saved["count"] > 0
        ws.send_json({"type": "regions"})
        lst = ws.receive_json()
        assert lst["regions"] == [{"name": "r1", "count": saved["count"]}]
        ws.send_json({"type": "ppl", "examples": ex})
        base = ws.receive_json()["value"]
        ws.send_json({"type": "intervene", "region": {"kind": "named", "name": "r1"}, "op": "zero"})
        assert ws.receive_json()["type"] == "intervened"
        ws.send_json({"type": "ppl", "examples": ex})
        assert ws.receive_json()["value"] != base    # named region intervention took effect
        ws.send_json({"type": "clear"})
        ws.receive_json()


def test_ws_delete_region_returns_refreshed_list():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "save_region", "name": "gone", "region": {"kind": "cell", "layer": 0, "module": "mlp.gate_proj.weight"}})
        ws.receive_json()
        ws.send_json({"type": "delete_region", "name": "gone"})
        msg = ws.receive_json()
    assert msg["type"] == "regions" and msg["regions"] == []


def test_ws_region_compare_roundtrip():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        for name in ("ra", "rb"):
            ws.send_json({"type": "save_region", "name": name, "region": {"kind": "cell", "layer": 0, "module": "mlp.gate_proj.weight"}})
            ws.receive_json()
        ws.send_json({"type": "region_compare", "names": ["ra", "rb"]})
        msg = ws.receive_json()
    assert msg["type"] == "region_comparison"
    assert msg["jaccard"]["ra|rb"] == 1.0 and set(msg["grids"]) == {"ra", "rb"}


def test_ws_suspend_resume_ab_cycle():
    api.SESSION = _tiny_session()
    ex = ["alpha", "beta"]
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "ppl", "examples": ex})
        base = ws.receive_json()["value"]
        ws.send_json({"type": "intervene", "region": {"kind": "cell", "layer": 0, "module": "mlp.gate_proj.weight"}, "op": "zero", "key": "k"})
        assert ws.receive_json()["type"] == "intervened"
        ws.send_json({"type": "ppl", "examples": ex})
        inter = ws.receive_json()["value"]
        ws.send_json({"type": "suspend"})
        assert ws.receive_json()["type"] == "suspended"
        ws.send_json({"type": "ppl", "examples": ex})
        assert abs(ws.receive_json()["value"] - base) < 1e-4    # suspended == baseline
        ws.send_json({"type": "resume"})
        assert ws.receive_json()["type"] == "resumed"
        ws.send_json({"type": "ppl", "examples": ex})
        assert abs(ws.receive_json()["value"] - inter) < 1e-4   # resumed == intervened


def test_ws_dataset_save_list_read_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "save_dataset", "name": "bench/my.jsonl", "content": '{"q":"a"}'})
        assert ws.receive_json()["type"] == "dataset_saved"
        assert (tmp_path / "datasets" / "bench" / "my.jsonl").exists()   # persisted on disk
        ws.send_json({"type": "datasets"})
        items = ws.receive_json()["items"]
        assert [i["name"] for i in items] == ["bench/my.jsonl"] and items[0]["size"] > 0
        ws.send_json({"type": "read_dataset", "name": "bench/my.jsonl"})
        msg = ws.receive_json()
        assert msg["type"] == "dataset_content" and msg["content"] == '{"q":"a"}'
        ws.send_json({"type": "delete_dataset", "name": "bench/my.jsonl"})
        assert ws.receive_json()["type"] == "dataset_saved"              # generic "changed" ack
        ws.send_json({"type": "datasets"})
        assert ws.receive_json()["items"] == []


def test_ws_link_path_symlinks_external_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path / "home"))
    ext = tmp_path / "external.jsonl"
    ext.write_text('{"q":"x"}')
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "link_path", "path": str(ext)})
        assert ws.receive_json()["type"] == "dataset_saved"
        ws.send_json({"type": "read_dataset", "name": "external.jsonl"})
        assert ws.receive_json()["content"] == '{"q":"x"}'               # reads through the symlink
        ws.send_json({"type": "link_path", "path": str(tmp_path / "nope.txt")})
        assert ws.receive_json()["type"] == "error"


def test_ws_delete_never_reaches_through_folder_symlink(tmp_path, monkeypatch):
    # regression: × on a file inside a linked folder must NOT delete the user's real file.
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path / "home"))
    ext = tmp_path / "myfolder"
    ext.mkdir()
    (ext / "real.jsonl").write_text('{"q":"keep me"}')
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "link_path", "path": str(ext)})
        assert ws.receive_json()["type"] == "dataset_saved"
        ws.send_json({"type": "datasets"})
        items = ws.receive_json()["items"]
        assert items[0]["name"] == "myfolder/real.jsonl" and items[0]["link"] == "myfolder"  # linked flag
        ws.send_json({"type": "delete_dataset", "name": "myfolder/real.jsonl"})
        assert ws.receive_json()["type"] == "error"                    # refused
        assert (ext / "real.jsonl").exists()                           # original untouched
        ws.send_json({"type": "delete_dataset", "name": "myfolder"})   # unlink the link itself
        assert ws.receive_json()["type"] == "dataset_saved"
        assert (ext / "real.jsonl").exists()                           # original still fine
        ws.send_json({"type": "datasets"})
        assert ws.receive_json()["items"] == []                        # just disconnected


def test_ws_dataset_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "read_dataset", "name": "../../etc/passwd"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "save_dataset", "name": "../evil.txt", "content": "x"})
        assert ws.receive_json()["type"] == "error"
        assert not (tmp_path.parent / "evil.txt").exists()


def test_ws_stop_spot_ends_early_with_partial_map():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "spot", "examples": ["x"] * 200})
        ws.receive_json()                      # at least one progress step arrived
        ws.send_json({"type": "stop_spot"})
        progress, final = 1, None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "spotmap":
                final = msg
                break
            progress += 1
        assert progress < 200                  # stopped before finishing all examples
        assert final["reason"] == "stopped" and len(final["grid"]) == 2  # partial map still usable


def test_ws_train_streams_steps_then_trained_and_resets():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "train", "examples": ["alpha", "beta"], "mode": "full", "steps": 3, "lr": 1e-3})
        steps, final = [], None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "trained":
                final = msg
                break
            steps.append(msg)
    assert [s["type"] for s in steps] == ["train_step"] * 3
    assert [s["step"] for s in steps] == [0, 1, 2]
    assert all(isinstance(s["loss"], float) and s["total"] == 3 for s in steps)
    assert final["mode"] == "full" and final["steps"] == 3 and final["reason"] == "done"
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset_train"})
        assert ws.receive_json()["type"] == "train_reset"


def test_ws_busy_rejects_ops_while_training():
    api.SESSIONS = {"m1": _tiny_session()}
    api.TRAINING.add("m1")  # simulate mid-training (the guard is what we lock here)
    try:
        with TestClient(api.app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "generate", "model": "m1", "prompt": "hi", "max_tokens": 2})
            msg = ws.receive_json()
            assert msg["type"] == "error" and "busy" in msg["reason"]
            ws.send_json({"type": "ppl", "model": "m1", "examples": ["x"]})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "ppl", "model": None, "examples": ["x"]})  # other models unaffected
    finally:
        api.TRAINING.discard("m1")


class _Stoppable:
    def __init__(self):
        self._stop = False

    def stop(self):
        self._stop = True

    def generate_text(self, prompt, max_tokens, probes=("attention",), **kw):
        step = 0
        while not self._stop and step < max_tokens:
            yield {"step": step, "token_id": step, "text": str(step)}
            step += 1


def test_stop_ends_with_done_stopped():
    api.SESSION = _Stoppable()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "x", "max_tokens": 5000})
        ws.receive_json()  # at least one token before stopping
        ws.send_json({"type": "stop"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
    assert msg["reason"] == "stopped"


def test_ws_run_save_list_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    run = {"prompt": "p" * 200, "output": "hello", "frames": [{"t": 0}, {"t": 1}, {"t": 2}]}
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "save_run", "model": "m/x", "run": run})
        saved = ws.receive_json()
        assert saved["type"] == "run_saved" and saved["id"]
        ws.send_json({"type": "runs", "model": "m/x"})
        items = ws.receive_json()["items"]
        assert len(items) == 1
        assert items[0]["prompt"] == "p" * 80 and items[0]["tokens"] == 3   # 80-char prompt, frame count
        ws.send_json({"type": "load_run", "model": "m/x", "id": saved["id"]})
        msg = ws.receive_json()
        assert msg["type"] == "run_data" and msg["run"] == run              # gzip roundtrip, verbatim


def test_ws_runs_newest_first_and_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        for i in range(52):
            ws.send_json({"type": "save_run", "model": "m/x", "run": {"prompt": f"r{i}", "frames": []}})
            ws.receive_json()
        ws.send_json({"type": "runs", "model": "m/x"})
        items = ws.receive_json()["items"]
    assert len(items) == 50                                                # retention: newest 50 kept
    ids = [int(i["id"]) for i in items]
    assert ids == sorted(ids, reverse=True)                                # newest first
    d = tmp_path / "runs" / "m__x"
    idx = json.loads((d / "index.json").read_text())
    gz = {p.stem.removesuffix(".json") for p in d.glob("*.json.gz")}
    assert len(idx) == 50 and set(idx) == gz                               # index matches files on disk


def test_ws_load_run_rejects_id_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "load_run", "model": "m/x", "id": "../../etc/passwd"})
        assert ws.receive_json()["type"] == "error"


def test_ws_stats_reports_rss_models_regions():
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "stats"})
        msg = ws.receive_json()
    assert msg["type"] == "stats"
    assert msg["rss_mb"] > 0 and isinstance(msg["models"], int) and msg["regions_cached"] == 0


def test_ws_op_error_is_tagged_and_connection_survives():
    # unified errors: an op that raises must yield error{op, reason} AND the same connection
    # must still serve the next op (one op's exception can't kill the kernel link).
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "drilldown", "layer": 0})  # no generation yet → last_raw is None → raises
        err = ws.receive_json()
        assert err["type"] == "error" and err["op"] == "drilldown" and err["reason"]
        ws.send_json({"type": "tensors"})                # same connection still works
        ok = ws.receive_json()
        assert ok["type"] == "tensors"


def test_ws_intervene_spot_streams_locate_progress_then_intervened():
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    ex = ["alpha", "beta", "gamma"]
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "intervene",
                      "region": {"kind": "spot", "examples": ex, "topk": 0.5}, "op": "zero"})
        progress, final = [], None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "intervened":
                final = msg
                break
            progress.append(msg)
    assert [p["type"] for p in progress] == ["locate_progress"] * 3
    assert [p["i"] for p in progress] == [1, 2, 3]
    assert all(p["op"] == "intervene" and p["total"] == 3 for p in progress)
    assert final["type"] == "intervened"


def test_ws_auth_gate_rejects_missing_auth_frame(monkeypatch):
    # token set → first frame must be a matching auth frame; anything else closes with 4401.
    monkeypatch.setenv("PARAMETIC_STUDIO_TOKEN", "secret")
    api.SESSION = _tiny_session()
    with pytest.raises(WebSocketDisconnect):
        with TestClient(api.app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "catalog"})  # not an auth frame
            ws.receive_json()  # kernel closes the socket → disconnect


def test_ws_auth_gate_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_TOKEN", "secret")
    api.SESSION = _tiny_session()
    with pytest.raises(WebSocketDisconnect):
        with TestClient(api.app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "token": "wrong"})
            ws.receive_json()


def test_ws_auth_gate_accepts_matching_token(monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_TOKEN", "secret")
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": "secret"})  # gate passes, no reply
        ws.send_json({"type": "catalog"})
        msg = ws.receive_json()
    assert msg["type"] == "catalog"  # normal op flows after auth


def test_ws_no_token_ignores_stray_auth_frame():
    # token unset (default localhost) → no gate, and a stray auth frame is a no-op, not an error.
    assert "PARAMETIC_STUDIO_TOKEN" not in os.environ
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": "x"})  # ignored — client always sends auth first
        ws.send_json({"type": "catalog"})
        msg = ws.receive_json()
    assert msg["type"] == "catalog"


def test_ws_save_region_spot_streams_locate_progress_then_saved():
    api.SESSION = _tiny_session()
    api.SESSIONS.clear()
    ex = ["alpha", "beta"]
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "save_region", "name": "r1",
                      "region": {"kind": "spot", "examples": ex, "topk": 0.5}})
        progress, final = [], None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "region_saved":
                final = msg
                break
            progress.append(msg)
    assert [p["type"] for p in progress] == ["locate_progress"] * 2
    assert [p["i"] for p in progress] == [1, 2]
    assert all(p["op"] == "save_region" and p["total"] == 2 for p in progress)
    assert final["type"] == "region_saved" and final["name"] == "r1" and final["count"] > 0
