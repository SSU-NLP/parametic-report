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


def test_ws_spot_request_returns_grid():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "spot", "examples": ["alpha", "beta"]})
        msg = ws.receive_json()
    assert msg["type"] == "spotmap"
    assert msg["layers"] == 2
    assert len(msg["grid"]) == 2
    assert all(len(row) == len(msg["modules"]) for row in msg["grid"])


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


class _Stoppable:
    def __init__(self):
        self._stop = False

    def stop(self):
        self._stop = True

    def generate_text(self, prompt, max_tokens, probes=("attention",)):
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
