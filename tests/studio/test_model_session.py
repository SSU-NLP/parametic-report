import torch
from transformers import LlamaConfig, LlamaForCausalLM

from parametic_studio.kernel.model_session import ModelSession

PROMPT = torch.tensor([[1, 2, 3]])


def _tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"  # required for output_attentions
    return LlamaForCausalLM(cfg).eval()


class _Tok:
    def __init__(self, eos):
        self.eos_token_id = eos

    def decode(self, ids):
        return f"<{ids[0]}>"


def test_yields_max_tokens_when_no_eos():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=5))
    assert [e["step"] for e in evs] == [0, 1, 2, 3, 4]
    assert all("token_id" in e and "text" in e for e in evs)


def test_stop_halts_early():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    g = s.generate(PROMPT, max_tokens=100)
    next(g)
    next(g)
    s.stop()
    assert list(g) == []


def test_eos_stops_before_max():
    # greedy is deterministic (seeded weights) → first token is reproducible.
    first = next(ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu")).generate(PROMPT, max_tokens=5))["token_id"]
    s = ModelSession(_tiny(), _Tok(eos=first), torch.device("cpu"))
    assert list(s.generate(PROMPT, max_tokens=5)) == []


def test_generate_omits_attn_when_not_subscribed():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=2, probes=()))
    assert [e["step"] for e in evs] == [0, 1]
    assert all("attn" not in e for e in evs)


def test_generate_yields_logitlens_frame():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=2, probes=("logitlens",)))
    L = 2
    for e in evs:
        assert "attn" not in e and "act" not in e
        assert len(e["logit"]) == L  # one row per layer
        assert all(isinstance(r["token"], str) and 0 <= r["prob"] <= 1 for r in e["logit"])


def test_generate_yields_activation_frame():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=3, probes=("activation",)))
    L = 2  # tiny model layers
    for e in evs:
        assert "attn" not in e            # not subscribed
        assert e["act"].shape == (L, 2)   # [L, modules: self_attn, mlp]
        assert (e["act"] >= 0).all()      # norms are non-negative


class _EncTok(_Tok):
    def encode(self, text):
        return [1, 2, 3]


def test_compute_spot_grid_shape_and_nonneg():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    spot = s.compute_spot(["alpha", "beta"])
    assert spot["layers"] == 2
    assert len(spot["modules"]) > 0
    assert len(spot["grid"]) == 2                              # one row per layer
    assert all(len(row) == len(spot["modules"]) for row in spot["grid"])
    assert all(v >= 0 for row in spot["grid"] for v in row)    # |grad*param| ≥ 0


def test_generate_text_encodes_prompt():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    evs = list(s.generate_text("hi", max_tokens=3))
    assert [e["step"] for e in evs] == [0, 1, 2]


def test_generate_yields_square_causal_rows():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    rows = []
    for e in s.generate(PROMPT, max_tokens=3):  # PROMPT len 3
        rows += e.get("attn_rows", [])
    L = 2
    # prefill emits P=3 query rows (kv 1,2,3), then 2 decode rows (kv 4,5) → full causal square: row i kv=i+1
    assert [tuple(r.shape) for r in rows] == [(L, 1), (L, 2), (L, 3), (L, 4), (L, 5)]
    assert all(torch.allclose(r.sum(1), torch.ones(L), atol=1e-3) for r in rows)  # each row is a distribution


def test_drilldown_returns_per_head_of_last_step():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    rows = []
    for e in s.generate(PROMPT, max_tokens=3):
        rows += e.get("attn_rows", [])
    ph = s.drilldown(0)
    assert ph.shape == (2, 5)  # last query's per-head [heads, kv]
    assert torch.allclose(ph.mean(0), rows[-1][0], atol=1e-4)  # head-mean == last row, layer 0
