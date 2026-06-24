"""Unit tests for the module/layer restriction predicate in eval_bridge_cowork.

Pure (no torch/model load) — torch/transformers are lazy-imported inside build()/main(),
so importing the module here only pulls the regex helpers.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "transplant"))
import eval_bridge_cowork as eb  # noqa: E402

L = "model.layers.0."
FFN = [f"{L}mlp.gate_proj.weight", f"{L}mlp.up_proj.weight", "model.layers.23.mlp.down_proj.weight"]
ATTN_W = [f"{L}self_attn.q_proj.weight", f"{L}self_attn.k_proj.weight",
          f"{L}self_attn.v_proj.weight", f"{L}self_attn.o_proj.weight"]
ATTN_B = [f"{L}self_attn.q_proj.bias", f"{L}self_attn.k_proj.bias", f"{L}self_attn.v_proj.bias"]
NORM = [f"{L}input_layernorm.weight", f"{L}post_attention_layernorm.weight"]
NONLAYER = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]


def test_all_keeps_everything():
    for n in FFN + ATTN_W + ATTN_B + NORM + NONLAYER:
        assert eb.keep_tensor(n, "all", None) is True


def test_ffn_only():
    for n in FFN:
        assert eb.keep_tensor(n, "ffn", None) is True
    for n in ATTN_W + ATTN_B + NORM + NONLAYER:
        assert eb.keep_tensor(n, "ffn", None) is False


def test_attn_families():
    for n in ATTN_W + ATTN_B:
        assert eb.keep_tensor(n, "attn", None) is True
    for n in FFN + NORM + NONLAYER:
        assert eb.keep_tensor(n, "attn", None) is False
    # vo
    assert eb.keep_tensor(f"{L}self_attn.v_proj.weight", "attn-vo", None) is True
    assert eb.keep_tensor(f"{L}self_attn.o_proj.weight", "attn-vo", None) is True
    assert eb.keep_tensor(f"{L}self_attn.v_proj.bias", "attn-vo", None) is True
    assert eb.keep_tensor(f"{L}self_attn.q_proj.weight", "attn-vo", None) is False
    # qk
    assert eb.keep_tensor(f"{L}self_attn.q_proj.weight", "attn-qk", None) is True
    assert eb.keep_tensor(f"{L}self_attn.k_proj.bias", "attn-qk", None) is True
    assert eb.keep_tensor(f"{L}self_attn.v_proj.weight", "attn-qk", None) is False
    # qkv
    assert eb.keep_tensor(f"{L}self_attn.v_proj.weight", "attn-qkv", None) is True
    assert eb.keep_tensor(f"{L}self_attn.o_proj.weight", "attn-qkv", None) is False


def test_norm_only_excludes_model_norm():
    for n in NORM:
        assert eb.keep_tensor(n, "norm", None) is True
    assert eb.keep_tensor("model.norm.weight", "norm", None) is False   # non-layer final norm
    for n in FFN + ATTN_W:
        assert eb.keep_tensor(n, "norm", None) is False


def test_layer_filter():
    assert eb.keep_tensor("model.layers.12.mlp.up_proj.weight", "ffn", {12}) is True
    assert eb.keep_tensor("model.layers.12.mlp.up_proj.weight", "ffn", {0, 1, 2}) is False
    assert eb.keep_tensor("model.embed_tokens.weight", "all", {0}) is False   # no layer idx


def test_parse_layers():
    assert eb._parse_layers("all") is None
    assert eb._parse_layers("12") == {12}
    assert eb._parse_layers("0-5") == {0, 1, 2, 3, 4, 5}
    assert eb._parse_layers("0-3,12") == {0, 1, 2, 3, 12}


def test_unknown_module_raises():
    with pytest.raises(SystemExit):
        eb.keep_tensor(f"{L}mlp.up_proj.weight", "bogus", None)
