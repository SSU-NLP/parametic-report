"""Regression + generalization tests for scripts/arch_adapter.py.

The adapter replaced per-script hardcoded `model.layers.N.` parsing. These tests
lock in that (1) it produces *identical* results to the old llama/qwen logic on real
llama-3.2-3b and qwen3-8b parameter names, and (2) it generalizes to other common HF
decoder layouts so a newly-registered model parses without per-arch code.
"""
import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import arch_adapter  # noqa: E402


# --- the old, hardcoded logic these scripts used, reproduced verbatim --------------
_OLD_PARAM_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.pt$")


def _old_parse(filename):
    match = _OLD_PARAM_RE.fullmatch(filename)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _old_is_target(filename):
    return filename.startswith("model.layers.")


def _llama_qwen_param_names(n_layers, qwen=False):
    """Representative HF parameter names for llama-3.2-3b / qwen3-8b."""
    per_layer = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
    if qwen:  # qwen3 adds per-head QK norms
        per_layer += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(n_layers):
        for module in per_layer:
            names.append(f"model.layers.{layer}.{module}")
    return names


@pytest.mark.parametrize("qwen", [False, True], ids=["llama", "qwen"])
def test_matches_old_logic_exactly(qwen):
    names = _llama_qwen_param_names(n_layers=4, qwen=qwen)
    for name in names:
        filename = f"{name}.pt"
        # is_target (new, on stem) == old startswith (on filename)
        assert arch_adapter.is_target(name) == _old_is_target(filename), name
        # parse_param (new, on stem) == old regex (on filename)
        assert arch_adapter.parse_param(name) == _old_parse(filename), name


def test_non_layer_params_excluded():
    for name in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        assert arch_adapter.parse_param(name) is None
        assert arch_adapter.is_target(name) is False


@pytest.mark.parametrize(
    "name,expected",
    [
        # llama / qwen / mistral / gemma
        ("model.layers.0.self_attn.q_proj.weight", (0, "self_attn.q_proj.weight")),
        ("model.layers.31.mlp.down_proj.weight", (31, "mlp.down_proj.weight")),
        # opt
        ("model.decoder.layers.5.self_attn.k_proj.weight", (5, "self_attn.k_proj.weight")),
        # gpt2 / gptj / falcon
        ("transformer.h.11.attn.c_attn.weight", (11, "attn.c_attn.weight")),
        ("transformer.h.0.mlp.c_fc.weight", (0, "mlp.c_fc.weight")),
        # gpt-neox / pythia
        ("gpt_neox.layers.7.attention.query_key_value.weight", (7, "attention.query_key_value.weight")),
        # mpt
        ("transformer.blocks.3.attn.Wqkv.weight", (3, "attn.Wqkv.weight")),
    ],
)
def test_generalizes_across_architectures(name, expected):
    assert arch_adapter.parse_param(name) == expected
    assert arch_adapter.is_target(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "transformer.wte.weight",       # gpt2 embeddings
        "transformer.ln_f.weight",      # gpt2 final norm
        "gpt_neox.embed_in.weight",     # neox embeddings
        "model.decoder.embed_tokens.weight",  # opt embeddings
    ],
)
def test_other_arch_non_layer_excluded(name):
    assert arch_adapter.parse_param(name) is None


def test_param_name_strips_single_suffix():
    assert arch_adapter.param_name("model.layers.0.self_attn.q_proj.weight.pt") == (
        "model.layers.0.self_attn.q_proj.weight"
    )


def test_layer_pt_files_filters_and_sorts(tmp_path):
    for name in [
        "model.embed_tokens.weight.pt",     # excluded (non-layer)
        "model.layers.1.mlp.up_proj.weight.pt",
        "model.layers.0.self_attn.q_proj.weight.pt",
        "lm_head.weight.pt",                # excluded
    ]:
        (tmp_path / name).write_bytes(b"")
    files = arch_adapter.layer_pt_files(tmp_path)
    assert [p.name for p in files] == [
        "model.layers.0.self_attn.q_proj.weight.pt",
        "model.layers.1.mlp.up_proj.weight.pt",
    ]
