"""Unit tests for parametic_platform.resolve (pure, no HF network)."""
from __future__ import annotations

import pytest

from parametic_platform import resolve

# Real config.json fields (trimmed) for the two validated models.
LLAMA_3_2_3B = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "tie_word_embeddings": True,
}
QWEN3_8B = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
}


@pytest.mark.parametrize(
    "hf_id,expected",
    [
        ("Qwen/Qwen3-8B", "qwen3-8b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama-3.2-3b-instruct"),
        ("mistralai/Mistral-7B-v0.1", "mistral-7b-v0.1"),
        ("bigscience/bloom", "bloom"),
    ],
)
def test_slugify(hf_id, expected):
    assert resolve.slugify(hf_id) == expected


def test_expected_tensors_matches_catalog_llama():
    # catalog.py hardcodes 254 for llama-3.2-3b — derivation must reproduce it exactly.
    count, estimated = resolve.estimate_expected_tensors(LLAMA_3_2_3B)
    assert count == 254
    assert estimated is False


def test_expected_tensors_matches_catalog_qwen3():
    # catalog.py hardcodes 399 for qwen3-8b.
    count, estimated = resolve.estimate_expected_tensors(QWEN3_8B)
    assert count == 399
    assert estimated is False


def test_unknown_family_tensor_estimate_is_flagged_and_skip_safe():
    cfg = {"model_type": "exaone", "num_hidden_layers": 30, "tie_word_embeddings": False}
    count, estimated = resolve.estimate_expected_tensors(cfg)
    assert estimated is True
    # fallback per-layer (13) biases high so the calibration's completion check
    # cannot fire on a partial run.
    assert count == 30 * 13 + 3


def test_param_estimate_in_expected_range():
    assert 3.0e9 <= resolve.estimate_param_count(LLAMA_3_2_3B) <= 3.6e9
    assert 7.5e9 <= resolve.estimate_param_count(QWEN3_8B) <= 8.7e9


def test_resolve_validated_models_supported():
    for hf_id, cfg in [("meta-llama/Llama-3.2-3B-Instruct", LLAMA_3_2_3B), ("Qwen/Qwen3-8B", QWEN3_8B)]:
        result = resolve.resolve_model(hf_id, config=cfg)
        assert result["compatibility"]["status"] == "supported"
        spec = result["model_spec"]
        assert spec["hf_model_id"] == hf_id
        assert spec["tokenizer_path"] == hf_id
        assert spec["config_path"] == "config.json"
        assert spec["model_output_name"] == hf_id.split("/")[-1]


def test_resolve_supported_unvalidated_family_has_note():
    cfg = dict(LLAMA_3_2_3B, model_type="mistral")
    result = resolve.resolve_model("mistralai/Mistral-7B-v0.1", config=cfg)
    assert result["compatibility"]["status"] == "supported"
    assert any("not itself end-to-end validated" in n for n in result["compatibility"]["notes"])


def test_resolve_non_causal_lm_unsupported():
    cfg = {"model_type": "bert", "architectures": ["BertForMaskedLM"], "hidden_size": 768,
           "num_hidden_layers": 12, "vocab_size": 30522}
    result = resolve.resolve_model("google-bert/bert-base-uncased", config=cfg)
    assert result["compatibility"]["status"] == "unsupported"
    assert result["model_spec"] is None


def test_resolve_oversized_unsupported():
    cfg = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 28672,
        "vocab_size": 128256, "tie_word_embeddings": False,
    }
    result = resolve.resolve_model("meta-llama/Llama-3.1-70B", config=cfg)
    assert result["compatibility"]["status"] == "unsupported"
    assert result["model_spec"] is None
    assert any("exceeds" in n for n in result["compatibility"]["notes"])


def test_resolve_gpt2_lmhead_is_causal_but_needs_review():
    cfg = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "n_embd": 768,
           "n_layer": 12, "n_head": 12, "vocab_size": 50257}
    result = resolve.resolve_model("openai-community/gpt2", config=cfg)
    # LMHeadModel counts as causal; gpt2 is a recognized-but-different-layout family.
    assert result["compatibility"]["status"] == "needs-review"
    assert result["model_spec"] is not None


def test_resolve_uses_injected_fetch_when_no_config():
    calls = {}

    def fake_fetch(hf_id, revision="main", token=None):
        calls["hf_id"] = hf_id
        calls["revision"] = revision
        return LLAMA_3_2_3B

    result = resolve.resolve_model("meta-llama/Llama-3.2-3B-Instruct", "abc123", fetch=fake_fetch)
    assert calls == {"hf_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "abc123"}
    assert result["model_spec"]["revision"] == "abc123"
