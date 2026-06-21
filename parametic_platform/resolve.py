"""Resolve an arbitrary Hugging Face model id into a platform ModelSpec.

`resolve_model()` fetches only the model's `config.json` (not the weights), detects
the architecture, estimates parameter count for an A100×1 fit check, derives the
catalog ModelSpec fields, and returns a compatibility report:

    supported     — recognized causal-LM family on the validated `model.layers.N`
                    layout that fits A100×1; safe to register and run.
    needs-review  — runnable in principle but unverified (different block layout,
                    borderline size, or a family we estimate but haven't validated).
    unsupported   — not a causal LM, an architecture the spot scripts can't parse,
                    or too large for A100×1 gradient computation.

The network fetch is isolated in `fetch_hf_config()` so the resolution logic stays a
pure function over a config dict (fully unit-testable without HF access).

Why the derived ModelSpec works without per-model files: in the platform calibration
path the only model-specific values (`hf_model_id`, tokenizer) flow through runner
*arguments*, overriding the shared `config.json` — so every registered model can
reuse it as `config_path`. See handoff.md (new-model feature).
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Callable

from .catalog import ModelSpec

# Decoder LM families on the standard `model.layers.N.` layout we treat as supported.
# The grad*param calibration loads them via AutoModelForCausalLM and the arch adapter
# parses their parameter names. llama-3.2-3b and qwen3-8b are validated end-to-end;
# the rest share the same layout and are accepted but flagged in `notes`.
_VALIDATED_TYPES = {"llama", "qwen3"}
_LLAMA_LIKE_TYPES = {
    "llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3",
    "phi3", "phi", "stablelm", "cohere", "olmo", "olmo2", "granite",
    "exaone", "minicpm", "internlm2", "starcoder2",
}
# Recognized causal-LM families that the arch adapter parses but that sit on a
# *different* block layout (transformer.h.N / blocks.N / decoder.layers.N). Runnable
# in principle, never validated here → needs-review.
_OTHER_DECODER_TYPES = {"gpt2", "gptj", "gpt_neox", "falcon", "mpt", "opt", "bloom"}

# Exact named-parameter-tensor count per transformer block, by family. Used to derive
# `expected_tensors` (total .pt files the calibration writes). Only families we are
# confident about are exact; unknown families fall back to a skip-safe UPPER bound so
# the calibration's "already complete" check can never fire on a partial run.
_PER_LAYER_TENSORS = {
    "llama": 9,      # q,k,v,o + gate,up,down + input_ln, post_attn_ln
    "mistral": 9,
    "qwen3": 11,     # llama-like + q_norm, k_norm
}
_PER_LAYER_FALLBACK = 13  # skip-safe overestimate for un-tabulated families

# A100×1 gradient-computation fit, by estimated parameter count (heuristic; qwen3-8b
# at ~8.2B is the validated upper anchor).
_SIZE_SUPPORTED = 9.5e9
_SIZE_REVIEW = 15e9


def slugify(hf_model_id: str) -> str:
    """Derive a filesystem/url-safe model id from an HF repo id.

    `Qwen/Qwen3-8B` -> `qwen3-8b`, `meta-llama/Llama-3.2-3B-Instruct` ->
    `llama-3.2-3b-instruct`. Keeps dots (used in dir names already).
    """
    last = hf_model_id.rstrip("/").split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", last).strip("-._")
    return slug or "model"


def human_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


def estimate_param_count(config: dict[str, Any]) -> int:
    """Rough total parameter count from config dims (weights only; norms negligible)."""
    hidden = int(config.get("hidden_size") or 0)
    layers = int(config.get("num_hidden_layers") or 0)
    vocab = int(config.get("vocab_size") or 0)
    if not (hidden and layers and vocab):
        return 0
    inter = int(config.get("intermediate_size") or 4 * hidden)
    heads = int(config.get("num_attention_heads") or max(hidden // 64, 1))
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))

    q = hidden * heads * head_dim
    kv = 2 * hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    mlp = 3 * hidden * inter
    per_layer = q + kv + o + mlp

    tied = bool(config.get("tie_word_embeddings", True))
    embed = vocab * hidden * (1 if tied else 2)
    return layers * per_layer + embed


def estimate_expected_tensors(config: dict[str, Any]) -> tuple[int, bool]:
    """(expected named-parameter .pt count, is_estimated).

    Exact for tabulated families; a skip-safe overestimate (is_estimated=True)
    otherwise. Non-layer tensors: embed + final norm, plus a separate lm_head when
    embeddings are not tied.
    """
    layers = int(config.get("num_hidden_layers") or 0)
    model_type = (config.get("model_type") or "").lower()
    per_layer = _PER_LAYER_TENSORS.get(model_type)
    estimated = per_layer is None
    if per_layer is None:
        per_layer = _PER_LAYER_FALLBACK
    tied = bool(config.get("tie_word_embeddings", True))
    non_layer = 2 if tied else 3  # embed + final_norm (+ lm_head if untied)
    return layers * per_layer + non_layer, estimated


def assess_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    model_type = (config.get("model_type") or "").lower()
    architectures = list(config.get("architectures") or [])
    # Causal-LM heads appear as *ForCausalLM (llama/qwen/…) or *LMHeadModel (gpt2/gptj).
    is_causal_lm = (
        any(("CausalLM" in arch) or ("LMHeadModel" in arch) for arch in architectures)
        or not architectures
    )
    params = estimate_param_count(config)
    notes: list[str] = []

    if architectures and not is_causal_lm:
        return {
            "status": "unsupported",
            "model_type": model_type,
            "architectures": architectures,
            "estimated_params": params,
            "estimated_params_human": human_params(params),
            "notes": ["Not a causal language model (no *ForCausalLM architecture)."],
        }

    if model_type in _LLAMA_LIKE_TYPES:
        status = "supported"
        if model_type not in _VALIDATED_TYPES:
            notes.append(f"Family '{model_type}' shares the validated model.layers layout but is not itself end-to-end validated.")
    elif model_type in _OTHER_DECODER_TYPES:
        status = "needs-review"
        notes.append(f"'{model_type}' uses a non-standard block layout; parsed by the arch adapter but the calibration pipeline is unverified for it.")
    else:
        status = "needs-review"
        notes.append(f"Unrecognized model_type '{model_type}'; verify the parameter layout before running.")

    if params > _SIZE_REVIEW:
        status = "unsupported"
        notes.append(f"Estimated {human_params(params)} parameters exceeds the A100x1 gradient-fit ceiling (~{human_params(int(_SIZE_REVIEW))}).")
    elif params > _SIZE_SUPPORTED:
        if status == "supported":
            status = "needs-review"
        notes.append(f"Estimated {human_params(params)} parameters is above the validated ~{human_params(int(_SIZE_SUPPORTED))} A100x1 anchor; may OOM.")

    return {
        "status": status,
        "model_type": model_type,
        "architectures": architectures,
        "estimated_params": params,
        "estimated_params_human": human_params(params),
        "notes": notes,
    }


def derive_model_spec(hf_model_id: str, revision: str, config: dict[str, Any]) -> ModelSpec:
    expected_tensors, _ = estimate_expected_tensors(config)
    output_name = hf_model_id.rstrip("/").split("/")[-1]
    return ModelSpec(
        id=slugify(hf_model_id),
        display_name=output_name,
        hf_model_id=hf_model_id,
        config_path="config.json",  # shared base; model-specific values come via runner args
        tokenizer_path=hf_model_id,  # HF tokenizers load by id
        model_output_name=output_name,
        expected_tensors=expected_tensors,
        revision=revision or "main",
    )


def fetch_hf_config(hf_model_id: str, revision: str = "main", token: str | None = None) -> dict[str, Any]:
    """Download just `config.json` for the model. Network; isolated for testability."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=hf_model_id,
        filename="config.json",
        revision=revision or "main",
        token=token,
    )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model(
    hf_model_id: str,
    revision: str = "main",
    *,
    config: dict[str, Any] | None = None,
    fetch: Callable[..., dict[str, Any]] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Resolve an HF model id to {model_spec, compatibility, ...}.

    Pass `config` to skip the network entirely (tests); otherwise `fetch` (default
    `fetch_hf_config`) downloads config.json. A non-causal-LM yields model_spec=None.
    """
    if config is None:
        fetch = fetch or fetch_hf_config
        config = fetch(hf_model_id, revision=revision, token=token)

    compatibility = assess_compatibility(config)
    expected_tensors, tensors_estimated = estimate_expected_tensors(config)
    compatibility["expected_tensors_estimated"] = tensors_estimated

    model_spec = None
    if compatibility["status"] != "unsupported":
        model_spec = asdict(derive_model_spec(hf_model_id, revision, config))

    return {
        "hf_model_id": hf_model_id,
        "revision": revision or "main",
        "compatibility": compatibility,
        "model_spec": model_spec,
    }
