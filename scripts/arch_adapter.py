#!/usr/bin/env python3
"""Architecture adapter for the spot scripts.

The grad*param accumulation step
(`training/further_training/accumulate_grad_mul_params-10000.py`) saves one tensor
per model parameter, named `<hf_param_name>.pt` (the raw `model.named_parameters()`
name, with any DataParallel `module.` prefix stripped). The spot scripts then walk
those `.pt` files and need to know, for each one, which transformer block it belongs
to and which module inside the block.

Historically every script hardcoded llama/qwen's `model.layers.N.` layout. This
module generalizes that to the per-layer block prefix used across common HF decoder
architectures so a newly-registered HF model works without per-arch code:

    model.layers.N.              llama, qwen, mistral, gemma, phi3
    model.decoder.layers.N.      opt
    transformer.h.N.             gpt2, gptj, falcon
    gpt_neox.layers.N.           pythia / gpt-neox
    transformer.blocks.N.        mpt

All helpers take a *bare* HF parameter name (no trailing `.pt`); use `param_name()`
to strip the suffix from a path, or the `iter_layer_*` glob helpers which do it for
you.

Note: this is intentionally limited to identifying (layer_idx, module). The
human-facing module *labels/ordering* (MODULE_ORDER/MODULE_LABELS in the plot
scripts) stay llama/qwen-flavored and fall back gracefully to the raw module string
for unknown architectures — that is cosmetic and out of scope here.
"""
from pathlib import Path

import re

# The numbered transformer block appears as `<block>.<int>.` where <block> is one of
# layers / h / blocks. We anchor on a preceding start-or-dot so we don't match a bare
# substring, capture the integer index, and treat everything after it as the module.
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.(.+)$")


def param_name(path):
    """Return the bare HF parameter name for a `<name>.pt` path (strips one suffix)."""
    return Path(path).stem


def parse_param(name):
    """Map a bare HF parameter name to ``(layer_idx, module)``.

    Returns ``None`` for parameters that do not live inside a numbered transformer
    block (embeddings, final norm, lm_head, ...).
    """
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def is_target(name):
    """True iff ``name`` is a parameter inside a numbered transformer block."""
    return parse_param(name) is not None


def layer_pt_files(checkpoint_dir):
    """Sorted `.pt` files in ``checkpoint_dir`` that belong to a transformer block."""
    return [path for path in sorted(Path(checkpoint_dir).glob("*.pt")) if is_target(path.stem)]
