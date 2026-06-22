#!/usr/bin/env python3
"""Parameter-transplant core: which base positions receive which coder values.

The transplant experiment moves a donor (coder) model's java-spot parameters into a
base model. Both models share transformer-block tensor shapes, and the spot is the
top-k% of ``|grad·param|`` per tensor, so for any tensor the base spot and coder spot
have the *same* count ``N = int(k·numel)`` — a 1:1 mapping always exists.

This module is the pure, GPU-free heart of the experiment: given a single tensor's
base/coder weights and importance scores, it returns the transplanted base weights for
a chosen strategy. ``stable_top_indices`` / ``stable_seed`` are reused from
``create_approx_spot_masks`` so the spot definition is identical to the mask pipeline.

Strategies::

    v1     coder positions, coder values   base[coder_idx]            := coder[coder_idx]
    v2     intersection only               base[coder_idx ∩ base_idx] := coder[...]
    v3a    base spot <- coder values, rank-aligned by |grad·param|
    v3b    base spot <- coder values, rank-aligned by |weight|
    v3c    base spot <- coder values, index-order
    v3d    base spot <- coder values, random pairing (control)
    v3ctrl base spot <- coder NON-spot values (control)

v1 vs v3* is the key contrast: same number of edited positions, but v1 writes at the
coder's coordinates (arbitrary w.r.t. base) while v3* writes at the base spot.
"""
import torch

from create_approx_spot_masks import stable_seed, stable_top_indices

STRATEGIES = ("v1", "v2", "v3a", "v3b", "v3c", "v3d", "v3ctrl")


def spot_indices(score, k):
    """Flat indices of the top-k% spot for a 1-D ``score`` (== the mask pipeline)."""
    score = score.reshape(-1)
    count = int(k * score.numel())
    return stable_top_indices(score, count, largest=True)


def _gen(seed, name):
    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, name))
    return generator


def _sort_by(indices, keys, descending):
    """Reorder ``indices`` by ``keys`` (stable, so the result is deterministic)."""
    order = torch.argsort(keys, descending=descending, stable=True)
    return indices[order]


def _complement(indices, numel):
    """Flat indices NOT in ``indices`` (sorted ascending)."""
    mask = torch.ones(numel, dtype=torch.bool)
    mask[indices] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def transplant_tensor(base_w, coder_w, base_score, coder_score, k, strategy, *, name="", seed=0):
    """Return a transplanted copy of ``base_w`` (1-D flat) for ``strategy``.

    ``base_w``/``coder_w`` are the two models' weights for one tensor; ``base_score``/
    ``coder_score`` are their ``|grad·param|`` importances. Inputs are not mutated.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r} (expected one of {STRATEGIES})")

    base_w = base_w.reshape(-1)
    coder_w = coder_w.reshape(-1)
    base_score = base_score.reshape(-1)
    coder_score = coder_score.reshape(-1)
    out = base_w.clone()

    numel = base_w.numel()
    count = int(k * numel)
    if count <= 0:
        return out

    base_idx = stable_top_indices(base_score, count, largest=True)
    coder_idx = stable_top_indices(coder_score, count, largest=True)

    if strategy == "v1":
        out[coder_idx] = coder_w[coder_idx]
        return out

    if strategy == "v2":
        inter = base_idx[torch.isin(base_idx, coder_idx)]
        out[inter] = coder_w[inter]
        return out

    # v3*: every position written is a BASE spot position (dst); strategies differ only
    # in which coder value (src) each base position receives.
    if strategy == "v3a":
        dst = _sort_by(base_idx, base_score[base_idx], descending=True)
        src = _sort_by(coder_idx, coder_score[coder_idx], descending=True)
    elif strategy == "v3b":
        dst = _sort_by(base_idx, base_w[base_idx].abs().float(), descending=True)
        src = _sort_by(coder_idx, coder_w[coder_idx].abs().float(), descending=True)
    elif strategy == "v3c":
        dst = base_idx.sort().values
        src = coder_idx.sort().values
    elif strategy == "v3d":
        dst = base_idx.sort().values
        src = coder_idx[torch.randperm(count, generator=_gen(seed, name))]
    else:  # v3ctrl
        dst = base_idx.sort().values
        nonspot = _complement(coder_idx, numel)
        pick = torch.randperm(nonspot.numel(), generator=_gen(seed, name))[:count]
        src = nonspot[pick]

    out[dst] = coder_w[src]
    return out


def transplant_param(base_param, coder_param, base_score, coder_score, k, strategy, *, name, seed=0):
    """Transplant one named parameter (n-D), returning a new tensor of the same shape.

    Convenience wrapper over :func:`transplant_tensor` that preserves shape/dtype — the
    per-model driver calls this per target tensor.
    """
    shape = base_param.shape
    out = transplant_tensor(
        base_param.reshape(-1),
        coder_param.reshape(-1),
        base_score,
        coder_score,
        k,
        strategy,
        name=name,
        seed=seed,
    )
    return out.reshape(shape)
