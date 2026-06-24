"""Unit tests for D0 diagnostic pure helpers (no torch model load)."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "d0_diagnostic",
    Path(__file__).resolve().parents[1] / "scripts" / "paper_repro" / "d0_diagnostic.py",
)
d0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d0)


def test_top_k_bool_count_and_selection():
    s = torch.tensor([0.1, 0.9, 0.5, 0.8, 0.2])
    m = d0.top_k_bool(s, 0.4)  # 40% of 5 = 2
    assert int(m.sum()) == 2
    assert m[1] and m[3]  # the two largest


def test_top_k_bool_zero():
    s = torch.arange(10).float()
    assert int(d0.top_k_bool(s, 0.0).sum()) == 0


def test_matched_null_excludes_and_counts():
    wa = torch.arange(100).float()
    excl = torch.zeros(100, dtype=torch.bool); excl[:10] = True
    null = d0.matched_null_bool(wa, excl, count=10, seed=1)
    assert (null & excl).sum() == 0          # never overlaps excluded
    assert 1 <= int(null.sum()) <= 12         # ~count (binning rounding)


def test_enrichment_ratio():
    delta = torch.tensor([10.0, 10.0, 1.0, 1.0])
    hi = torch.tensor([True, True, False, False])
    lo = torch.tensor([False, False, True, True])
    assert abs(d0.enrichment(delta, hi, lo) - 10.0) < 1e-6


def test_enrichment_empty_none():
    delta = torch.ones(4)
    assert d0.enrichment(delta, torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)) is None


def test_module_of():
    assert d0.module_of("model.layers.0.self_attn.q_proj.weight") == "attn"
    assert d0.module_of("model.layers.0.mlp.down_proj.weight") == "mlp"
    assert d0.module_of("model.layers.0.input_layernorm.weight") == "norm"
    assert d0.module_of("model.embed_tokens.weight") == "embed"


def test_spearman_monotonic():
    a = torch.tensor([1.0, 2, 3, 4, 5])
    assert d0.spearman(a, a * 2) > 0.99           # perfectly monotonic
    assert d0.spearman(a, -a) < -0.99
