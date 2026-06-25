"""Unit tests for gated_mask (scripts/transplant/gated_delta.py) — compatibility-gated surgery.

Pure tensor logic; torch required (skip if absent on host).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "transplant"))
import gated_delta as gd  # noqa: E402


def _toy():
    # 6 coords. delta and G chosen so descent (-G*delta>0) holds for a known subset.
    G = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    delta = torch.tensor([-1.0, 1.0, -0.001, 1.0, 1.0, -1.0])  # coord2 tiny |delta|
    A = G.abs() * 3            # A>=|G| so rho<=1
    F = torch.ones_like(G)
    return G, A, F, delta


def test_gain_gate_selects_only_descent():
    G, A, F, delta = _toy()
    # -G*delta: [1, 1, 0.002, 2, -0.5, -0.5] -> descent (>0) at idx 0,1,2,3
    m, st = gd.gated_mask(G, A, F, delta, k=1.0, mode="gain")   # k=100% → all eligible
    sel = set(torch.nonzero(m).flatten().tolist())
    assert sel == {0, 1, 2, 3}
    assert st["eligible"] == 4


def test_T_small_delta_bias_visible():
    # coord2 has tiny |delta| → gain/risk = -G/(F*delta) huge → T ranks it top.
    G, A, F, delta = _toy()
    m, st = gd.gated_mask(G, A, F, delta, k=1.0 / 6, mode="T")  # pick top-1 of 6
    assert m.sum().item() == 1
    assert torch.nonzero(m).flatten().item() == 2               # the tiny-delta coord


def test_matched_rand_count_and_predicted_dL():
    G, A, F, delta = _toy()
    m, st = gd.gated_mask(G, A, F, delta, k=0.5, mode="matched-rand", seed=0)
    assert m.sum().item() == 3                                  # 50% of 6, any coords
    # predicted ΔL = -α*gain + 0.5 α² risk ; gain over selected = Σ -G*delta (selected)
    assert "gain" in st and "risk" in st and "abs_delta" in st


def test_descent_rand_only_in_eligible():
    G, A, F, delta = _toy()
    m, st = gd.gated_mask(G, A, F, delta, k=0.5, mode="descent-rand", seed=1)
    sel = set(torch.nonzero(m).flatten().tolist())
    assert sel.issubset({0, 1, 2, 3})                           # never picks non-descent


def test_saliency_signed_abs_intersection_difference():
    # theta picks coord by |theta|·|G| (signed) vs |theta|·A (abs).
    G = torch.tensor([3.0, 1.0, 1.0, 0.1])
    A = torch.tensor([3.0, 9.0, 1.0, 0.1])   # coord1 high abs-activity, low signed
    F = torch.ones(4)
    delta = torch.tensor([1.0, 1.0, 1.0, 1.0])
    theta = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # s_signed=|θ||G|=[3,1,1,.1] -> top1=coord0 ; s_abs=|θ|A=[3,9,1,.1] -> top1=coord1
    ms, _ = gd.gated_mask(G, A, F, delta, k=1 / 4, mode="signed", theta=theta)
    ma, _ = gd.gated_mask(G, A, F, delta, k=1 / 4, mode="abs", theta=theta)
    assert torch.nonzero(ms).flatten().item() == 0
    assert torch.nonzero(ma).flatten().item() == 1
    # top-2 each: signed={0,1}, abs={1,0}? recompute at k=0.5
    msd, _ = gd.gated_mask(G, A, F, delta, k=0.5, mode="signed-and-abs", theta=theta)
    mns, _ = gd.gated_mask(G, A, F, delta, k=0.5, mode="abs-not-signed", theta=theta)
    sd = set(torch.nonzero(msd).flatten().tolist()); ns = set(torch.nonzero(mns).flatten().tolist())
    assert sd.isdisjoint(ns)                                    # ∩ and ∖ are disjoint
    assert ns.issubset({0, 1, 2, 3})


def test_saliency_needs_theta():
    G, A, F, delta = _toy()
    with pytest.raises(SystemExit):
        gd.gated_mask(G, A, F, delta, k=0.5, mode="abs", theta=None)
