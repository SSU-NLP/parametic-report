"""TDD spec for scripts/transplant_mapping.py — the parameter-transplant core.

The transplant experiment moves a donor (coder) model's java-spot parameters into a
base model and asks whether *location* is what matters. All the interesting logic is
the per-tensor mapping: given base/coder importance scores + weights, which base
positions get which coder values. That logic is a pure function over 1-D flat tensors
(no GPU, no model load), so it is unit-tested directly here.

Strategies (see plan):
  v1     coder positions, coder values     base[coder_idx] := coder[coder_idx]
  v2     intersection only                 base[coder_idx ∩ base_idx] := coder[...]
  v3a    rank-aligned by |grad·param|      base spot positions <- coder spot values
  v3b    rank-aligned by |weight|
  v3c    index-order
  v3d    base spot positions, random coder pairing (control)
  v3ctrl base spot positions, coder NON-spot values (control)
"""
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import transplant_mapping as tm  # noqa: E402


K = 0.3  # numel=10 -> count=3, easy to reason about by hand

# base importance: ascending -> top-3 spot = {7, 8, 9}
BASE_SCORE = torch.arange(10).float()
# coder importance: descending -> top-3 spot = {0, 1, 2}  (disjoint from base)
CODER_SCORE_DISJOINT = torch.arange(10).flip(0).float()
# coder importance overlapping base: top-3 = {0, 8, 9}  (shares {8,9} with base)
CODER_SCORE_OVERLAP = torch.tensor([5, 0, 0, 0, 0, 0, 0, 0, 9, 8]).float()

# Disjoint value ranges so any transplanted position is detectable: base in [0,9],
# coder in [100,109]. No coder value can accidentally equal a base value.
BASE_W = torch.arange(10).float()
CODER_W = torch.arange(10).float() + 100.0


def changed(out, base):
    """Set of flat positions where `out` differs from `base`."""
    return set(torch.nonzero(out != base, as_tuple=False).flatten().tolist())


def spot(score):
    return set(tm.spot_indices(score, K).tolist())


V3 = ["v3a", "v3b", "v3c", "v3d", "v3ctrl"]


# --- spot_indices reuses the validated stable_top_indices --------------------------

def test_spot_indices_matches_hand_computation():
    assert spot(BASE_SCORE) == {7, 8, 9}
    assert spot(CODER_SCORE_DISJOINT) == {0, 1, 2}
    assert spot(CODER_SCORE_OVERLAP) == {0, 8, 9}


# --- v1: copy coder spot, same coordinates -----------------------------------------

def test_v1_copies_coder_positions_and_values():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v1", name="t")
    assert changed(out, BASE_W) == {0, 1, 2}
    for i in (0, 1, 2):
        assert out[i] == CODER_W[i]
    # untouched positions keep base values
    for i in (3, 4, 5, 6, 7, 8, 9):
        assert out[i] == BASE_W[i]


# --- v2: intersection of the two spots ---------------------------------------------

def test_v2_intersection_overlap():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_OVERLAP, K, "v2", name="t")
    assert changed(out, BASE_W) == {8, 9}  # {7,8,9} ∩ {0,8,9}
    assert out[8] == CODER_W[8]
    assert out[9] == CODER_W[9]


def test_v2_empty_intersection_is_noop():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v2", name="t")
    assert changed(out, BASE_W) == set()
    assert torch.equal(out, BASE_W)


# --- v3*: all write into the BASE spot positions (that is the whole point) ----------

@pytest.mark.parametrize("strategy", V3)
def test_v3_writes_into_base_spot_positions(strategy):
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, strategy, name="t", seed=0)
    assert changed(out, BASE_W) == spot(BASE_SCORE) == {7, 8, 9}


@pytest.mark.parametrize("strategy", V3)
def test_v3_preserves_count(strategy):
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, strategy, name="t", seed=0)
    assert len(changed(out, BASE_W)) == 3


# --- v1 vs v3 is the experiment's key contrast: same count, different positions -----

def test_v1_and_v3_touch_different_positions():
    v1 = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v1", name="t")
    v3 = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3a", name="t")
    assert changed(v1, BASE_W) == {0, 1, 2}
    assert changed(v3, BASE_W) == {7, 8, 9}
    assert len(changed(v1, BASE_W)) == len(changed(v3, BASE_W)) == 3


# --- v3a: most-important base position gets most-important coder value --------------

def test_v3a_rank_aligned_by_importance():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3a", name="t")
    # base spot sorted by base_score desc: 9 (score9), 8 (8), 7 (7)
    # coder spot sorted by coder_score desc: 0 (score9), 1 (8), 2 (7)
    assert out[9] == CODER_W[0]  # 100
    assert out[8] == CODER_W[1]  # 101
    assert out[7] == CODER_W[2]  # 102


# --- v3c: index-order pairing ------------------------------------------------------

def test_v3c_index_order():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3c", name="t")
    # base spot asc: 7,8,9 ; coder spot asc: 0,1,2
    assert out[7] == CODER_W[0]
    assert out[8] == CODER_W[1]
    assert out[9] == CODER_W[2]


# --- v3d: deterministic given seed, still base positions ----------------------------

def test_v3d_deterministic():
    a = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3d", name="t", seed=7)
    b = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3d", name="t", seed=7)
    assert torch.equal(a, b)
    assert changed(a, BASE_W) == {7, 8, 9}
    # transplanted values still come from the coder spot {100,101,102}
    placed = {a[i].item() for i in (7, 8, 9)}
    assert placed <= {100.0, 101.0, 102.0}


def test_v3d_seed_changes_pairing():
    a = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3d", name="t", seed=1)
    b = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3d", name="t", seed=2)
    # different seed -> generally different pairing (allowed to coincide rarely, but
    # with these values the two permutations differ)
    assert not torch.equal(a, b)


# --- v3ctrl: base positions, coder NON-spot values ---------------------------------

def test_v3ctrl_uses_nonspot_values():
    out = tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3ctrl", name="t", seed=0)
    assert changed(out, BASE_W) == {7, 8, 9}
    coder_spot_values = {CODER_W[i].item() for i in (0, 1, 2)}  # {100,101,102}
    placed = {out[i].item() for i in (7, 8, 9)}
    # values come from coder's non-spot region, never the coder spot
    assert placed.isdisjoint(coder_spot_values)


# --- guards ------------------------------------------------------------------------

def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "nope", name="t")


def test_known_strategies_constant():
    assert set(tm.STRATEGIES) == {"v1", "v2", "v3a", "v3b", "v3c", "v3d", "v3ctrl"}


def test_input_not_mutated():
    base_copy = BASE_W.clone()
    tm.transplant_tensor(BASE_W, CODER_W, BASE_SCORE, CODER_SCORE_DISJOINT, K, "v3a", name="t")
    assert torch.equal(BASE_W, base_copy)
