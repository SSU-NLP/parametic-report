"""Unit test for the residual-gate blend helper (scripts/transplant/residual_gate.py).

blend(out, yd, beta) = out + beta*(yd - out): β=0 identity, β=1 full donor, linear between.
Pure arithmetic (works on floats) so the test needs no torch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "transplant"))
import residual_gate as rg  # noqa: E402


def test_blend_identity_and_full():
    assert rg.blend(2.0, 6.0, 0.0) == 2.0     # β=0 → recipient unchanged
    assert rg.blend(2.0, 6.0, 1.0) == 6.0     # β=1 → donor output
    assert rg.blend(2.0, 6.0, 0.5) == 4.0     # midpoint
    assert rg.blend(2.0, 6.0, 0.25) == 3.0
