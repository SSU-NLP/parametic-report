#!/usr/bin/env python3
"""Spot-location overlap between two calibration score sets of the same architecture.

Answers "do two models put their java spot in the same parameter positions?" — e.g.
base vs base-instruct, or base vs coder. Reuses the exact spot definition
(stable_top_indices on mean |grad·param|) from the mask pipeline, so the overlap is
over the same top-k% the transplant uses.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/

from arch_adapter import layer_pt_files  # noqa: E402
from create_approx_spot_masks import load_mean_score  # noqa: E402
from transplant_mapping import spot_indices  # noqa: E402


def model_spots(score_dirs, k):
    files = layer_pt_files(score_dirs[0])
    spots = {}
    for f in files:
        score, _ = load_mean_score([d / f.name for d in score_dirs], torch.device("cpu"))
        spots[f.stem] = set(spot_indices(score, k).tolist())
    return spots


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", nargs="+", type=Path, required=True)
    p.add_argument("--b", nargs="+", type=Path, required=True)
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--label", default="A vs B")
    args = p.parse_args()

    A = model_spots(args.a, args.k)
    B = model_spots(args.b, args.k)
    names = [n for n in A if n in B]

    tot_a = tot_int = 0
    per = []
    for n in names:
        inter = len(A[n] & B[n])
        per.append((n, inter / len(A[n]) * 100 if A[n] else 0.0))
        tot_a += len(A[n])
        tot_int += inter

    print(f"\n=== {args.label}  (k={args.k}, {len(names)} tensors) ===", flush=True)
    print(f"OVERALL spot overlap: {tot_int / tot_a * 100:.1f}%  ({tot_int}/{tot_a} positions)", flush=True)
    per.sort(key=lambda x: x[1])
    print(f"  min  {per[0][0]}: {per[0][1]:.1f}%", flush=True)
    print(f"  max  {per[-1][0]}: {per[-1][1]:.1f}%", flush=True)
    mid = per[len(per) // 2]
    print(f"  median tensor {mid[0]}: {mid[1]:.1f}%", flush=True)
    # random-baseline reference: for top-k%, two independent models would overlap ~k
    print(f"  (random baseline ≈ {args.k * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
