#!/usr/bin/env python3
"""Layer/module distribution of spot (B=|grad·param| top-k), weight-top-k (A), and A∩B.

spot is per-tensor top-k, so the *count* is uniform across layers by construction. What
varies is (1) the grad·param *strength* per layer (is java importance concentrated in
late layers?) and (2) how much weight-top-k (A) overlaps grad-top-k (B) per layer — the
A∩B that decides how big the colleague's code-spot (A∖B) is.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arch_adapter import layer_pt_files, parse_param  # noqa: E402
from create_approx_spot_masks import load_mean_score, stable_top_indices  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-scores", nargs="+", type=Path, required=True)
    p.add_argument("--k", type=float, default=0.01)
    a = p.parse_args()
    tok = os.getenv("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=tok, torch_dtype=torch.float32)
    params = dict(model.named_parameters())
    files = layer_pt_files(a.base_scores[0])

    by_layer = defaultdict(lambda: {"grad_sum": 0.0, "numel": 0, "B": 0, "AnB": 0})
    by_module = defaultdict(lambda: {"grad_sum": 0.0, "numel": 0, "B": 0, "AnB": 0})

    with torch.no_grad():
        for f in files:
            pp = parse_param(f.stem)
            if pp is None or f.stem not in params:
                continue
            li, mod = pp
            score, _ = load_mean_score([d / f.name for d in a.base_scores], torch.device("cpu"))
            w = params[f.stem].data.reshape(-1).abs().float()
            cnt = int(a.k * score.numel())
            if cnt == 0:
                continue
            A = stable_top_indices(w, cnt, largest=True)      # weight top-k
            B = stable_top_indices(score, cnt, largest=True)  # grad top-k (spot)
            anb = int(torch.isin(A, B).sum().item())
            for tgt, key in ((by_layer, li), (by_module, mod)):
                d = tgt[key]
                d["grad_sum"] += float(score.sum()); d["numel"] += score.numel()
                d["B"] += cnt; d["AnB"] += anb

    print(f"\n=== per-LAYER ===")
    print(f"{'layer':>5} {'grad·param mean':>16} {'A∩B / B':>9}")
    for li in sorted(by_layer):
        d = by_layer[li]
        gm = d["grad_sum"] / d["numel"]
        ov = d["AnB"] / d["B"] * 100 if d["B"] else 0.0
        print(f"{li:>5} {gm:>16.6f} {ov:>8.1f}%")

    print(f"\n=== per-MODULE ===")
    for mod in sorted(by_module):
        d = by_module[mod]
        gm = d["grad_sum"] / d["numel"]
        ov = d["AnB"] / d["B"] * 100 if d["B"] else 0.0
        print(f"{mod:42} grad_mean={gm:.6f}  A∩B/B={ov:.1f}%")

    tg = sum(d["grad_sum"] for d in by_layer.values()); tn = sum(d["numel"] for d in by_layer.values())
    tab = sum(d["AnB"] for d in by_layer.values()); tb = sum(d["B"] for d in by_layer.values())
    print(f"\nOVERALL grad·param mean={tg/tn:.6f}  A∩B/B={tab/tb*100:.1f}%  (B=spot, A=weight-top-k)")


if __name__ == "__main__":
    main()
