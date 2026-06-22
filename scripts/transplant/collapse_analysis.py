#!/usr/bin/env python3
"""Why does a 1% spot transplant collapse the model? Inspect the weight perturbation.

For each target tensor we reconstruct base/coder spots and apply each strategy, then
measure — at the touched positions — how big the change is and whether the transplanted
value's scale matches what was there. The hypothesis: v3 (in-tensor *re-arrangement*)
puts a value drawn from one structural position into a *different* one, breaking the
weight's scale structure and blowing up activations, whereas v1 (same coordinate) at
least preserves position so the change is smaller.

No GPU: loads the two models on CPU, scores from /shared.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/

from arch_adapter import layer_pt_files  # noqa: E402
from create_approx_spot_masks import load_mean_score  # noqa: E402
from transplant_mapping import transplant_tensor  # noqa: E402

STRATS = ["v1", "v2", "v3a"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--base-scores", nargs="+", type=Path, required=True)
    p.add_argument("--donor-scores", nargs="+", type=Path, required=True)
    p.add_argument("--k", type=float, default=0.01)
    args = p.parse_args()
    tok = os.getenv("HF_TOKEN")

    print(f"loading {args.base_model} / {args.donor_model} (cpu, fp32)", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, token=tok, torch_dtype=torch.float32)
    coder = AutoModelForCausalLM.from_pretrained(args.donor_model, token=tok, torch_dtype=torch.float32)
    bp = dict(base.named_parameters())
    cp = dict(coder.named_parameters())

    files = layer_pt_files(args.base_scores[0])
    # accumulators per strategy
    agg = {s: {"n": 0, "abs_old": 0.0, "abs_new": 0.0, "abs_delta": 0.0,
               "max_delta": 0.0, "blowup": 0.0} for s in STRATS}

    with torch.no_grad():
        for idx, sp in enumerate(files, 1):
            name = sp.stem
            if name not in bp or name not in cp:
                continue
            bw = bp[name].data.reshape(-1).float()
            cw = cp[name].data.reshape(-1).float()
            bs, _ = load_mean_score([d / sp.name for d in args.base_scores], torch.device("cpu"))
            cs, _ = load_mean_score([d / sp.name for d in args.donor_scores], torch.device("cpu"))
            pre_max = bw.abs().max().item()
            for s in STRATS:
                out = transplant_tensor(bw, cw, bs, cs, args.k, s, name=name)
                touched = (out != bw).nonzero(as_tuple=False).flatten()
                if touched.numel() == 0:
                    continue
                old = bw[touched]
                new = out[touched]
                delta = (new - old).abs()
                a = agg[s]
                a["n"] += touched.numel()
                a["abs_old"] += old.abs().sum().item()
                a["abs_new"] += new.abs().sum().item()
                a["abs_delta"] += delta.sum().item()
                a["max_delta"] = max(a["max_delta"], delta.max().item())
                post_max = out.abs().max().item()
                a["blowup"] = max(a["blowup"], post_max / pre_max if pre_max else 0.0)
            if idx % 50 == 0:
                print(f"  {idx}/{len(files)} tensors", flush=True)

    print(f"\n=== collapse analysis (k={args.k}) ===", flush=True)
    print(f"{'strat':6} {'mean|old|':>10} {'mean|new|':>10} {'mean|Δ|':>10} {'max|Δ|':>10} {'newscale/old':>13} {'max blowup':>11}")
    for s in STRATS:
        a = agg[s]
        n = max(a["n"], 1)
        mo, mn, md = a["abs_old"] / n, a["abs_new"] / n, a["abs_delta"] / n
        ratio = mn / mo if mo else float("inf")
        print(f"{s:6} {mo:>10.5f} {mn:>10.5f} {md:>10.5f} {a['max_delta']:>10.3f} {ratio:>13.2f} {a['blowup']:>11.1f}x")


if __name__ == "__main__":
    main()
