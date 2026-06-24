#!/usr/bin/env python3
"""Build bridge (A∖B) boolean masks per tensor for ONE model, for the Phase C transplant.

  A = |theta| top-kA  (weight magnitude, computed from the model)
  B = code spot       (top-kB by summed |grad·param|, loaded as bool mask from the volume)
  bridge = A ∖ B = A & ~B

Saves one bool .pt per target tensor (filename = param name) into --output-dir, so
scripts/transplant/eval_bridge_cowork.py can consume it via --bridge-base/--bridge-coder.

Reuses top_k_bool from d0_diagnostic (same A∖B definition D0 measured) + arch_adapter.is_target.
CPU-only.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "paper_repro"))
from arch_adapter import is_target          # noqa: E402
from d0_diagnostic import top_k_bool         # noqa: E402  (same top-k definition as the D0 gate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True, help="HF model (A = |theta| top-kA from here)")
    p.add_argument("--spot-dir", required=True, help="dir of code-spot bool masks (B), one .pt per tensor")
    p.add_argument("--ka", type=float, nargs="+", required=True, help="A fraction(s) |theta| top-kA; multiple = one model load, per-kA output")
    p.add_argument("--kb-label", default=None, help="kB label for output subdir name (e.g. 0.0025); required with --output-root")
    p.add_argument("--output-root", type=Path, default=None, help="multi-kA: masks -> <root>/ka<ka>_kb<kb>/")
    p.add_argument("--output-dir", type=Path, default=None, help="single-kA: masks -> here")
    a = p.parse_args()
    if a.output_dir is None and a.output_root is None:
        raise SystemExit("provide --output-dir (single-kA) or --output-root + --kb-label (multi-kA)")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    from transformers import AutoModelForCausalLM
    print(f"loading {a.model_id} (CPU)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.float32, token=token)
    weights = {n: p_.detach() for n, p_ in model.named_parameters()}

    # output dir per kA
    def out_dir(ka):
        if a.output_dir is not None:
            return a.output_dir
        return a.output_root / f"ka{ka}_kb{a.kb_label}"
    for ka in a.ka:
        out_dir(ka).mkdir(parents=True, exist_ok=True)

    names = sorted(q.stem for q in Path(a.spot_dir).glob("*.pt"))
    built = {ka: 0 for ka in a.ka}; empty = {ka: 0 for ka in a.ka}
    for i, name in enumerate(names):
        if name not in weights or not is_target(name):
            continue
        theta = weights[name]
        shape = tuple(theta.shape)
        B = torch.load(Path(a.spot_dir) / f"{name}.pt", map_location="cpu").bool()
        if tuple(B.shape) != shape:
            continue
        wabs = theta.abs()
        for ka in a.ka:
            A = top_k_bool(wabs, ka).reshape(shape)
            bridge = A & ~B
            torch.save(bridge, out_dir(ka) / f"{name}.pt")
            built[ka] += 1
            if int(bridge.sum()) == 0:
                empty[ka] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(names)} (kA={a.ka})", flush=True)
    for ka in a.ka:
        print(f"DONE kA={ka} built={built[ka]} empty-bridge={empty[ka]} -> {out_dir(ka)}", flush=True)


if __name__ == "__main__":
    main()
