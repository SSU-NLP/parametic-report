#!/usr/bin/env python3
import argparse
import gc
import hashlib
from pathlib import Path

import torch


def stable_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def save_random_mask(reference_mask, output_path, seed, name):
    count = int(reference_mask.sum().item())
    numel = reference_mask.numel()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, name))
    indices = torch.randperm(numel, generator=generator)[:count]
    mask = torch.zeros(numel, dtype=torch.bool)
    mask[indices] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.reshape(reference_mask.shape), output_path)


def save_bottom_mask(reference_mask, importance_path, output_path):
    count = int(reference_mask.sum().item())
    importance = torch.load(importance_path, map_location="cpu").abs().float()
    if importance.shape != reference_mask.shape:
        raise ValueError(f"Shape mismatch for {importance_path.name}: {importance.shape} != {reference_mask.shape}")
    indices = importance.reshape(-1).topk(count, largest=False).indices
    mask = torch.zeros(importance.numel(), dtype=torch.bool)
    mask[indices] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.reshape(importance.shape), output_path)


def main():
    parser = argparse.ArgumentParser(description="Create random and bottom-k control masks matched to a code mask.")
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--importance-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--k-label", default="top0.01")
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-bottom", action="store_true")
    args = parser.parse_args()

    mask_files = sorted(args.reference_mask.glob("*.pt"))
    if not mask_files:
        raise SystemExit(f"No .pt mask files found in {args.reference_mask}")

    total = len(mask_files)
    for idx, mask_path in enumerate(mask_files, 1):
        reference_mask = torch.load(mask_path, map_location="cpu").bool()
        for seed in args.random_seeds:
            output_path = args.output_root / f"random_seed{seed}" / args.k_label / mask_path.name
            save_random_mask(reference_mask, output_path, seed, mask_path.name)
        if not args.skip_bottom:
            importance_path = args.importance_checkpoint / mask_path.name
            if not importance_path.exists():
                raise FileNotFoundError(importance_path)
            output_path = args.output_root / "bottom" / args.k_label / mask_path.name
            save_bottom_mask(reference_mask, importance_path, output_path)

        if idx % 25 == 0 or idx == total:
            print(f"processed {idx}/{total}", flush=True)
        del reference_mask
        gc.collect()

    print(f"wrote controls under {args.output_root}")


if __name__ == "__main__":
    main()
