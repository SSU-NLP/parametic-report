#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create intersection masks from two boolean .pt mask directories.")
    parser.add_argument("--mask-a", required=True)
    parser.add_argument("--mask-b", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    mask_a = Path(args.mask_a).resolve()
    mask_b = Path(args.mask_b).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = sorted(mask_a.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {mask_a}")

    total = 0
    count_a = 0
    count_b = 0
    intersection = 0
    written = 0

    for path_a in tqdm(files, desc="intersecting", dynamic_ncols=True):
        path_b = mask_b / path_a.name
        if not path_b.exists():
            raise FileNotFoundError(path_b)

        a = torch.load(path_a, map_location="cpu").bool()
        b = torch.load(path_b, map_location="cpu").bool()
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"Shape mismatch for {path_a.name}: {tuple(a.shape)} vs {tuple(b.shape)}")

        merged = torch.logical_and(a, b)
        total += a.numel()
        count_a += int(a.sum().item())
        count_b += int(b.sum().item())
        intersection += int(merged.sum().item())
        torch.save(merged, output / path_a.name)
        written += 1

    summary = {
        "mask_a": str(mask_a),
        "mask_b": str(mask_b),
        "output": str(output),
        "tensors": written,
        "total_parameters": total,
        "mask_a_parameters": count_a,
        "mask_b_parameters": count_b,
        "intersection_parameters": intersection,
        "intersection_ratio_total": intersection / total if total else 0.0,
        "intersection_ratio_mask_a": intersection / count_a if count_a else 0.0,
        "intersection_ratio_mask_b": intersection / count_b if count_b else 0.0,
    }
    with (output.parent / f"{output.name}.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
