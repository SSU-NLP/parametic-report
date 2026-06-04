#!/usr/bin/env python3
import argparse
import gc
import hashlib
from pathlib import Path

import torch


def stable_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def stable_top_indices(score, count, largest=True):
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    if count >= score.numel():
        return torch.arange(score.numel(), dtype=torch.long, device=score.device)

    values = torch.topk(score, count, largest=largest, sorted=False).values
    threshold = values.min() if largest else values.max()
    if largest:
        primary = torch.nonzero(score > threshold, as_tuple=False).flatten()
        equal = torch.nonzero(score == threshold, as_tuple=False).flatten()
    else:
        primary = torch.nonzero(score < threshold, as_tuple=False).flatten()
        equal = torch.nonzero(score == threshold, as_tuple=False).flatten()
    needed = count - primary.numel()
    if needed <= 0:
        return primary[:count]
    equal = equal.sort().values[:needed]
    return torch.cat([primary, equal])


def load_mean_score(paths, device):
    score = None
    shape = None
    for path in paths:
        tensor = torch.load(path, map_location="cpu")
        if shape is None:
            shape = tuple(tensor.shape)
            score = tensor.abs().float()
        else:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Shape mismatch for {path.name}: {tuple(tensor.shape)} != {shape}")
            score += tensor.abs().float()
        del tensor
    score /= len(paths)
    return score.reshape(-1).to(device), shape


def save_mask(indices, shape, numel, output_path):
    mask = torch.zeros(numel, dtype=torch.bool)
    if indices.numel() > 0:
        mask[indices.cpu()] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.reshape(shape), output_path)


def save_random_mask(count, shape, numel, output_path, seed, name):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, name))
    indices = torch.randperm(numel, generator=generator)[:count]
    save_mask(indices, shape, numel, output_path)


def main():
    parser = argparse.ArgumentParser(description="Create approximate top/bottom/random spot masks from one or more grad*param checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--code-output", required=True, type=Path)
    parser.add_argument("--control-output-root", required=True, type=Path)
    parser.add_argument("--k", type=float, default=0.01)
    parser.add_argument("--k-label", default=None)
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-non-layer", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    k_label = args.k_label or f"top{args.k:g}"
    files = sorted(args.checkpoints[0].glob("*.pt"))
    if not args.include_non_layer:
        files = [path for path in files if path.name.startswith("model.layers.")]
    if not files:
        raise SystemExit(f"No .pt tensors found in {args.checkpoints[0]}")

    total_selected = 0
    for idx, first_path in enumerate(files, 1):
        paths = [checkpoint / first_path.name for checkpoint in args.checkpoints]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        score, shape = load_mean_score(paths, device)
        numel = score.numel()
        count = int(args.k * numel)
        total_selected += count

        top_idx = stable_top_indices(score, count, largest=True)
        save_mask(top_idx, shape, numel, args.code_output / first_path.name)

        bottom_idx = stable_top_indices(score, count, largest=False)
        save_mask(bottom_idx, shape, numel, args.control_output_root / "bottom" / k_label / first_path.name)

        for seed in args.random_seeds:
            save_random_mask(count, shape, numel, args.control_output_root / f"random_seed{seed}" / k_label / first_path.name, seed, first_path.name)

        del score, top_idx, bottom_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        if idx % 25 == 0 or idx == len(files):
            print(f"processed {idx}/{len(files)} tensors selected_total={total_selected}", flush=True)

    print(f"wrote code mask to {args.code_output}")
    print(f"wrote controls under {args.control_output_root}")


if __name__ == "__main__":
    main()
