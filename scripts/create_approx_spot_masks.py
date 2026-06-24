#!/usr/bin/env python3
import argparse
import gc
import hashlib
from pathlib import Path

import torch

from arch_adapter import is_target


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
    parser.add_argument("--code-output", type=Path, default=None, help="single-k: dir for the top mask")
    parser.add_argument("--code-output-root", type=Path, default=None, help="multi-k: masks go under <root>/top<k>/")
    parser.add_argument("--control-output-root", required=True, type=Path)
    parser.add_argument("--k", type=float, default=0.01)
    parser.add_argument("--ks", nargs="+", default=None,
                        help="multi-k list (literal strings, used verbatim in top<k> labels to avoid {:g} exponent form); each tensor read ONCE, reused for every k")
    parser.add_argument("--k-label", default=None)
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-non-layer", action="store_true")
    args = parser.parse_args()

    if args.code_output is None and args.code_output_root is None:
        raise SystemExit("provide --code-output (single-k) or --code-output-root (multi-k)")
    # keep literal k strings for labels (f"{k:g}" turns 0.000025 into "2.5e-05" → label mismatch)
    ks_strs = args.ks if args.ks else [str(args.k)]

    device = resolve_device(args.device)
    # Union of tensor names across all checkpoints. deepspeed occasionally yields a None grad for
    # a parameter, so accumulate skips it — per-language tensor sets differ slightly. Take every
    # tensor present in >=1 checkpoint and average only over the ones that have it.
    names = set()
    for ck in args.checkpoints:
        for p in ck.glob("*.pt"):
            if args.include_non_layer or is_target(p.stem):
                names.add(p.name)
    files = sorted(names)
    if not files:
        raise SystemExit(f"No target .pt tensors found across {args.checkpoints}")

    for idx, fname in enumerate(files, 1):
        paths = [ck / fname for ck in args.checkpoints if (ck / fname).exists()]
        if not paths:
            continue
        score, shape = load_mean_score(paths, device)  # mean over languages that have it; read once, reused for all k
        numel = score.numel()
        for ks_str in ks_strs:
            k = float(ks_str)
            k_label = args.k_label if (args.k_label and len(ks_strs) == 1) else f"top{ks_str}"
            count = int(k * numel)
            code_path = (args.code_output / fname) if args.code_output \
                else (args.code_output_root / k_label / fname)
            save_mask(stable_top_indices(score, count, largest=True), shape, numel, code_path)
            save_mask(stable_top_indices(score, count, largest=False), shape, numel,
                      args.control_output_root / "bottom" / k_label / fname)
            for seed in args.random_seeds:
                save_random_mask(count, shape, numel,
                                 args.control_output_root / f"random_seed{seed}" / k_label / fname,
                                 seed, fname)
        del score
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        if idx % 25 == 0 or idx == len(files):
            print(f"processed {idx}/{len(files)} tensors (ks={ks_strs})", flush=True)

    print(f"wrote masks (ks={ks_strs}) code-> {args.code_output or args.code_output_root}, controls-> {args.control_output_root}", flush=True)


if __name__ == "__main__":
    main()
