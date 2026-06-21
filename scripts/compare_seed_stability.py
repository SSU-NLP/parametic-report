#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import torch

from arch_adapter import is_target


def format_k(k):
    return f"{k:g}"


def parse_tensor_name(path):
    name = path.name
    stem = name[:-3] if name.endswith(".pt") else path.stem
    parts = stem.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2]), ".".join(parts[3:])
    return "", stem


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return device


def checkpoint_dir(root, seed, language, sample_size):
    return root / f"seed_{seed}" / language / f"grad-mul-param_checkpoint_{sample_size}"


def tensor_files(path, include_non_layer):
    files = sorted(path.glob("*.pt"))
    if not include_non_layer:
        files = [item for item in files if is_target(item.stem)]
    return files


def load_score(path, device):
    tensor = torch.load(path, map_location="cpu")
    shape = tuple(tensor.shape)
    numel = tensor.numel()
    score = tensor.abs().float().reshape(-1).to(device)
    del tensor
    return score, shape, numel


def top_indices(score, max_count):
    if max_count <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    return torch.topk(score, max_count, sorted=True).indices



def selected_indices_by_k(score, counts, max_count, tie_mode):
    if tie_mode == "native":
        top = top_indices(score, max_count)
        return {k: top[:count] for k, count in counts.items()}

    if max_count <= 0:
        empty = torch.empty(0, dtype=torch.long, device=score.device)
        return {k: empty for k in counts}

    top_values = torch.topk(score, max_count, sorted=True).values
    selected = {}
    for k, count in counts.items():
        if count <= 0:
            selected[k] = torch.empty(0, dtype=torch.long, device=score.device)
            continue
        if count >= score.numel():
            selected[k] = torch.arange(score.numel(), dtype=torch.long, device=score.device)
            continue

        threshold = top_values[count - 1]
        above = torch.nonzero(score > threshold, as_tuple=False).flatten()
        needed = count - above.numel()
        if needed <= 0:
            selected[k] = above[:count]
            continue
        equal = torch.nonzero(score == threshold, as_tuple=False).flatten()
        equal = equal.sort().values[:needed]
        selected[k] = torch.cat([above, equal])
    return selected


def index_metrics(indices_a, indices_b, numel, device):
    selected_a = int(indices_a.numel())
    selected_b = int(indices_b.numel())
    if selected_a == 0 or selected_b == 0:
        intersection = 0
    else:
        mask = torch.zeros(numel, dtype=torch.bool, device=device)
        mask[indices_a] = True
        intersection = int(mask[indices_b].sum().item())
        del mask
    union = selected_a + selected_b - intersection
    return {
        "selected_a": selected_a,
        "selected_b": selected_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "overlap_a": intersection / selected_a if selected_a else 0.0,
        "overlap_b": intersection / selected_b if selected_b else 0.0,
    }


def add_aggregate(item, metrics):
    item["selected_a"] += metrics["selected_a"]
    item["selected_b"] += metrics["selected_b"]
    item["intersection"] += metrics["intersection"]
    item["union"] += metrics["union"]
    item["tensor_count"] += 1


def finalize_pair(key, values):
    sample_size, k, seed_a, seed_b = key
    intersection = values["intersection"]
    union = values["union"]
    selected_a = values["selected_a"]
    selected_b = values["selected_b"]
    return {
        "sample_size": sample_size,
        "k": k,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "tensor_count": values["tensor_count"],
        "selected_a": selected_a,
        "selected_b": selected_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "overlap_a": intersection / selected_a if selected_a else 0.0,
        "overlap_b": intersection / selected_b if selected_b else 0.0,
    }


def finalize_module(key, values):
    sample_size, k, seed_a, seed_b, module = key
    row = finalize_pair((sample_size, k, seed_a, seed_b), values)
    row["module"] = module
    return row


def mean(values):
    return sum(values) / len(values) if values else ""


def min_value(values):
    return min(values) if values else ""


def max_value(values):
    return max(values) if values else ""


def compact_summary(pair_rows, sample_sizes, k_values):
    rows = []
    for sample_size in sample_sizes:
        for k in k_values:
            k_label = format_k(k)
            matching = [
                row for row in pair_rows
                if row["sample_size"] == sample_size and row["k"] == k_label
            ]
            jaccards = [row["jaccard"] for row in matching]
            overlaps = [row["overlap_a"] for row in matching]
            rows.append({
                "sample_size": sample_size,
                "k": k_label,
                "tensor_count": matching[0]["tensor_count"] if matching else "",
                "total_selected": matching[0]["selected_a"] if matching else "",
                "stability_jaccard_mean": mean(jaccards),
                "stability_jaccard_min": min_value(jaccards),
                "stability_jaccard_max": max_value(jaccards),
                "stability_overlap_mean": mean(overlaps),
                "stability_overlap_min": min_value(overlaps),
                "stability_overlap_max": max_value(overlaps),
            })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_value(value):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_report(path, args, device, summary_rows):
    lines = [
        "# Seed Stability Report",
        "",
        "This report compares independent small-sample Taylor importance masks.",
        "It does not require a full 10k Taylor checkpoint, so it reports stability only.",
        "",
        "## Inputs",
        "",
        f"- Approx root: `{args.approx_root}`",
        f"- Language: `{args.language}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- Sample sizes: `{', '.join(str(size) for size in args.sample_sizes)}`",
        f"- k values: `{', '.join(format_k(k) for k in args.k_values)}`",
        f"- Device: `{device}`",
        f"- Tie mode: `{args.tie_mode}`",
        "- Tensor scope: all checkpoint tensors" if args.include_non_layer else "- Tensor scope: transformer layer tensors only",
        "",
        "## Summary",
        "",
        "| Samples | k | Stability Jaccard | Stability Overlap |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {sample_size} | {k} | {jaccard} | {overlap} |".format(
                sample_size=row["sample_size"],
                k=row["k"],
                jaccard=markdown_value(row["stability_jaccard_mean"]),
                overlap=markdown_value(row["stability_overlap_mean"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare seed-to-seed stability for grad*param checkpoints.")
    parser.add_argument("--approx-root", required=True, type=Path)
    parser.add_argument("--language", default="java")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 5678])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--k-values", nargs="+", type=float, default=[0.005, 0.01, 0.03, 0.05])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-non-layer", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tie-mode", choices=["native", "stable"], default="stable")
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise SystemExit("At least two seeds are required")
    if not args.approx_root.exists():
        raise SystemExit(f"Approx root does not exist: {args.approx_root}")

    device = resolve_device(args.device)
    k_values = sorted(args.k_values)
    max_k = max(k_values)
    pair_acc = defaultdict(lambda: defaultdict(int))
    module_acc = defaultdict(lambda: defaultdict(int))
    per_tensor_rows = []

    for sample_size in args.sample_sizes:
        first_dir = checkpoint_dir(args.approx_root, args.seeds[0], args.language, sample_size)
        files = tensor_files(first_dir, args.include_non_layer)
        if not files:
            raise SystemExit(f"No checkpoint tensors found in {first_dir}")

        for seed in args.seeds[1:]:
            other_dir = checkpoint_dir(args.approx_root, seed, args.language, sample_size)
            other_files = {path.name for path in tensor_files(other_dir, args.include_non_layer)}
            missing = [path.name for path in files if path.name not in other_files]
            if missing:
                raise FileNotFoundError(f"{other_dir} is missing {missing[0]}")

        for tensor_idx, first_path in enumerate(files, 1):
            layer, module = parse_tensor_name(first_path)
            seed_top = {}
            expected_shape = None
            numel = None
            max_count = None
            counts_by_k = None

            for seed in args.seeds:
                path = checkpoint_dir(args.approx_root, seed, args.language, sample_size) / first_path.name
                if max_count is None:
                    probe = torch.load(path, map_location="cpu")
                    expected_shape = tuple(probe.shape)
                    numel = probe.numel()
                    del probe
                    counts_by_k = {format_k(k): int(k * numel) for k in k_values}
                    max_count = int(max_k * numel)
                score, shape, loaded_numel = load_score(path, device)
                if shape != expected_shape or loaded_numel != numel:
                    raise ValueError(f"Shape mismatch for {first_path.name}: {shape} != {expected_shape}")
                seed_top[seed] = selected_indices_by_k(score, counts_by_k, max_count, args.tie_mode)
                del score

            for k in k_values:
                k_label = format_k(k)
                count = counts_by_k[k_label]
                if count <= 0:
                    continue
                for seed_a, seed_b in itertools.combinations(args.seeds, 2):
                    seed_a_label = f"seed_{seed_a}"
                    seed_b_label = f"seed_{seed_b}"
                    metrics = index_metrics(
                        seed_top[seed_a][k_label],
                        seed_top[seed_b][k_label],
                        numel,
                        device,
                    )
                    row = {
                        "sample_size": sample_size,
                        "k": k_label,
                        "tensor": first_path.name,
                        "layer": layer,
                        "module": module,
                        "seed_a": seed_a_label,
                        "seed_b": seed_b_label,
                        **metrics,
                    }
                    per_tensor_rows.append(row)
                    key = (sample_size, k_label, seed_a_label, seed_b_label)
                    add_aggregate(pair_acc[key], metrics)
                    module_key = (sample_size, k_label, seed_a_label, seed_b_label, module)
                    add_aggregate(module_acc[module_key], metrics)

            del seed_top
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if tensor_idx % args.progress_every == 0 or tensor_idx == len(files):
                print(f"processed sample={sample_size} {tensor_idx}/{len(files)} tensors on {device}", flush=True)

    pair_rows = [finalize_pair(key, values) for key, values in sorted(pair_acc.items())]
    module_rows = [finalize_module(key, values) for key, values in sorted(module_acc.items())]
    summary_rows = compact_summary(pair_rows, args.sample_sizes, k_values)

    summary_fields = [
        "sample_size", "k", "tensor_count", "total_selected",
        "stability_jaccard_mean", "stability_jaccard_min", "stability_jaccard_max",
        "stability_overlap_mean", "stability_overlap_min", "stability_overlap_max",
    ]
    pair_fields = [
        "sample_size", "k", "seed_a", "seed_b", "tensor_count",
        "selected_a", "selected_b", "intersection", "union", "jaccard", "overlap_a", "overlap_b",
    ]
    tensor_fields = ["sample_size", "k", "tensor", "layer", "module", *pair_fields[2:]]
    module_fields = ["sample_size", "k", "module", *pair_fields[2:]]

    write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "seed_pair_summary.csv", pair_rows, pair_fields)
    write_csv(args.output_dir / "per_tensor_metrics.csv", per_tensor_rows, tensor_fields)
    write_csv(args.output_dir / "per_module_metrics.csv", module_rows, module_fields)

    payload = {
        "inputs": {
            "approx_root": str(args.approx_root),
            "language": args.language,
            "seeds": args.seeds,
            "sample_sizes": args.sample_sizes,
            "k_values": [format_k(k) for k in k_values],
            "device": str(device),
            "include_non_layer": args.include_non_layer,
            "tie_mode": args.tie_mode,
        },
        "summary": summary_rows,
        "seed_pair_summary": pair_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.output_dir / "sample_calibration_report.md", args, device, summary_rows)
    print(f"wrote seed stability report to {args.output_dir}")


if __name__ == "__main__":
    main()
