#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def format_k(k):
    return f"{k:g}"


def parse_tensor_name(path):
    name = path.name
    stem = name[:-3] if name.endswith(".pt") else path.stem
    parts = stem.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2]), ".".join(parts[3:])
    return "", stem


def load_score(path):
    tensor = torch.load(path, map_location="cpu")
    shape = tuple(tensor.shape)
    score = tensor.abs().float().reshape(-1)
    return score, shape


def top_indices(score, max_count):
    if max_count <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.topk(score, max_count, sorted=True).indices.cpu()


def make_mask(numel, indices, count):
    mask = torch.zeros(numel, dtype=torch.bool)
    if count > 0:
        mask[indices[:count]] = True
    return mask


def mask_metrics(mask_a, mask_b):
    selected_a = int(mask_a.sum().item())
    selected_b = int(mask_b.sum().item())
    intersection = int(torch.logical_and(mask_a, mask_b).sum().item())
    union = selected_a + selected_b - intersection
    return {
        "selected_a": selected_a,
        "selected_b": selected_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "recall_b": intersection / selected_b if selected_b else 0.0,
        "overlap_a": intersection / selected_a if selected_a else 0.0,
    }


def add_aggregate(item, metrics):
    item["selected_a"] += metrics["selected_a"]
    item["selected_b"] += metrics["selected_b"]
    item["intersection"] += metrics["intersection"]
    item["union"] += metrics["union"]
    item["tensor_count"] += 1


def finalize_aggregate(key, values):
    sample_size, k, comparison, seed_a, seed_b = key
    intersection = values["intersection"]
    union = values["union"]
    selected_a = values["selected_a"]
    selected_b = values["selected_b"]
    return {
        "sample_size": sample_size,
        "k": k,
        "comparison": comparison,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "tensor_count": values["tensor_count"],
        "selected_a": selected_a,
        "selected_b": selected_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 0.0,
        "recall_b": intersection / selected_b if selected_b else 0.0,
        "overlap_a": intersection / selected_a if selected_a else 0.0,
    }


def finalize_module_aggregate(key, values):
    sample_size, k, comparison, seed_a, seed_b, module = key
    row = finalize_aggregate((sample_size, k, comparison, seed_a, seed_b), values)
    row["module"] = module
    return row


def mean(values):
    return sum(values) / len(values) if values else ""


def min_value(values):
    return min(values) if values else ""


def max_value(values):
    return max(values) if values else ""


def build_compact_summary(comparison_rows, sample_sizes, k_values):
    rows = []
    for sample_size in sample_sizes:
        for k in k_values:
            k_label = format_k(k)
            matching = [
                row for row in comparison_rows
                if row["sample_size"] == sample_size and row["k"] == k_label
            ]
            stability = [row for row in matching if row["comparison"] == "seed_vs_seed"]
            full = [row for row in matching if row["comparison"] == "seed_vs_full"]
            stability_j = [row["jaccard"] for row in stability]
            stability_overlap = [row["overlap_a"] for row in stability]
            full_recall = [row["recall_b"] for row in full]
            full_j = [row["jaccard"] for row in full]
            selected_total = full[0]["selected_b"] if full else (stability[0]["selected_b"] if stability else "")
            tensor_count = full[0]["tensor_count"] if full else (stability[0]["tensor_count"] if stability else "")
            rows.append({
                "sample_size": sample_size,
                "k": k_label,
                "tensor_count": tensor_count,
                "total_selected": selected_total,
                "stability_jaccard_mean": mean(stability_j),
                "stability_overlap_mean": mean(stability_overlap),
                "full_recall_mean": mean(full_recall),
                "full_recall_min": min_value(full_recall),
                "full_recall_max": max_value(full_recall),
                "full_jaccard_mean": mean(full_j),
                "full_jaccard_min": min_value(full_j),
                "full_jaccard_max": max_value(full_j),
            })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def value_for_markdown(value):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_report(path, args, summary_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Java Sample Calibration Report",
        "",
        "This report compares small-sample Taylor importance masks against the existing Java 10k full-data Taylor estimate.",
        "The full-data Taylor estimate is a stronger baseline, not an absolute ground truth.",
        "",
        "## Inputs",
        "",
        f"- Full checkpoint: `{args.full_checkpoint}`",
        f"- Approx root: `{args.approx_root}`",
        f"- Language: `{args.language}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- Sample sizes: `{', '.join(str(size) for size in args.sample_sizes)}`",
        f"- k values: `{', '.join(format_k(k) for k in args.k_values)}`",
        "- Tensor scope: transformer layer tensors only",
        "",
        "## Summary",
        "",
        "| Samples | k | Stability Jaccard | Stability Overlap | Recall vs Full | Jaccard vs Full |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {sample_size} | {k} | {stability_jaccard_mean} | {stability_overlap_mean} | {full_recall_mean} | {full_jaccard_mean} |".format(
                sample_size=row["sample_size"],
                k=row["k"],
                stability_jaccard_mean=value_for_markdown(row["stability_jaccard_mean"]),
                stability_overlap_mean=value_for_markdown(row["stability_overlap_mean"]),
                full_recall_mean=value_for_markdown(row["full_recall_mean"]),
                full_jaccard_mean=value_for_markdown(row["full_jaccard_mean"]),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use `stability_jaccard_mean` to decide whether two independent fast runs are selecting the same region.",
        "Use `full_recall_mean` only where the 10k Java checkpoint exists; it reports how much of the full-data top-k mask is recovered by the small-sample estimate.",
        "For product/report wording, prefer `stability` and `recall vs full Taylor estimate` over `error rate`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare small-sample grad*param checkpoints against a full Taylor checkpoint.")
    parser.add_argument("--full-checkpoint", required=True, type=Path)
    parser.add_argument("--approx-root", required=True, type=Path)
    parser.add_argument("--language", default="java")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 5678])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--k-values", nargs="+", type=float, default=[0.005, 0.01, 0.03, 0.05])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-non-layer", action="store_true")
    args = parser.parse_args()

    if not args.full_checkpoint.exists():
        raise SystemExit(f"Full checkpoint does not exist: {args.full_checkpoint}")
    if not args.approx_root.exists():
        raise SystemExit(f"Approx root does not exist: {args.approx_root}")

    full_files = sorted(args.full_checkpoint.glob("*.pt"))
    if not args.include_non_layer:
        full_files = [path for path in full_files if path.name.startswith("model.layers.")]
    if not full_files:
        raise SystemExit(f"No checkpoint tensors found in {args.full_checkpoint}")

    pair_acc = defaultdict(lambda: defaultdict(int))
    module_acc = defaultdict(lambda: defaultdict(int))
    per_tensor_rows = []
    k_values = sorted(args.k_values)
    max_k = max(k_values)

    for tensor_idx, full_path in enumerate(full_files, 1):
        layer, module = parse_tensor_name(full_path)
        full_score, full_shape = load_score(full_path)
        numel = full_score.numel()
        max_count = int(max_k * numel)
        counts = {format_k(k): int(k * numel) for k in k_values}
        full_top = top_indices(full_score, max_count)
        del full_score

        for sample_size in args.sample_sizes:
            seed_top = {}
            for seed in args.seeds:
                approx_path = args.approx_root / f"seed_{seed}" / args.language / f"grad-mul-param_checkpoint_{sample_size}" / full_path.name
                if not approx_path.exists():
                    raise FileNotFoundError(approx_path)
                approx_score, approx_shape = load_score(approx_path)
                if approx_shape != full_shape:
                    raise ValueError(f"Shape mismatch for {full_path.name}: {approx_shape} != {full_shape}")
                seed_top[seed] = top_indices(approx_score, max_count)
                del approx_score

            for k in k_values:
                k_label = format_k(k)
                count = counts[k_label]
                full_mask = make_mask(numel, full_top, count)
                seed_masks = {}
                for seed, indices in seed_top.items():
                    seed_label = f"seed_{seed}"
                    seed_mask = make_mask(numel, indices, count)
                    seed_masks[seed] = seed_mask
                    metrics = mask_metrics(seed_mask, full_mask)
                    row = {
                        "tensor": full_path.name,
                        "layer": layer,
                        "module": module,
                        "sample_size": sample_size,
                        "k": k_label,
                        "comparison": "seed_vs_full",
                        "seed_a": seed_label,
                        "seed_b": "full_10000",
                        **metrics,
                    }
                    per_tensor_rows.append(row)
                    key = (sample_size, k_label, "seed_vs_full", seed_label, "full_10000")
                    add_aggregate(pair_acc[key], metrics)
                    module_key = (sample_size, k_label, "seed_vs_full", seed_label, "full_10000", module)
                    add_aggregate(module_acc[module_key], metrics)

                for seed_a, seed_b in itertools.combinations(args.seeds, 2):
                    seed_a_label = f"seed_{seed_a}"
                    seed_b_label = f"seed_{seed_b}"
                    metrics = mask_metrics(seed_masks[seed_a], seed_masks[seed_b])
                    row = {
                        "tensor": full_path.name,
                        "layer": layer,
                        "module": module,
                        "sample_size": sample_size,
                        "k": k_label,
                        "comparison": "seed_vs_seed",
                        "seed_a": seed_a_label,
                        "seed_b": seed_b_label,
                        **metrics,
                    }
                    per_tensor_rows.append(row)
                    key = (sample_size, k_label, "seed_vs_seed", seed_a_label, seed_b_label)
                    add_aggregate(pair_acc[key], metrics)
                    module_key = (sample_size, k_label, "seed_vs_seed", seed_a_label, seed_b_label, module)
                    add_aggregate(module_acc[module_key], metrics)

                del full_mask
                del seed_masks
            del seed_top
        del full_top

        if tensor_idx % 25 == 0 or tensor_idx == len(full_files):
            print(f"processed {tensor_idx}/{len(full_files)} tensors", flush=True)

    comparison_rows = [finalize_aggregate(key, values) for key, values in sorted(pair_acc.items())]
    module_rows = [finalize_module_aggregate(key, values) for key, values in sorted(module_acc.items())]
    summary_rows = build_compact_summary(comparison_rows, args.sample_sizes, k_values)

    summary_fields = [
        "sample_size", "k", "tensor_count", "total_selected",
        "stability_jaccard_mean", "stability_overlap_mean",
        "full_recall_mean", "full_recall_min", "full_recall_max",
        "full_jaccard_mean", "full_jaccard_min", "full_jaccard_max",
    ]
    comparison_fields = [
        "sample_size", "k", "comparison", "seed_a", "seed_b", "tensor_count",
        "selected_a", "selected_b", "intersection", "union", "jaccard", "recall_b", "overlap_a",
    ]
    tensor_fields = ["tensor", "layer", "module", *comparison_fields]
    module_fields = ["module", *comparison_fields]

    write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "comparison_summary.csv", comparison_rows, comparison_fields)
    write_csv(args.output_dir / "per_tensor_metrics.csv", per_tensor_rows, tensor_fields)
    write_csv(args.output_dir / "per_module_metrics.csv", module_rows, module_fields)

    payload = {
        "inputs": {
            "full_checkpoint": str(args.full_checkpoint),
            "approx_root": str(args.approx_root),
            "language": args.language,
            "seeds": args.seeds,
            "sample_sizes": args.sample_sizes,
            "k_values": [format_k(k) for k in k_values],
            "include_non_layer": args.include_non_layer,
        },
        "summary": summary_rows,
        "comparison_summary": comparison_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.output_dir / "sample_calibration_report.md", args, summary_rows)
    print(f"wrote calibration report to {args.output_dir}")


if __name__ == "__main__":
    main()
