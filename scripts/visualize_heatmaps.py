#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


PARAM_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.pt$")
MODULE_ORDER = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]


def parse_param(path: Path):
    match = PARAM_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def safe_load(path: Path):
    return torch.load(path, map_location="cpu")


def collect_stats(checkpoint_dir: Path, mask_dir: Path):
    rows = []
    for grad_path in sorted(checkpoint_dir.glob("model.layers.*.pt")):
        parsed = parse_param(grad_path)
        if parsed is None:
            continue
        layer, module = parsed
        grad_tensor = safe_load(grad_path).float().abs()
        mask_path = mask_dir / grad_path.name
        mask_density = np.nan
        masked_importance_mean = np.nan
        if mask_path.exists():
            mask_tensor = safe_load(mask_path).bool()
            mask_density = mask_tensor.float().mean().item()
            if mask_tensor.any():
                masked_importance_mean = grad_tensor[mask_tensor].mean().item()
        rows.append({
            "layer": layer,
            "module": module,
            "numel": grad_tensor.numel(),
            "importance_mean": grad_tensor.mean().item(),
            "importance_max": grad_tensor.max().item(),
            "importance_sum": grad_tensor.sum().item(),
            "mask_density": mask_density,
            "masked_importance_mean": masked_importance_mean,
        })
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "module",
        "numel",
        "importance_mean",
        "importance_max",
        "importance_sum",
        "mask_density",
        "masked_importance_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_csv(rows, path: Path):
    ranked_rows = sorted(rows, key=lambda row: row["importance_sum"], reverse=True)
    write_csv(ranked_rows, path)


def write_module_summary(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = {}
    total_importance = sum(row["importance_sum"] for row in rows)
    for row in rows:
        module = row["module"]
        if module not in totals:
            totals[module] = {"module": module, "numel": 0, "importance_sum": 0.0}
        totals[module]["numel"] += row["numel"]
        totals[module]["importance_sum"] += row["importance_sum"]

    summary_rows = []
    for row in totals.values():
        importance_sum = row["importance_sum"]
        numel = row["numel"]
        summary_rows.append({
            "module": row["module"],
            "numel": numel,
            "importance_sum": importance_sum,
            "importance_share": importance_sum / total_importance if total_importance else 0.0,
            "importance_mean": importance_sum / numel if numel else 0.0,
        })
    summary_rows.sort(key=lambda row: row["importance_sum"], reverse=True)

    fields = ["module", "numel", "importance_sum", "importance_share", "importance_mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def matrix_from_rows(rows, value_key):
    layers = sorted({row["layer"] for row in rows})
    modules = [module for module in MODULE_ORDER if any(row["module"] == module for row in rows)]
    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    module_index = {module: idx for idx, module in enumerate(modules)}
    for row in rows:
        if row["module"] not in module_index:
            continue
        matrix[layer_index[row["layer"]], module_index[row["module"]]] = row[value_key]
    return matrix, layers, modules


def plot_heatmap(rows, value_key, title, output_path: Path, log_scale=False):
    matrix, layers, modules = matrix_from_rows(rows, value_key)
    plot_matrix = matrix.copy()
    if log_scale:
        plot_matrix = np.log10(np.maximum(plot_matrix, 1e-30))
        cbar_label = f"log10({value_key})"
    else:
        cbar_label = value_key

    height = max(8, len(layers) * 0.32)
    width = max(10, len(modules) * 1.35)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        plot_matrix,
        cmap="viridis",
        xticklabels=modules,
        yticklabels=layers,
        cbar_kws={"label": cbar_label},
        mask=np.isnan(plot_matrix),
    )
    plt.title(title)
    plt.xlabel("module")
    plt.ylabel("layer")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_module_bar(summary_rows, output_path: Path):
    modules = [row["module"] for row in summary_rows]
    shares = [row["importance_share"] * 100 for row in summary_rows]
    height = max(5, len(modules) * 0.42)
    plt.figure(figsize=(10, height))
    sns.barplot(x=shares, y=modules, color="#3b82f6")
    plt.xlabel("share of total importance (%)")
    plt.ylabel("module")
    plt.title("Module share of total abs(grad * param)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create layer/module heatmaps for grad*param and mask density.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", default=Path("reports/heatmaps"), type=Path)
    args = parser.parse_args()

    rows = collect_stats(args.checkpoint, args.mask)
    if not rows:
        raise SystemExit(f"No layer parameter .pt files found in {args.checkpoint}")

    write_csv(rows, args.output / "parameter_heatmap_summary.csv")
    write_ranked_csv(rows, args.output / "parameter_importance_ranked.csv")
    module_summary = write_module_summary(rows, args.output / "module_importance_summary.csv")
    plot_module_bar(module_summary, args.output / "module_importance_share.png")
    plot_heatmap(
        rows,
        "importance_sum",
        "Total abs(grad * param) by layer/module",
        args.output / "importance_sum_heatmap.png",
        log_scale=True,
    )
    plot_heatmap(
        rows,
        "importance_mean",
        "Mean abs(grad * param) by layer/module",
        args.output / "importance_mean_heatmap.png",
        log_scale=True,
    )
    plot_heatmap(
        rows,
        "importance_max",
        "Max abs(grad * param) by layer/module",
        args.output / "importance_max_heatmap.png",
        log_scale=True,
    )
    plot_heatmap(
        rows,
        "mask_density",
        "Top-k mask density by layer/module",
        args.output / "mask_density_heatmap.png",
        log_scale=False,
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
