#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
EPS = 1e-30

def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = {}
    for row in rows:
        key = (int(row["layer"]), row["module"])
        parsed[key] = {
            "layer": int(row["layer"]),
            "module": row["module"],
            "numel": int(float(row["numel"])),
            "importance_sum": float(row["importance_sum"]),
            "importance_mean": float(row["importance_mean"]),
        }
    return parsed

def compare(code_rows, baseline_rows):
    keys = sorted(set(code_rows) & set(baseline_rows))
    code_total = sum(code_rows[key]["importance_sum"] for key in keys)
    baseline_total = sum(baseline_rows[key]["importance_sum"] for key in keys)
    rows = []
    for key in keys:
        code = code_rows[key]
        baseline = baseline_rows[key]
        code_sum = code["importance_sum"]
        baseline_sum = baseline["importance_sum"]
        code_share = code_sum / code_total if code_total else 0.0
        baseline_share = baseline_sum / baseline_total if baseline_total else 0.0
        raw_ratio = (code_sum + EPS) / (baseline_sum + EPS)
        share_ratio = (code_share + EPS) / (baseline_share + EPS)
        rows.append({
            "layer": code["layer"],
            "module": code["module"],
            "numel": code["numel"],
            "code_importance_sum": code_sum,
            "baseline_importance_sum": baseline_sum,
            "code_importance_share": code_share,
            "baseline_importance_share": baseline_share,
            "importance_raw_ratio": raw_ratio,
            "importance_share_ratio": share_ratio,
            "log2_importance_share_ratio": math.log2(share_ratio),
            "importance_share_diff": code_share - baseline_share,
        })
    return rows

def write_csv(rows, path: Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def module_summary(rows):
    modules = {}
    for row in rows:
        module = row["module"]
        item = modules.setdefault(module, {
            "module": module,
            "numel": 0,
            "code_importance_sum": 0.0,
            "baseline_importance_sum": 0.0,
        })
        item["numel"] += row["numel"]
        item["code_importance_sum"] += row["code_importance_sum"]
        item["baseline_importance_sum"] += row["baseline_importance_sum"]
    code_total = sum(row["code_importance_sum"] for row in modules.values())
    baseline_total = sum(row["baseline_importance_sum"] for row in modules.values())
    out = []
    for row in modules.values():
        code_share = row["code_importance_sum"] / code_total if code_total else 0.0
        baseline_share = row["baseline_importance_sum"] / baseline_total if baseline_total else 0.0
        share_ratio = (code_share + EPS) / (baseline_share + EPS)
        out.append({
            "module": row["module"],
            "numel": row["numel"],
            "code_importance_sum": row["code_importance_sum"],
            "baseline_importance_sum": row["baseline_importance_sum"],
            "code_importance_share": code_share,
            "baseline_importance_share": baseline_share,
            "importance_share_ratio": share_ratio,
            "log2_importance_share_ratio": math.log2(share_ratio),
            "importance_share_diff": code_share - baseline_share,
        })
    return sorted(out, key=lambda row: row["log2_importance_share_ratio"], reverse=True)

def matrix_from_rows(rows, value_key):
    layers = sorted({row["layer"] for row in rows})
    modules = [module for module in MODULE_ORDER if any(row["module"] == module for row in rows)]
    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    module_index = {module: idx for idx, module in enumerate(modules)}
    for row in rows:
        if row["module"] in module_index:
            matrix[layer_index[row["layer"]], module_index[row["module"]]] = row[value_key]
    return matrix, layers, modules

def plot_heatmap(rows, value_key, title, path: Path, center_zero=True):
    matrix, layers, modules = matrix_from_rows(rows, value_key)
    plt.figure(figsize=(max(10, len(modules) * 1.35), max(8, len(layers) * 0.32)))
    kwargs = {"xticklabels": modules, "yticklabels": layers, "mask": np.isnan(matrix), "cbar_kws": {"label": value_key}}
    if center_zero:
        sns.heatmap(matrix, cmap="coolwarm", center=0, **kwargs)
    else:
        sns.heatmap(matrix, cmap="viridis", **kwargs)
    plt.title(title)
    plt.xlabel("module")
    plt.ylabel("layer")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()

def plot_module_enrichment(rows, path: Path):
    modules = [row["module"] for row in rows]
    values = [row["log2_importance_share_ratio"] for row in rows]
    colors = ["#dc2626" if value >= 0 else "#2563eb" for value in values]
    plt.figure(figsize=(10, max(5, len(modules) * 0.42)))
    sns.barplot(x=values, y=modules, palette=colors, hue=modules, legend=False)
    plt.axvline(0, color="#111827", linewidth=1)
    plt.xlabel("log2(code importance share / baseline importance share)")
    plt.ylabel("module")
    plt.title("Code importance enrichment by module")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare code and non-code grad*param importance summaries.")
    parser.add_argument("--code", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = compare(load_rows(args.code), load_rows(args.baseline))
    if not rows:
        raise SystemExit("No overlapping layer/module rows to compare")

    parameter_fields = [
        "layer", "module", "numel", "code_importance_sum", "baseline_importance_sum",
        "code_importance_share", "baseline_importance_share", "importance_raw_ratio",
        "importance_share_ratio", "log2_importance_share_ratio", "importance_share_diff",
    ]
    write_csv(sorted(rows, key=lambda row: row["log2_importance_share_ratio"], reverse=True), args.output / "parameter_importance_comparison.csv", parameter_fields)

    modules = module_summary(rows)
    module_fields = [
        "module", "numel", "code_importance_sum", "baseline_importance_sum",
        "code_importance_share", "baseline_importance_share", "importance_share_ratio",
        "log2_importance_share_ratio", "importance_share_diff",
    ]
    write_csv(modules, args.output / "module_importance_comparison.csv", module_fields)

    plot_heatmap(rows, "log2_importance_share_ratio", "Code enrichment: log2(code share / baseline share)", args.output / "code_enrichment_heatmap.png")
    percent_rows = [dict(row, importance_share_diff=row["importance_share_diff"] * 100) for row in rows]
    plot_heatmap(percent_rows, "importance_share_diff", "Code minus baseline importance share, percentage points", args.output / "code_minus_baseline_share_heatmap.png")
    plot_module_enrichment(modules, args.output / "module_code_enrichment.png")
    print(f"wrote comparison for {len(rows)} layer/module rows to {args.output}")

if __name__ == "__main__":
    main()
