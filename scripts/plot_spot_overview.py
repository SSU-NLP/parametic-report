#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PARAM_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.weight$")
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
MODULE_LABELS = {
    "self_attn.q_proj.weight": "Q",
    "self_attn.k_proj.weight": "K",
    "self_attn.v_proj.weight": "V",
    "self_attn.o_proj.weight": "O",
    "mlp.gate_proj.weight": "MLP gate",
    "mlp.up_proj.weight": "MLP up",
    "mlp.down_proj.weight": "MLP down",
    "input_layernorm.weight": "in norm",
    "post_attention_layernorm.weight": "post norm",
}


def load_importance_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append({
            "layer": int(row["layer"]),
            "module": row["module"],
            "importance_sum": float(row["importance_sum"]),
        })
    return parsed


def build_matrix(rows):
    layers = sorted({row["layer"] for row in rows})
    modules = [module for module in MODULE_ORDER if any(row["module"] == module for row in rows)]
    matrix = np.zeros((len(layers), len(modules)), dtype=float)
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    module_index = {module: idx for idx, module in enumerate(modules)}
    for row in rows:
        if row["module"] in module_index:
            matrix[layer_index[row["layer"]], module_index[row["module"]]] = row["importance_sum"]
    matrix = matrix / matrix.sum() * 100
    return matrix, layers, modules


def load_control_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    random_ppls = []
    for row in rows:
        model = row["model"]
        if model == "original":
            out["original"] = row["ppl"]
        elif model.startswith("code_"):
            out["code"] = row["ppl"]
        elif model.startswith("bottom_"):
            out["bottom"] = row["ppl"]
        elif model.startswith("random_"):
            random_ppls.append(row["ppl"])
    if random_ppls:
        out["random_mean"] = sum(random_ppls) / len(random_ppls)
        out["random_min"] = min(random_ppls)
        out["random_max"] = max(random_ppls)
    return out


def plot(matrix, layers, modules, control, output):
    module_totals = matrix.sum(axis=0)
    layer_totals = matrix.sum(axis=1)
    module_labels = [MODULE_LABELS.get(module, module) for module in modules]

    fig = plt.figure(figsize=(17, 11), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=[9.4, 0.35, 2.3, 4.2],
        height_ratios=[2.0, 8.0, 0.25],
        hspace=0.24,
        wspace=0.24,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0])
    ax_cbar = fig.add_subplot(grid[1, 1])
    ax_right = fig.add_subplot(grid[1, 2])
    ax_summary = fig.add_subplot(grid[0:2, 3])
    ax_bottom = fig.add_subplot(grid[2, 0:3])

    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap="magma",
        xticklabels=module_labels,
        yticklabels=[f"L{layer:02d}" for layer in layers],
        cbar=True,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "share of total importance (%)"},
        linewidths=0.25,
        linecolor="#1f2937",
    )
    ax_heat.set_xlabel("model component")
    ax_heat.set_ylabel("transformer layer")
    ax_heat.tick_params(axis="x", rotation=30, labelsize=9)
    ax_heat.tick_params(axis="y", labelsize=8)
    ax_heat.set_title("Java Code Spot Map: layer x component importance", pad=10)

    bars = ax_top.bar(range(len(modules)), module_totals, color="#7c3aed")
    ax_top.set_xlim(-0.5, len(modules) - 0.5)
    ax_top.set_ylim(0, max(module_totals) * 1.28)
    ax_top.set_xticks([])
    ax_top.set_ylabel("module\nshare %")
    ax_top.grid(True, axis="y", alpha=0.25)
    ax_top.bar_label(
        bars,
        labels=[f"{value:.1f}" if value >= 3 else "" for value in module_totals],
        label_type="center",
        color="white",
        fontsize=8,
        fontweight="bold",
    )
    for idx, value in enumerate(module_totals):
        if value < 3:
            ax_top.text(idx, value + max(module_totals) * 0.035, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#334155")

    ax_right.barh(range(len(layers)), layer_totals, color="#2563eb")
    ax_right.set_ylim(len(layers) - 0.5, -0.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("layer\nshare %")
    ax_right.grid(True, axis="x", alpha=0.25)

    top_cells = []
    for layer_idx, layer in enumerate(layers):
        for module_idx, module in enumerate(modules):
            top_cells.append((matrix[layer_idx, module_idx], layer, MODULE_LABELS.get(module, module)))
    top_cells = sorted(top_cells, reverse=True)[:6]

    ax_summary.axis("off")
    random_text = "n/a"
    if "random_mean" in control:
        random_text = f"{control['random_mean']:.2f} ({control['random_min']:.2f}-{control['random_max']:.2f})"
    code_ratio = control.get("code", 0) / control.get("original", 1)
    summary = [
        "Validation",
        "Java test subset, top0.01 zero-out",
        "",
        f"Original PPL       {control.get('original', float('nan')):.2f}",
        f"Bottom PPL         {control.get('bottom', float('nan')):.2f}",
        f"Random PPL         {random_text}",
        f"Code-region PPL    {control.get('code', float('nan')):.2f}",
        f"Code/original      {code_ratio:,.0f}x",
        "",
        "Top Cells",
    ]
    summary.extend(f"- L{layer:02d} {module}: {value:.2f}%" for value, layer, module in top_cells)
    ax_summary.text(
        0.03,
        0.98,
        "\n".join(summary),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    ax_bottom.axis("off")

    fig.suptitle("Where is the Java Code Spot?", fontsize=18, y=0.975)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.045)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create a one-page spot overview figure.")
    parser.add_argument("--importance-summary", required=True, type=Path)
    parser.add_argument("--control-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    matrix, layers, modules = build_matrix(load_importance_rows(args.importance_summary))
    control = load_control_rows(args.control_results)
    plot(matrix, layers, modules, control, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
