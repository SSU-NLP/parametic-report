#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MODULE_ORDER = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
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
    "self_attn.q_norm.weight": "Q norm",
    "self_attn.k_norm.weight": "K norm",
    "mlp.gate_proj.weight": "MLP gate",
    "mlp.up_proj.weight": "MLP up",
    "mlp.down_proj.weight": "MLP down",
    "input_layernorm.weight": "in norm",
    "post_attention_layernorm.weight": "post norm",
}


MODULE_GROUPS = {
    "all": None,
    "attention-proj": {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    },
}


def module_filter(module_group):
    if module_group not in MODULE_GROUPS:
        raise ValueError(f"Unknown module group: {module_group}")
    allowed = MODULE_GROUPS[module_group]
    if allowed is None:
        return lambda module: True
    return lambda module: module in allowed


def suffix_for_group(module_group):
    return "" if module_group == "all" else f"_{module_group.replace('-', '_')}"


def format_k(k):
    return f"{float(k):g}"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(report_dir, model, module_group="all"):
    if module_group == "all":
        rows = []
        for row in read_csv(report_dir / "summary.csv"):
            item = {"model": model}
            item.update(row)
            item["sample_size"] = int(item["sample_size"])
            item["k_float"] = float(item["k"])
            for field in [
                "stability_jaccard_mean",
                "stability_overlap_mean",
                "tensor_count",
                "total_selected",
            ]:
                if item.get(field) != "":
                    item[field] = float(item[field])
            rows.append(item)
        return rows

    keep = module_filter(module_group)
    acc = {}
    for row in read_csv(report_dir / "per_tensor_metrics.csv"):
        if not keep(row["module"]):
            continue
        key = (int(row["sample_size"]), row["k"])
        item = acc.setdefault(key, {
            "model": model,
            "sample_size": key[0],
            "k": key[1],
            "k_float": float(key[1]),
            "tensor_count": 0,
            "total_selected": 0,
            "selected_b": 0,
            "intersection": 0,
            "union": 0,
        })
        item["tensor_count"] += 1
        item["total_selected"] += int(float(row["selected_a"]))
        item["selected_b"] += int(float(row["selected_b"]))
        item["intersection"] += int(float(row["intersection"]))
        item["union"] += int(float(row["union"]))

    rows = []
    for item in acc.values():
        item["stability_jaccard_mean"] = item["intersection"] / item["union"] if item["union"] else np.nan
        item["stability_overlap_mean"] = item["intersection"] / item["total_selected"] if item["total_selected"] else np.nan
        rows.append(item)
    return sorted(rows, key=lambda row: (row["sample_size"], row["k_float"]))


def read_tensor_rows(report_dir, model, sample_size, k, module_group="all"):
    out = []
    k_label = format_k(k)
    keep = module_filter(module_group)
    for row in read_csv(report_dir / "per_tensor_metrics.csv"):
        if row["sample_size"] != str(sample_size) or row["k"] != k_label:
            continue
        if not keep(row["module"]):
            continue
        item = {"model": model}
        item.update(row)
        item["layer"] = int(item["layer"])
        item["jaccard"] = float(item["jaccard"])
        item["overlap_a"] = float(item["overlap_a"])
        item["instability"] = 1.0 - item["jaccard"]
        out.append(item)
    return out


def summary_matrix(rows, model, value_key):
    model_rows = [row for row in rows if row["model"] == model]
    samples = sorted({row["sample_size"] for row in model_rows})
    ks = sorted({row["k_float"] for row in model_rows})
    matrix = np.full((len(samples), len(ks)), np.nan, dtype=float)
    index = {(row["sample_size"], row["k_float"]): row for row in model_rows}
    for row_idx, sample in enumerate(samples):
        for col_idx, k in enumerate(ks):
            matrix[row_idx, col_idx] = index[(sample, k)][value_key]
    return matrix, samples, ks


def plot_summary(llama_rows, qwen_rows, output, module_group="all"):
    rows = llama_rows + qwen_rows
    llama_j, samples, ks = summary_matrix(rows, "Llama-3.2-3B", "stability_jaccard_mean")
    qwen_j, _, _ = summary_matrix(rows, "Qwen3-8B", "stability_jaccard_mean")
    llama_o, _, _ = summary_matrix(rows, "Llama-3.2-3B", "stability_overlap_mean")
    qwen_o, _, _ = summary_matrix(rows, "Qwen3-8B", "stability_overlap_mean")

    diff_j = qwen_j - llama_j
    diff_o = qwen_o - llama_o
    labels_x = [format_k(k) for k in ks]
    labels_y = [str(sample) for sample in samples]

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.34, wspace=0.28)
    ax_line = fig.add_subplot(grid[0, :])
    ax_l = fig.add_subplot(grid[1, 0])
    ax_q = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])

    colors = {0.005: "#2563eb", 0.01: "#dc2626", 0.03: "#16a34a", 0.05: "#7c3aed"}
    for k in ks:
        llama_vals = [row["stability_jaccard_mean"] for row in llama_rows if row["k_float"] == k]
        qwen_vals = [row["stability_jaccard_mean"] for row in qwen_rows if row["k_float"] == k]
        ax_line.plot(samples, llama_vals, marker="o", linewidth=2.2, color=colors.get(k), label=f"Llama k={format_k(k)}")
        ax_line.plot(samples, qwen_vals, marker="s", linewidth=2.2, linestyle="--", color=colors.get(k), label=f"Qwen k={format_k(k)}")
    ax_line.set_title(f"Seed-to-seed top-k stability across sample sizes ({module_group})")
    ax_line.set_xlabel("samples per seed")
    ax_line.set_ylabel("global Jaccard")
    ax_line.set_xticks(samples)
    ax_line.set_ylim(min(np.nanmin(llama_j), np.nanmin(qwen_j)) - 0.001, 1.0002)
    ax_line.legend(ncol=4, fontsize=9, loc="lower right")
    ax_line.grid(True, alpha=0.28)

    vmin = min(np.nanmin(llama_j), np.nanmin(qwen_j))
    vmax = max(np.nanmax(llama_j), np.nanmax(qwen_j))
    sns.heatmap(llama_j, ax=ax_l, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=".4f", xticklabels=labels_x, yticklabels=labels_y, cbar_kws={"label": "Jaccard"})
    ax_l.set_title("Llama-3.2-3B stability")
    ax_l.set_xlabel("k")
    ax_l.set_ylabel("samples")
    sns.heatmap(qwen_j, ax=ax_q, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=".4f", xticklabels=labels_x, yticklabels=labels_y, cbar_kws={"label": "Jaccard"})
    ax_q.set_title("Qwen3-8B stability")
    ax_q.set_xlabel("k")
    ax_q.set_ylabel("samples")

    diff_bound = max(abs(float(np.nanmin(diff_j))), abs(float(np.nanmax(diff_j))))
    sns.heatmap(diff_j, ax=ax_d, cmap="vlag", center=0.0, vmin=-diff_bound, vmax=diff_bound, annot=True, fmt="+.4f", xticklabels=labels_x, yticklabels=labels_y, cbar_kws={"label": "Qwen - Llama Jaccard"})
    ax_d.set_title("Difference: Qwen minus Llama")
    ax_d.set_xlabel("k")
    ax_d.set_ylabel("samples")

    fig.suptitle(f"Small-sample Coding Spot stability: Llama vs Qwen ({module_group})", fontsize=16, y=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)

    return diff_j, diff_o, samples, ks


def matrix_from_tensor_rows(rows):
    layers = sorted({row["layer"] for row in rows})
    modules = [module for module in MODULE_ORDER if any(row["module"] == module for row in rows)]
    for module in sorted({row["module"] for row in rows}):
        if module not in modules:
            modules.append(module)
    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    module_index = {module: idx for idx, module in enumerate(modules)}
    for row in rows:
        matrix[layer_index[row["layer"]], module_index[row["module"]]] = row["instability"]
    return matrix, layers, modules


def plot_layer_module_atlas(llama_tensor_rows, qwen_tensor_rows, output, sample_size, k, module_group="all"):
    llama_matrix, llama_layers, llama_modules = matrix_from_tensor_rows(llama_tensor_rows)
    qwen_matrix, qwen_layers, qwen_modules = matrix_from_tensor_rows(qwen_tensor_rows)
    modules = [module for module in MODULE_ORDER if module in set(llama_modules) | set(qwen_modules)]
    module_labels = [MODULE_LABELS.get(module, module) for module in modules]

    def reindex(matrix, old_modules, modules):
        out = np.full((matrix.shape[0], len(modules)), np.nan, dtype=float)
        old_index = {module: idx for idx, module in enumerate(old_modules)}
        for col_idx, module in enumerate(modules):
            if module in old_index:
                out[:, col_idx] = matrix[:, old_index[module]]
        return out

    llama_plot = reindex(llama_matrix, llama_modules, modules)
    qwen_plot = reindex(qwen_matrix, qwen_modules, modules)
    vmax = float(np.nanpercentile(np.concatenate([llama_plot.reshape(-1), qwen_plot.reshape(-1)]), 98))
    vmax = max(vmax, 0.01)

    sns.set_theme(style="white")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e5e7eb")
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.18)
    ax_l = fig.add_subplot(grid[0, 0])
    ax_q = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    sns.heatmap(
        llama_plot,
        ax=ax_l,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        mask=np.isnan(llama_plot),
        cbar=False,
        xticklabels=module_labels,
        yticklabels=[f"L{layer:02d}" for layer in llama_layers],
        linewidths=0.18,
        linecolor="#111827",
    )
    sns.heatmap(
        qwen_plot,
        ax=ax_q,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        mask=np.isnan(qwen_plot),
        cbar=True,
        cbar_ax=ax_c,
        cbar_kws={"label": "instability = 1 - Jaccard"},
        xticklabels=module_labels,
        yticklabels=[f"L{layer:02d}" for layer in qwen_layers],
        linewidths=0.18,
        linecolor="#111827",
    )
    for ax, title in [(ax_l, "Llama-3.2-3B"), (ax_q, "Qwen3-8B")]:
        ax.set_title(title)
        ax.set_xlabel("module")
        ax.set_ylabel("layer")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_facecolor("#e5e7eb")
    fig.text(0.5, 0.025, "Color encodes instability = 1 - Jaccard. Dark cells are more stable; brighter cells are less stable. Gray cells mean the module is absent in that model.", ha="center", fontsize=10)
    fig.suptitle(f"Layer/module stability atlas ({module_group}), sample={sample_size}, k={format_k(k)}", fontsize=16, y=0.97)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_comparison_csv(llama_rows, qwen_rows, output):
    key = lambda row: (row["sample_size"], row["k"])
    llama = {key(row): row for row in llama_rows}
    qwen = {key(row): row for row in qwen_rows}
    fields = [
        "sample_size",
        "k",
        "llama_jaccard",
        "qwen_jaccard",
        "qwen_minus_llama_jaccard",
        "llama_overlap",
        "qwen_overlap",
        "qwen_minus_llama_overlap",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_key in sorted(llama):
            if item_key not in qwen:
                continue
            lrow = llama[item_key]
            qrow = qwen[item_key]
            writer.writerow({
                "sample_size": item_key[0],
                "k": item_key[1],
                "llama_jaccard": lrow["stability_jaccard_mean"],
                "qwen_jaccard": qrow["stability_jaccard_mean"],
                "qwen_minus_llama_jaccard": qrow["stability_jaccard_mean"] - lrow["stability_jaccard_mean"],
                "llama_overlap": lrow["stability_overlap_mean"],
                "qwen_overlap": qrow["stability_overlap_mean"],
                "qwen_minus_llama_overlap": qrow["stability_overlap_mean"] - lrow["stability_overlap_mean"],
            })


def main():
    parser = argparse.ArgumentParser(description="Plot sample-stability comparison figures.")
    parser.add_argument("--llama-report-dir", required=True, type=Path)
    parser.add_argument("--qwen-report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--atlas-sample", type=int, default=1024)
    parser.add_argument("--atlas-k", type=float, default=0.01)
    parser.add_argument("--module-group", choices=sorted(MODULE_GROUPS), default="all")
    args = parser.parse_args()

    suffix = suffix_for_group(args.module_group)
    llama_rows = read_summary(args.llama_report_dir, "Llama-3.2-3B", args.module_group)
    qwen_rows = read_summary(args.qwen_report_dir, "Qwen3-8B", args.module_group)
    plot_summary(
        llama_rows,
        qwen_rows,
        args.output_dir / f"sample_stability_model_comparison{suffix}.png",
        args.module_group,
    )
    write_comparison_csv(llama_rows, qwen_rows, args.output_dir / f"sample_stability_model_comparison{suffix}.csv")

    llama_tensor = read_tensor_rows(args.llama_report_dir, "Llama-3.2-3B", args.atlas_sample, args.atlas_k, args.module_group)
    qwen_tensor = read_tensor_rows(args.qwen_report_dir, "Qwen3-8B", args.atlas_sample, args.atlas_k, args.module_group)
    plot_layer_module_atlas(
        llama_tensor,
        qwen_tensor,
        args.output_dir / f"layer_module_stability_atlas_s{args.atlas_sample}_k{format_k(args.atlas_k)}{suffix}.png",
        args.atlas_sample,
        args.atlas_k,
        args.module_group,
    )
    print(f"wrote sample stability figures to {args.output_dir}")


if __name__ == "__main__":
    main()
