#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def group_name(model):
    if model == "original":
        return "original"
    if model.startswith("code_"):
        return "code"
    if model.startswith("random_"):
        return "random"
    if model.startswith("bottom_"):
        return "bottom"
    return model


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "group", "loss", "loss_delta", "ppl", "ppl_ratio", "tokens"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for group in ["original", "bottom", "random", "code"]:
        items = [row for row in rows if row["group"] == group]
        if not items:
            continue
        losses = [row["loss"] for row in items]
        ppls = [row["ppl"] for row in items]
        summary.append({
            "group": group,
            "n": len(items),
            "loss_mean": statistics.mean(losses),
            "loss_std": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
            "ppl_mean": statistics.mean(ppls),
            "ppl_std": statistics.pstdev(ppls) if len(ppls) > 1 else 0.0,
        })
    return summary


def write_summary(summary, path):
    fields = ["group", "n", "loss_mean", "loss_std", "ppl_mean", "ppl_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot(rows, summary, output):
    group_order = ["original", "bottom", "random", "code"]
    colors = {
        "original": "#111827",
        "bottom": "#2563eb",
        "random": "#64748b",
        "code": "#dc2626",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, ylabel, log_scale in [
        (axes[0], "loss", "cross-entropy loss", False),
        (axes[1], "ppl", "perplexity, log scale", True),
    ]:
        x_positions = []
        labels = []
        means = []
        errors = []
        bar_colors = []
        for idx, group in enumerate(group_order):
            item = next((row for row in summary if row["group"] == group), None)
            if item is None:
                continue
            x_positions.append(idx)
            labels.append(f"{group}\n(n={item['n']})")
            means.append(item[f"{key}_mean"])
            errors.append(item[f"{key}_std"])
            bar_colors.append(colors[group])

        ax.bar(x_positions, means, yerr=errors, color=bar_colors, alpha=0.9, capsize=4)
        for row in rows:
            group_idx = group_order.index(row["group"])
            ax.scatter(group_idx, row[key], color="white", edgecolor="#111827", zorder=3, s=42)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Top0.01 control comparison: {key}")
        ax.grid(True, axis="y", alpha=0.25)
        if log_scale:
            ax.set_yscale("log")

    fig.suptitle("Code Region vs Random/Bottom Controls on Java Test Subset", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot code-region damage against random and bottom controls.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    original = next(row for row in rows if row["model"] == "original")
    for row in rows:
        row["group"] = group_name(row["model"])
        row["loss_delta"] = row["loss"] - original["loss"]
        row["ppl_ratio"] = row["ppl"] / original["ppl"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "control_ppl_rows.csv")
    summary = summarize(rows)
    write_summary(summary, args.output_dir / "control_ppl_summary.csv")
    plot(rows, summary, args.output_dir / "control_ppl_comparison.png")
    print(f"wrote control comparison plots to {args.output_dir}")


if __name__ == "__main__":
    main()
