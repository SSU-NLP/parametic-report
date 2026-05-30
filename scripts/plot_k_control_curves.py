#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


K_RE = re.compile(r"top([0-9.]+)")


def parse_row(row):
    model = row["model"]
    if model == "original":
        return "original", 0.0
    match = K_RE.search(model)
    if not match:
        raise ValueError(f"Cannot parse k from model label: {model}")
    k = float(match.group(1))
    if model.startswith("code_"):
        group = "code"
    elif model.startswith("bottom_"):
        group = "bottom"
    elif model.startswith("random_"):
        group = "random"
    else:
        raise ValueError(f"Cannot parse group from model label: {model}")
    return group, k


def summarize(rows):
    original = next(row for row in rows if row["model"] == "original")
    grouped = {}
    for row in rows:
        group, k = parse_row(row)
        if group == "original":
            continue
        grouped.setdefault(k, {}).setdefault(group, []).append(row)

    summary = []
    for k in sorted(grouped):
        item = {"k": k, "k_percent": k * 100}
        for group in ["code", "bottom", "random"]:
            values = grouped[k].get(group, [])
            if not values:
                continue
            losses = [row["loss"] for row in values]
            ppls = [row["ppl"] for row in values]
            item[f"{group}_n"] = len(values)
            item[f"{group}_loss_mean"] = statistics.mean(losses)
            item[f"{group}_loss_std"] = statistics.pstdev(losses) if len(losses) > 1 else 0.0
            item[f"{group}_ppl_mean"] = statistics.mean(ppls)
            item[f"{group}_ppl_std"] = statistics.pstdev(ppls) if len(ppls) > 1 else 0.0
            item[f"{group}_ppl_ratio_mean"] = item[f"{group}_ppl_mean"] / original["ppl"]
        summary.append(item)
    return original, summary


def write_csv(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "k",
        "k_percent",
        "code_n",
        "code_loss_mean",
        "code_loss_std",
        "code_ppl_mean",
        "code_ppl_std",
        "code_ppl_ratio_mean",
        "random_n",
        "random_loss_mean",
        "random_loss_std",
        "random_ppl_mean",
        "random_ppl_std",
        "random_ppl_ratio_mean",
        "bottom_n",
        "bottom_loss_mean",
        "bottom_loss_std",
        "bottom_ppl_mean",
        "bottom_ppl_std",
        "bottom_ppl_ratio_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot(original, summary, output):
    x = [row["k_percent"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    styles = {
        "code": {"color": "#dc2626", "label": "code-region"},
        "random": {"color": "#64748b", "label": "random, mean +/- std"},
        "bottom": {"color": "#2563eb", "label": "bottom"},
    }

    for group, style in styles.items():
        losses = [row.get(f"{group}_loss_mean") for row in summary]
        loss_err = [row.get(f"{group}_loss_std", 0.0) for row in summary]
        ppls = [row.get(f"{group}_ppl_mean") for row in summary]
        ppl_err = [row.get(f"{group}_ppl_std", 0.0) for row in summary]
        ratios = [row.get(f"{group}_ppl_ratio_mean") for row in summary]
        axes[0].errorbar(x, losses, yerr=loss_err, marker="o", linewidth=2, capsize=3, **style)
        axes[1].errorbar(x, ppls, yerr=ppl_err, marker="o", linewidth=2, capsize=3, **style)
        axes[2].plot(x, ratios, marker="o", linewidth=2, color=style["color"], label=style["label"])

    axes[0].axhline(original["loss"], color="#111827", linestyle="--", linewidth=1, label="original")
    axes[1].axhline(original["ppl"], color="#111827", linestyle="--", linewidth=1, label="original")
    axes[2].axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="original")

    axes[0].set_title("Loss after zero-out")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_title("PPL after zero-out")
    axes[1].set_ylabel("perplexity, log scale")
    axes[1].set_yscale("log")
    axes[2].set_title("PPL ratio vs original")
    axes[2].set_ylabel("ratio, log scale")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("zeroed weights per tensor (%)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)

    axes[2].legend(loc="best", fontsize=9)
    fig.suptitle("Java Code-Region Damage vs Matched Controls Across k", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot k-sweep control curves for code/random/bottom damage.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    original, summary = summarize(rows)
    write_csv(summary, args.output_dir / "all_k_control_summary.csv")
    plot(original, summary, args.output_dir / "all_k_control_curves.png")
    print(f"wrote k-control curves to {args.output_dir}")


if __name__ == "__main__":
    main()
