#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_k(model_name):
    if model_name == "original":
        return 0.0
    if model_name.startswith("top"):
        return float(model_name[3:])
    raise ValueError(f"Cannot parse damage ratio from model name: {model_name}")


def load_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: parse_k(row["model"]))
    original = next(row for row in rows if row["model"] == "original")
    for row in rows:
        row["k"] = parse_k(row["model"])
        row["loss_delta"] = row["loss"] - original["loss"]
        row["ppl_ratio"] = row["ppl"] / original["ppl"]
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "k", "loss", "loss_delta", "ppl", "ppl_ratio", "tokens"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot(rows, output):
    damaged = [row for row in rows if row["k"] > 0]
    x = [row["k"] * 100 for row in damaged]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(x, [row["loss"] for row in damaged], marker="o", color="#2563eb")
    axes[0].axhline(rows[0]["loss"], color="#111827", linewidth=1, linestyle="--", label="original")
    axes[0].set_title("Java PPL Loss")
    axes[0].set_xlabel("zeroed code-region weights (%)")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()

    axes[1].plot(x, [row["loss_delta"] for row in damaged], marker="o", color="#dc2626")
    axes[1].set_title("Loss Increase vs Original")
    axes[1].set_xlabel("zeroed code-region weights (%)")
    axes[1].set_ylabel("loss delta")

    axes[2].plot(x, [row["ppl_ratio"] for row in damaged], marker="o", color="#7c3aed")
    axes[2].set_yscale("log")
    axes[2].set_title("PPL Ratio vs Original")
    axes[2].set_xlabel("zeroed code-region weights (%)")
    axes[2].set_ylabel("ppl ratio, log scale")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)

    fig.suptitle("Damage Effect on Java Test Subset", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot original vs damaged model evaluation results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input)
    write_csv(rows, args.output_dir / "damage_ppl_summary.csv")
    plot(rows, args.output_dir / "damage_ppl_comparison.png")
    print(f"wrote damage comparison plots to {args.output_dir}")


if __name__ == "__main__":
    main()
