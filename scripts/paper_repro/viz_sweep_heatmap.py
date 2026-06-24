#!/usr/bin/env python3
"""High-res heatmap of bridge transplant pass@1 over the (kA, kB) grid.

Reads metrics.json under a local dir laid out as <root>/ka<kA>_kb<kB>/<strategy>/metrics.json
(downloaded from /shared results-transplant-sweep). Produces a kB(rows) x kA(cols) heatmap of
HumanEval(chat) pass@1, annotated, with floor/ceiling in the title. CPU/host, matplotlib.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KAS = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
KBS = [0.000025, 0.0001, 0.0009, 0.0025]
FLOOR, CEILING = 0.5427, 0.628
_pat = re.compile(r"ka([0-9.]+)_kb([0-9.]+)")


def load_grid(root, strategy):
    g = np.full((len(KBS), len(KAS)), np.nan)
    for mf in Path(root).rglob(f"*/{strategy}/metrics.json"):
        m = _pat.search(str(mf))
        if not m:
            continue
        ka, kb = float(m.group(1)), float(m.group(2))
        if ka not in KAS or kb not in KBS:
            continue
        try:
            v = json.loads(mf.read_text())["humaneval"]["pass@1"]
        except Exception:
            continue
        if v is not None:
            g[KBS.index(kb), KAS.index(ka)] = float(v)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, help="local dir with ka*_kb*/<strategy>/metrics.json")
    p.add_argument("--strategy", default="v2")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    g = load_grid(a.results_dir, a.strategy)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    im = ax.imshow(g, aspect="auto", cmap="RdYlGn", vmin=0, vmax=CEILING, origin="lower")
    ax.set_xticks(range(len(KAS))); ax.set_xticklabels([f"{k*100:g}%" for k in KAS])
    ax.set_yticks(range(len(KBS))); ax.set_yticklabels([f"{k*100:g}%" for k in KBS])
    ax.set_xlabel("kA = |weight| top-k (A)"); ax.set_ylabel("kB = code spot top-k' (B excluded)")
    ax.set_title(f"bridge {a.strategy} transplant pass@1 (qwen base←coder)\n"
                 f"floor(base)={FLOOR}  ceiling(coder)={CEILING}  (green→ceiling, red→0)")
    for i in range(len(KBS)):
        for j in range(len(KAS)):
            if not np.isnan(g[i, j]):
                ax.text(j, i, f"{g[i,j]:.3f}", ha="center", va="center", fontsize=8,
                        color="black")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("pass@1")
    # floor reference line on colorbar
    cbar.ax.axhline(FLOOR, color="blue", lw=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight")
    print(f"saved {a.out}")
    print("grid (rows=kB, cols=kA):"); print(np.array2string(g, precision=3))


if __name__ == "__main__":
    main()
