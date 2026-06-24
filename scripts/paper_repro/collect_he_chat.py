#!/usr/bin/env python3
"""Collect chat-format Python-HumanEval (lm-eval `humaneval_instruct`) pass@1 from the durable
volume and rewrite the HumanEval column of reports/paper_repro_table1.csv with the corrected
(chat) numbers. General-task columns are kept as-is (format-independent).

Run from repo root (host):
    python3 scripts/paper_repro/collect_he_chat.py [--download]

`--download` first pulls /shared/<ns>/paper-repro/results-he-chat -> a local tmp dir via vesslctl.
Without it, reads an already-downloaded tree (default tmp path).
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "reports" / "paper_repro_table1.csv"
LOCAL = Path("/tmp/he_chat_results")

# CSV k-label (percent string) <-> results-he-chat top<frac> dir
KFRAC = {"0.0025%": "0.000025", "0.01%": "0.0001", "0.09%": "0.0009", "0.25%": "0.0025"}
# original is k-independent; we always ran it under the 0.25% (top0.0025) sweep
ORIG_FRAC = "0.0025"
CONTROL_FRAC = "0.0025"  # controls (random/bottom) run only at the largest k, like the paper


def vessl_env():
    env = {}
    for line in (REPO / "scripts" / "vessl" / "config.sh").read_text().splitlines():
        line = line.strip()
        for key in ("VESSL_OBJECT_VOL", "VESSL_NS"):
            if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
                env[key] = line.split("=", 1)[1].strip().strip('"').strip("'")
    return env


def download():
    e = vessl_env()
    vol, ns = e.get("VESSL_OBJECT_VOL"), e.get("VESSL_NS")
    # config.sh may compute these; fall back to known values
    vol = vol or "objvol-gsvyr0eu87wt"
    ns = ns or "seonghyeon/parametic"
    LOCAL.mkdir(parents=True, exist_ok=True)
    prefix = f"{ns}/paper-repro/results-he-chat"
    print(f"downloading {prefix} -> {LOCAL}", flush=True)
    subprocess.run(["vesslctl", "volume", "download", vol, str(LOCAL),
                    "--remote-prefix", prefix, "--overwrite"], check=False)


def read_pass1(model, frac, label):
    """Find metrics.json under the downloaded tree and return pass@1 (or None)."""
    for base in LOCAL.rglob(f"results-he-chat/{model}/top{frac}/{label}/metrics.json"):
        d = json.loads(base.read_text())
        return d.get("humaneval", {}).get("pass@1")
    # tree may be rooted differently depending on how vesslctl lays it out
    for base in LOCAL.rglob(f"{model}/top{frac}/{label}/metrics.json"):
        d = json.loads(base.read_text())
        return d.get("humaneval", {}).get("pass@1")
    return None


def chat_value(model, condition, k_label):
    if condition == "original":
        return read_pass1(model, ORIG_FRAC, "original")
    if condition == "code":
        return read_pass1(model, KFRAC.get(k_label, ""), "code")
    # random_seed1 / bottom — only at the largest k
    return read_pass1(model, CONTROL_FRAC, condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download:
        download()

    rows = list(csv.reader(CSV.open()))
    header, body = rows[0], rows[1:]
    he_idx = next(i for i, h in enumerate(header) if h.startswith("HumanEval"))
    header[he_idx] = "HumanEval(chat)"  # be explicit: humaneval_instruct + chat template

    updated, missing = 0, []
    for r in body:
        model, condition, k_label = r[0], r[1], r[2]
        v = chat_value(model, condition, k_label)
        if v is None:
            missing.append(f"{model}/{condition}/{k_label}")
            continue
        r[he_idx] = round(float(v), 4)
        updated += 1

    with CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)

    print(f"updated {updated} HumanEval(chat) cells; missing {len(missing)}: {missing}", flush=True)
    # echo the table
    print("\n".join(",".join(str(c) for c in row) for row in [header] + body))


if __name__ == "__main__":
    main()
