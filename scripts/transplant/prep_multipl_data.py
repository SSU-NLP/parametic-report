#!/usr/bin/env python3
"""Build per-language MultiPL-E completion parquets (name, prompt, tests, stop_tokens).

java/cpp/js/go come straight from nuprl/MultiPL-E (humaneval-{lang}). python is NOT a
MultiPL-E target (it is the source), so we reconstruct the same completion schema from the
original openai_humaneval: full program = prompt + completion + "\n" + test + check(entry).

usage: python3 scripts/transplant/prep_multipl_data.py --out-root <dir> [--langs py java cpp js go]
"""
import argparse
import ast
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

PY_STOP = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert "]


def build_python():
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for ex in ds:
        name = ex["task_id"].replace("/", "_")          # HumanEval/0 -> HumanEval_0
        tests = "\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        rows.append({"name": name, "prompt": ex["prompt"], "tests": tests, "stop_tokens": PY_STOP})
    return rows


def build_multipl(lang):
    ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", split="test")
    rows = []
    for ex in ds:
        st = ex.get("stop_tokens") or []
        if isinstance(st, str):
            try:
                st = ast.literal_eval(st)
            except Exception:
                st = [st]
        rows.append({"name": ex["name"], "prompt": ex["prompt"], "tests": ex["tests"],
                     "stop_tokens": list(st)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--langs", nargs="+", default=["py", "java", "cpp", "js", "go"])
    a = p.parse_args()
    for lang in a.langs:
        rows = build_python() if lang in ("py", "python") else build_multipl(lang)
        tbl = pa.Table.from_pylist(rows)
        d = a.out_root / lang
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, d / "test.parquet")
        print(f"{lang:5} -> {d/'test.parquet'}  ({len(rows)} problems)")


if __name__ == "__main__":
    main()
