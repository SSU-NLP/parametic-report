#!/usr/bin/env python3
"""Classify BigCode humanevalpack java per-problem results into failure types.

BigCode's harness writes logs.json (task_id -> [[completion_id, {passed, result}]])
when it scores with --allow_code_execution. pass@1 only tells you the pass rate; this
buckets every problem by *why* it failed — compile error vs runtime error vs wrong
answer vs timeout — which distinguishes "model emits garbage that won't compile"
(total collapse) from "compiles but logic is wrong" (ability damaged but coherent).
"""
import glob
import json
import os
import sys
from collections import Counter


def classify(res):
    passed = bool(res.get("passed", False))
    result = str(res.get("result", "")).lower()
    if passed or result == "passed":
        return "passed"
    if "compil" in result:
        return "compile_error"
    if "timed out" in result or "timeout" in result:
        return "timeout"
    if "assert" in result or "wrong" in result or "expected" in result:
        return "wrong_answer"
    return "runtime_error"


def main():
    files = []
    for a in sys.argv[1:]:
        files += glob.glob(a)
    cols = ["passed", "wrong_answer", "runtime_error", "compile_error", "timeout"]
    print(f"{'condition':14} " + " ".join(f"{c:>13}" for c in cols) + f" {'total':>6}")
    print("-" * 92)
    for f in sorted(files):
        cond = os.path.basename(os.path.dirname(f))
        logs = json.load(open(f))
        c = Counter()
        for _tid, lst in logs.items():
            for item in lst:
                res = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                c[classify(res)] += 1
        tot = sum(c.values())
        print(f"{cond:14} " + " ".join(f"{c[col]:>13}" for col in cols) + f" {tot:>6}")


if __name__ == "__main__":
    main()
