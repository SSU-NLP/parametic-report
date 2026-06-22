#!/usr/bin/env python3
"""base 대비 각 strategy의 per-problem flip을 McNemar 검정.

pass@1(통과 개수)이 같아도 *어떤 문제*가 바뀌었는지는 다르다. base vs strategy를 문제별
paired로 보고, base는 맞고 strat는 틀린(b) / base는 틀리고 strat는 맞은(c) 칸으로 McNemar:
  χ² = (|b−c|−1)² / (b+c)   (continuity-corrected, 1 dof)
effect가 노이즈인지(p 큼) 실재인지(p 작음 + net=c−b 일관) 가린다.

usage: mcnemar.py <root> <ref> <strat1> <strat2> ...
  root = .../results-bridge-10000-k0.05  (각 strat은 root/<strat>/results.jsonl)
  ref  = 기준 (보통 base)
"""
import json
import os
import sys
from math import erfc, sqrt


def load(root, s):
    f = os.path.join(root, s, "results.jsonl")
    if not os.path.exists(f):
        return None
    return {r["name"]: bool(r.get("passed")) for r in (json.loads(l) for l in open(f) if l.strip())}


def chi2_p(chi2):
    # survival of χ² with 1 dof = erfc(sqrt(χ²/2))
    return erfc(sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0


def main():
    root = sys.argv[1]
    ref = sys.argv[2]
    strats = sys.argv[3:]
    base = load(root, ref)
    if base is None:
        print(f"MISSING ref {ref}"); return
    print(f"ref={ref} (passed={sum(base.values())}/{len(base)})")
    hdr = f"{'strat':>8} | {'b(R✓S✗)':>8} {'c(R✗S✓)':>8} {'both':>5} {'neither':>7} | {'net=c-b':>7} {'χ²':>7} {'p':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in strats:
        d = load(root, s)
        if d is None:
            print(f"{s:>8} | MISSING"); continue
        names = sorted(set(base) & set(d))
        b = sum(base[n] and not d[n] for n in names)
        c = sum((not base[n]) and d[n] for n in names)
        both = sum(base[n] and d[n] for n in names)
        neither = sum((not base[n]) and (not d[n]) for n in names)
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p = chi2_p(chi2)
        print(f"{s:>8} | {b:>8} {c:>8} {both:>5} {neither:>7} | {c - b:>+7} {chi2:>7.2f} {p:>7.3f}")


if __name__ == "__main__":
    main()
