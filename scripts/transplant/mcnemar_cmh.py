#!/usr/bin/env python3
"""Language-stratified McNemar + Cochran–Mantel–Haenszel for the multi-language transplant eval.

Layout: <root>/<strategy>/<lang>/results.jsonl  (per-problem `name`,`passed`).

Per language (distinct problems → within-stratum McNemar is valid):
  b = ref✓ strat✗, c = ref✗ strat✓, χ² = (|b-c|-1)²/(b+c), p.
Combined across language strata (matched-pairs CMH = pooled discordant):
  χ²_CMH = (|Σ(b-c)|-1)² / Σ(b+c), 1 df.
CAVEAT printed: the SAME HumanEval problem appears once per language, so cross-language
copies are correlated → CMH is mildly anti-conservative (treat as an upper bound on
significance). Directional consistency across the 5 languages (sign of net per lang) is the
robustness check that pooling cannot fake.

usage: mcnemar_cmh.py <root> <ref> <strat1> [strat2 ...] [--langs py java cpp js go]
"""
import json
import os
import sys
from math import erfc, sqrt


def load(root, s, lang):
    f = os.path.join(root, s, lang, "results.jsonl")
    if not os.path.exists(f):
        return None
    return {r["name"]: bool(r.get("passed")) for r in (json.loads(l) for l in open(f) if l.strip())}


def chi2_p(chi2):
    return erfc(sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0


def bc(ref, d):
    names = sorted(set(ref) & set(d))
    b = sum(ref[n] and not d[n] for n in names)
    c = sum((not ref[n]) and d[n] for n in names)
    return b, c, len(names)


def main():
    argv = sys.argv[1:]
    langs = ["py", "java", "cpp", "js", "go"]
    if "--langs" in argv:
        i = argv.index("--langs"); langs = argv[i + 1:]; argv = argv[:i]
    root, ref, strats = argv[0], argv[1], argv[2:]
    print(f"root={root} ref={ref} langs={langs}\n")
    for s in strats:
        print(f"### {s} vs {ref} ###")
        hdr = f"{'lang':>6} | {'b(R✓S✗)':>8} {'c(R✗S✓)':>8} {'n':>4} | {'net=c-b':>7} {'χ²':>6} {'p':>6}"
        print(hdr); print("-" * len(hdr))
        sb = sc = 0; nets = []
        for lang in langs:
            a = load(root, ref, lang); d = load(root, s, lang)
            if a is None or d is None:
                print(f"{lang:>6} | MISSING"); continue
            b, c, n = bc(a, d)
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
            print(f"{lang:>6} | {b:>8} {c:>8} {n:>4} | {c - b:>+7} {chi2:>6.2f} {chi2_p(chi2):>6.3f}")
            sb += b; sc += c; nets.append(c - b)
        cmh = (abs(sb - sc) - 1) ** 2 / (sb + sc) if (sb + sc) > 0 else 0.0
        pos = sum(1 for x in nets if x > 0); neg = sum(1 for x in nets if x < 0)
        print(f"{'CMH':>6} | {sb:>8} {sc:>8} {'':>4} | {sc - sb:>+7} {cmh:>6.2f} {chi2_p(cmh):>6.3f}")
        print(f"  directional: {pos} langs net>0, {neg} langs net<0 (of {len(nets)})  "
              f"[caveat: cross-lang same-problem correlation → CMH anti-conservative]\n")


if __name__ == "__main__":
    main()
