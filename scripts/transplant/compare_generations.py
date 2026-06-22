#!/usr/bin/env python3
"""여러 strategy의 generations.jsonl completion을 쌍별로 비교.

rand/ndlo/vhi/vlo/v2가 전부 pass@1=0.2848로 동일한 게 "출력이 같아서"인지
(이식이 greedy 생성을 거의 안 바꿈) "다른 출력인데 pass 개수만 우연히 일치"인지 판별.

usage: compare_generations.py <root> <strat1> <strat2> ...
  root = .../results-bridge-10000-k0.05  (각 strat은 root/<strat>/generations.jsonl)
"""
import itertools
import json
import os
import sys


def load(root, s):
    f = os.path.join(root, s, "generations.jsonl")
    if not os.path.exists(f):
        print(f"MISSING {s} ({f})", flush=True)
        return None
    rows = [json.loads(l) for l in open(f)]
    print(f"{s}: {len(rows)} rows", flush=True)
    return {r["name"]: r["completion"] for r in rows}


def main():
    root = sys.argv[1]
    strats = sys.argv[2:]
    comp = {}
    for s in strats:
        c = load(root, s)
        if c is not None:
            comp[s] = c
    if len(comp) < 2:
        print("need >=2 present", flush=True); return
    names = sorted(set().union(*[set(c) for c in comp.values()]))
    print(f"\n=== 쌍별 identical completion 수 (/{len(names)}) ===", flush=True)
    for a, b in itertools.combinations(comp, 2):
        same = sum(comp[a].get(n) == comp[b].get(n) for n in names)
        print(f"{a:>6} vs {b:<6}: {same}/{len(names)} identical", flush=True)
    # 기준(첫 strat) 대비 얼마나 같은지 + 첫 차이 예시
    base = strats[0]
    if base in comp:
        print(f"\n=== '{base}' 대비 첫 차이 예시 ===", flush=True)
        for s in strats[1:]:
            if s not in comp:
                continue
            diff = [n for n in names if comp[base].get(n) != comp.get(s, {}).get(n)]
            print(f"{base} vs {s}: {len(diff)} differ", flush=True)
            if diff:
                n = diff[0]
                print(f"  [{n}] {base}: {comp[base].get(n)[:160]!r}", flush=True)
                print(f"  [{n}] {s}: {comp[s].get(n)[:160]!r}", flush=True)


if __name__ == "__main__":
    main()
