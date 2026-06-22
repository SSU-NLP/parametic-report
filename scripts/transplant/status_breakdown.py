#!/usr/bin/env python3
"""각 strategy의 summary.json을 모아 채점 status를 세분화 표로.

pass@1(통과 개수)만으론 둔감 — compile_error / runtime_error(=wrong answer 포함) /
timeout으로 쪼개면 "다른 출력인데 비슷한 pass"의 내막이 보인다.

usage: status_breakdown.py <root> <strat1> <strat2> ...
"""
import json
import os
import sys


def main():
    root = sys.argv[1]
    strats = sys.argv[2:]
    hdr = f"{'strat':>8} | {'pass':>4} {'comp_err':>8} {'comp_to':>7} {'run_err':>7} {'run_to':>6} | {'pass@1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in strats:
        f = os.path.join(root, s, "summary.json")
        if not os.path.exists(f):
            print(f"{s:>8} | MISSING ({f})")
            continue
        d = json.load(open(f))
        t = d.get("total", 0)
        print(
            f"{s:>8} | {d.get('passed',0):>4} {d.get('compile_error',0):>8} "
            f"{d.get('compile_timeout',0):>7} {d.get('runtime_error',0):>7} "
            f"{d.get('runtime_timeout',0):>6} | {d.get('pass_at_1',0):>7.4f}  (n={t})"
        )


if __name__ == "__main__":
    main()
