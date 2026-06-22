#!/usr/bin/env python3
"""지금까지의 모든 transplant 실험을 시트별 엑셀로 정리.

데이터 원본:
  - 단순 spot ablation: reports/transplant_ablation/{cowork,instruct}/*.json  (원본 summary)
  - Bridge 재현:        reports/bridge_repro/s{1024,10000}/*.json
  - k스윕/제외/통제:     reports/results_sequential.csv  (순차 분기점 박제)

usage: python3 scripts/transplant/build_excel.py  (repo 루트에서)
출력: reports/experiments.xlsx
"""
import csv
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

R = "reports"
HDR_FILL = PatternFill("solid", fgColor="DDDDDD")


def add_sheet(wb, name, headers, rows):
    ws = wb.create_sheet(name)
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = HDR_FILL
    for r in rows:
        ws.append(r)
    for i, h in enumerate(headers, 1):
        w = max(len(str(h)), *(len(str(r[i - 1])) for r in rows)) if rows else len(str(h))
        ws.column_dimensions[chr(64 + i) if i <= 26 else "A"].width = min(w + 2, 60)
    ws.freeze_panes = "A2"
    return ws


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    wb = Workbook()
    wb.remove(wb.active)

    # ── 0. README/맥락 시트 ──
    add_sheet(wb, "_README", ["항목", "내용"], [
        ["실험", "Qwen2.5-1.5B(base) ← Qwen2.5-Coder-1.5B(coder) 파라미터 이식"],
        ["평가", "MultiPL-E HumanEval-Java (completion, 158문제, greedy pass@1)"],
        ["기준", "base=0.2595(41/158), coder=0.3734(59/158, +11.4pp)"],
        ["시트1 ablation_cowork", "단순 spot(|grad·param| top1%) 이식 → 전부 붕괴 (완성=동료 셋업)"],
        ["시트2 ablation_instruct", "단순 spot, instruct+BigCode synthesize (ppl+pass@1)"],
        ["시트3 bridge_repro", "정식 차집합 spot(A∖B) bridge 이식, v1/v2/v3 × sample"],
        ["시트4 bridge_ksweep", "A=B=top-k 키움, v2 교집합 (순차)"],
        ["시트5 bridge_exclusion", "A=1% 고정, 코어 B만 키워 제외 (순차)"],
        ["시트6 bridge_control", "위치/값/인과 통제: rand/perm/ndhi/ndlo/vhi/vlo/reverse (순차)"],
        ["주의", "k스윕/제외/통제는 순차(batch=1). 배치 분기점(2026-06-22) 이후 재측정 예정"],
        ["주의2", "통제 결론: bridge가 random과 pass@1·status 구별 안 됨 (위치 특이성 불지지)"],
    ])

    # ── 1. ablation cowork (생모델 completion) ──
    rows = []
    for c in ["base", "coder", "v1", "v2", "v3a", "v3b", "v3c", "v3d", "v3ctrl"]:
        d = jload(f"{R}/transplant_ablation/cowork/{c}.json")
        if d:
            rows.append([c, d.get("pass_at_1"), d.get("passed"), d.get("total"),
                         d.get("compile_error"), d.get("runtime_error"), d.get("runtime_timeout")])
    add_sheet(wb, "1_ablation_cowork",
              ["condition", "pass@1", "passed", "total", "compile_err", "runtime_err", "timeout"], rows)

    # ── 2. ablation instruct (BigCode synthesize) ──
    rows = []
    for c in ["base-it", "coder-it", "v1-it", "v2-it", "v3a-it", "v3b-it", "v3c-it", "v3d-it", "v3ctrl-it"]:
        d = jload(f"{R}/transplant_ablation/instruct/{c}.json")
        if d:
            rows.append([c, d.get("ppl", {}).get("ppl"), d.get("humaneval", {}).get("pass@1"),
                         d.get("transplant", {}).get("k"), d.get("transplant", {}).get("transplanted_tensors")])
    add_sheet(wb, "2_ablation_instruct", ["condition", "ppl", "pass@1", "k", "transplanted_tensors"], rows)

    # ── 3. bridge repro ──
    rows = []
    for s in ["1024", "10000"]:
        for c in ["base", "coder", "v1", "v2", "v3"]:
            d = jload(f"{R}/bridge_repro/s{s}/{c}.json")
            if d:
                rows.append([s, c, d.get("pass_at_1"), d.get("passed"), d.get("total"),
                             d.get("compile_error"), d.get("runtime_error"), d.get("runtime_timeout")])
    add_sheet(wb, "3_bridge_repro",
              ["sample", "condition", "pass@1", "passed", "total", "compile_err", "runtime_err", "timeout"], rows)

    # ── 4/5/6. CSV에서 ksweep/exclusion/control ──
    csv_rows = list(csv.DictReader(open(f"{R}/results_sequential.csv")))

    def csv_sheet(name, exp):
        rows = []
        for r in csv_rows:
            if r["experiment"] == exp:
                rows.append([r["k"], r["core_k"], r["strategy"], r["decode"], r["pass_at_1"],
                             r["passed"], r["compile_error"], r["runtime_error"], r["timeout"],
                             r["selected"], r["notes"]])
        add_sheet(wb, name, ["k", "core_k", "strategy", "decode", "pass@1", "passed",
                             "compile_err", "runtime_err", "timeout", "selected", "notes"], rows)

    csv_sheet("4_bridge_ksweep", "ksweep")
    csv_sheet("5_bridge_exclusion", "exclusion")
    csv_sheet("6_bridge_control", "control")

    # ── 7/8/9. 배치(분기점 이후) 시트 + origin + 순차vs배치 비교 ──
    bpath = f"{R}/results_batch.csv"
    if os.path.exists(bpath):
        batch_rows = list(csv.DictReader(open(bpath)))

        def bsheet(name, exp):
            rows = [[r["k"], r["core_k"], r["strategy"], r["pass_at_1"], r["passed"],
                     r["compile_error"], r["runtime_error"], r["timeout"], r["selected"], r["notes"]]
                    for r in batch_rows if r["experiment"] == exp]
            add_sheet(wb, name, ["k", "core_k", "strategy", "pass@1", "passed",
                                 "compile_err", "runtime_err", "timeout", "selected", "notes"], rows)

        bsheet("7_batch_ksweep", "ksweep")
        bsheet("8_batch_exclusion", "exclusion")
        bsheet("9_batch_control", "control")
        orows = [[r["strategy"], r["pass_at_1"], r["passed"], r["compile_error"],
                  r["runtime_error"], r["notes"]]
                 for r in batch_rows if r["experiment"] == "origin"]
        add_sheet(wb, "9b_batch_origin",
                  ["strategy", "pass@1", "passed", "compile_err", "runtime_err", "notes"], orows)

        # 10. 순차 vs 배치 (같은 k·strategy의 pass@1 나란히)
        def key(r):
            return (r["experiment"], r["k"], r["core_k"], r["strategy"])
        seqd = {key(r): r["pass_at_1"] for r in csv_rows}
        cmp_rows = []
        for r in batch_rows:
            sk = seqd.get(key(r))
            if sk:
                d = round(float(r["pass_at_1"]) - float(sk), 4)
                cmp_rows.append([r["experiment"], r["k"], r["core_k"], r["strategy"],
                                 sk, r["pass_at_1"], d])
        add_sheet(wb, "10_seq_vs_batch",
                  ["experiment", "k", "core_k", "strategy", "seq_pass@1", "batch_pass@1", "diff"], cmp_rows)

    out = f"{R}/experiments.xlsx"
    wb.save(out)
    print(f"saved {out} ({len(wb.sheetnames)} sheets: {wb.sheetnames})")


if __name__ == "__main__":
    main()
