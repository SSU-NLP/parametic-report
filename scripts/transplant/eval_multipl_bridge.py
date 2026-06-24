#!/usr/bin/env python3
"""Multi-language MultiPL-E completion eval for a transplanted model — per-problem pass/fail.

Builds the transplant ONCE (reusing eval_bridge_cowork.build over a mask dir = spot OR
bridge), saves it, then for each language generates greedy completions and executes every
problem (scripts/transplant/multipl_exec.run_one). Writes results-{lang}.jsonl + per-lang
summary + an aggregate. Per-problem results feed language-stratified McNemar / CMH.

strategy: base/coder (originals) or any transplant strategy in eval_bridge_cowork
  (v1/v2/v3/rand/perm/ndhi/ndlo/vhi/vlo/reverse). reverse evaluates the donor (coder) side.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))            # scripts/ for the java cowork helpers
import multipl_exec as mx                        # noqa: E402
import eval_humaneval_java_cowork as cw          # noqa: E402 (load_tasks/generate_completions/truncate)
import eval_bridge_cowork as ebc                 # noqa: E402 (build())


def score_lang(rows, lang, timeout, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for r in rows:
        outcome = mx.run_one(r["source"], lang, timeout=timeout)
        scored.append({**r, **outcome})
    passed = sum(1 for r in scored if r["passed"]); total = len(scored)
    summary = {"language": lang, "total": total, "passed": passed,
               "pass_at_1": passed / total if total else 0.0}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--mask-base", type=Path, help="recipient mask dir (spot OR bridge)")
    p.add_argument("--mask-coder", type=Path, help="donor mask dir")
    p.add_argument("--data-root", required=True, type=Path, help="dir with {lang}/test.parquet")
    p.add_argument("--langs", nargs="+", default=["py", "java", "cpp", "js", "go"])
    p.add_argument("--result-root", required=True, type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("/work/mpl"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    a.work_dir.mkdir(parents=True, exist_ok=True)

    if a.strategy == "base":
        model_dir = a.base_model
    elif a.strategy == "coder":
        model_dir = a.donor_model
    else:
        if not a.mask_base or not a.mask_coder:
            raise SystemExit("--mask-base/--mask-coder required for transplant strategies")
        model, tok = ebc.build(a.base_model, a.donor_model, a.mask_base, a.mask_coder, a.strategy, token, dtype)
        md = a.work_dir / "model"; md.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(md); tok.save_pretrained(md)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = str(md)

    aggregate = {"strategy": a.strategy, "by_language": {}}
    for lang in a.langs:
        data = a.data_root / lang / "test.parquet"
        tasks = cw.load_tasks(str(data), limit=(a.limit or None))
        gen_args = argparse.Namespace(model=model_dir, dtype="bfloat16", device="auto",
                                      max_new_tokens=a.max_new_tokens, temperature=0.0, top_p=0.95,
                                      batch_size=a.batch_size, local_files_only=False)
        rows = cw.generate_completions(gen_args, tasks)
        summ = score_lang(rows, lang, a.timeout, a.result_root / lang)
        aggregate["by_language"][lang] = {k: summ[k] for k in ("total", "passed", "pass_at_1")}
        print(f"[{a.strategy}] {lang}: pass@1={summ['pass_at_1']:.4f} ({summ['passed']}/{summ['total']})", flush=True)

    tot = sum(v["total"] for v in aggregate["by_language"].values())
    pas = sum(v["passed"] for v in aggregate["by_language"].values())
    aggregate["pooled_pass_at_1"] = pas / tot if tot else 0.0
    aggregate["pooled_passed"] = pas
    aggregate["pooled_total"] = tot
    a.result_root.mkdir(parents=True, exist_ok=True)
    (a.result_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
