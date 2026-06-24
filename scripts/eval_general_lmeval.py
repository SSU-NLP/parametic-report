#!/usr/bin/env python3
"""General-task accuracy (GSM8K/HellaSwag/MMLU/TruthfulQA/WinoGrande) on an optionally-damaged model.

Paper reproduction (Kim et al. "Coding Spot", Table 1): besides HumanEval, the paper reports
the 5 general benchmarks to show the code spot's damage is (relatively) code-specific. This
driver reuses the validated damage/damage_model.py to zero the spot, then runs the EleutherAI
lm-evaluation-harness for each task at its paper n-shot, and writes one metrics.json.

n-shot per task is fixed here (paper-repro reproducibility, see reports/HANDOFF.md "B 본 실험 설정").
Tasks are grouped by n-shot so lm_eval runs once per distinct shot count.

The result parser is a pure function, unit-tested without lm-eval in
tests/test_eval_general_lmeval.py.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# paper n-shot per task (reproducibility-fixed; lm-eval task ids)
FEWSHOT = {"gsm8k": 5, "hellaswag": 10, "mmlu": 5, "truthfulqa_mc2": 0, "winogrande": 5}
# primary metric per task (lm-eval key prefix before the ",<filter>" suffix)
PRIMARY_METRIC = {
    "gsm8k": "exact_match",
    "hellaswag": "acc_norm",
    "mmlu": "acc",
    "truthfulqa_mc2": "acc",
    "winogrande": "acc",
}


def extract_lmeval_scores(results_json, primary):
    """lm-eval results json -> {task: primary-metric value}. Skips stderr keys."""
    res = results_json.get("results", results_json)
    out = {}
    for task, metric in primary.items():
        block = res.get(task)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            name = key.split(",")[0]
            if name == metric and not name.endswith("_stderr"):
                out[task] = value
                break
    return out


def damage(model_id, mask_dir, out_dir, token):
    cmd = [
        sys.executable, str(REPO / "damage" / "damage_model.py"),
        "--weights_folder", str(mask_dir),
        "--original_model", model_id,
        "--output_dir", str(out_dir),
    ]
    print("damage:", " ".join(cmd), flush=True)
    env = dict(os.environ)
    if token:
        env.setdefault("HF_TOKEN", token)
    subprocess.run(cmd, check=True, env=env)


def run_lmeval(model_dir, tasks, num_fewshot, batch_size, out_dir, token, limit, gen_max_toks=None):
    """Run lm_eval for a set of tasks at a single n-shot; return merged results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = "bfloat16"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},dtype={dtype}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    # cap generation length — a damaged (collapsed) model never emits EOS and fills max_gen_toks,
    # making generative tasks (gsm8k) crawl. Only affects generate_until tasks; loglikelihood
    # tasks (hellaswag/mmlu/winogrande/truthfulqa_mc2) ignore it.
    if gen_max_toks:
        cmd += ["--gen_kwargs", f"max_gen_toks={gen_max_toks}"]
    env = dict(os.environ)
    if token:
        env.setdefault("HF_TOKEN", token)
    print("lm_eval:", " ".join(cmd), flush=True)
    # retry transient failures (e.g. HF rate-limit / network) a couple times
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, env=env)
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            print(f"lm_eval failed (attempt {attempt + 1}/3) — retrying in 90s", flush=True)
            time.sleep(90)
    # lm_eval writes results_*.json under out_dir (possibly in a model-named subdir)
    files = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True))
    if not files:
        files = sorted(glob.glob(str(out_dir / "**" / "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no lm_eval results json under {out_dir}")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="General-task (lm-eval) accuracy on an optionally-damaged model.")
    p.add_argument("--label", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--mask-dir", default=None)
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--tasks", default=",".join(FEWSHOT), help="subset of the 5 tasks (default: all)")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--gen-max-toks", type=int, default=None,
                   help="cap generate_until length (use for collapsed/damaged models so gsm8k doesn't run to max)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("/work/pr_general"))
    a = p.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    a.work_dir.mkdir(parents=True, exist_ok=True)

    has_mask = a.mask_dir is not None and str(a.mask_dir).lower() != "none"
    if has_mask:
        dmg = a.work_dir / "damaged"
        damage(a.model_id, a.mask_dir, dmg, token)
        model_dir = str(dmg)
    else:
        model_dir = a.model_id

    tasks = [t for t in a.tasks.split(",") if t]
    # group tasks by their fixed n-shot so each lm_eval call uses one --num_fewshot
    by_shot = {}
    for t in tasks:
        by_shot.setdefault(FEWSHOT[t], []).append(t)

    scores = {}
    errors = {}
    # each n-shot group is independent: one group failing (e.g. a dataset rate-limited)
    # must not block the others from being evaluated.
    for shot, group in sorted(by_shot.items()):
        try:
            results = run_lmeval(model_dir, group, shot, a.batch_size,
                                 a.work_dir / f"shot{shot}", token, a.limit or None, a.gen_max_toks)
            scores.update(extract_lmeval_scores(results, PRIMARY_METRIC))
        except Exception as exc:
            print(f"[warn] shot{shot} group {group} failed: {exc}", flush=True)
            for t in group:
                errors[t] = repr(exc)

    result = {
        "label": a.label,
        "model": a.model_id,
        "k": a.k,
        "mask_dir": a.mask_dir if has_mask else None,
        "fewshot": {t: FEWSHOT[t] for t in tasks},
        "general": scores,
        "errors": errors or None,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {a.output}\n{json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
