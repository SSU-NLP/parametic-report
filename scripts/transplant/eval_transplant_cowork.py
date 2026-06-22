#!/usr/bin/env python3
"""Evaluate transplant conditions with the colleague's HumanEval-Java (completion) harness.

Cross-validation: reuse OUR transplant (build_transplanted_model) to construct the
modified base model, then score it with the COLLEAGUE's MultiPL-E HumanEval-Java
completion eval (eval_humaneval_java_cowork) — i.e. base (non-instruct) model + raw
continuation prompt, NOT our instruct-style BigCode synthesize. This matches the setup
under which the colleague saw +2%.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

import eval_humaneval_java_cowork as cw  # noqa: E402
from eval_humanevalpack_java import build_transplanted_model  # noqa: E402

ORIGINALS = ("base", "coder")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--base-scores", nargs="+", type=Path, default=[])
    p.add_argument("--donor-scores", nargs="+", type=Path, default=[])
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--data", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/cw_model"))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.strategy == "base":
        model_dir = args.base_model
    elif args.strategy == "coder":
        model_dir = args.donor_model
    else:
        model, tok, applied = build_transplanted_model(
            args.base_model, args.donor_model, args.base_scores, args.donor_scores,
            args.k, args.strategy, token, dtype, 0,
        )
        args.work_dir.mkdir(parents=True, exist_ok=True)
        md = args.work_dir / "model"
        md.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(md)
        tok.save_pretrained(md)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = str(md)
        print(f"transplant {args.strategy}: applied={applied}", flush=True)

    # Colleague's completion harness
    tasks = cw.load_tasks(args.data, limit=args.limit)
    gen_args = argparse.Namespace(
        model=model_dir, dtype="bfloat16", device="auto",
        max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=0.95,
        local_files_only=False,
    )
    rows = cw.generate_completions(gen_args, tasks)
    Path(args.result).mkdir(parents=True, exist_ok=True)
    cw.write_jsonl(str(Path(args.result) / "generations.jsonl"), rows)
    score_args = argparse.Namespace(result=args.result, timeout=10)
    summary = cw.score_results(score_args, rows)
    summary["strategy"] = args.strategy
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
