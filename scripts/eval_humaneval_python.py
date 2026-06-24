#!/usr/bin/env python3
"""Original Python HumanEval (openai_humaneval, 164 problems) pass@1 via lm-eval-harness,
with the model's chat template applied — the correct setup for instruct models.

Paper reproduction (Kim et al. "Coding Spot"): the code spot is zeroed and the damaged
checkpoint is scored on the ORIGINAL HumanEval (openai_humaneval, 164). We use lm-eval's
`humaneval_instruct` task + `--apply_chat_template`: same data + pass@1, but with a
user-instruction prompt, an assistant gen_prefix, and a ```python code-block extraction
filter — the right setup for instruct models. (Plain `humaneval` is completion-style and
scores a chat model's markdown reply as 0 even when healthy — fair across models, 논문도 instruct.)

This driver:
  1. (optional) damages the model from a boolean mask dir (reuses damage/damage_model.py),
  2. runs lm-eval `humaneval` (+ chat template) with code execution,
  3. writes one metrics.json with pass@1.

The pass@1 extractor is a pure function (unit-tested in tests/test_eval_humaneval_python.py).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def extract_pass_at_1(results_json):
    """Pull pass@1 out of lm-eval results json (results.humaneval_instruct['pass@1,...'])."""
    res = results_json.get("results", results_json)
    block = res.get("humaneval_instruct") or res.get("humaneval") or res.get("humaneval_greedy") or {}
    if not isinstance(block, dict):
        return None
    for key, value in block.items():
        name = key.split(",")[0]
        if name == "pass@1" and not name.endswith("_stderr"):
            return value
    return None


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


def run_humaneval(model_dir, out_dir, token, apply_chat, limit, gen_max_toks, log_samples=False):
    print(f"\n== HumanEval (lm-eval, chat={apply_chat}) ({model_dir}) ==", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},dtype=bfloat16",
        # humaneval_instruct: same openai_humaneval data + pass@1, but with a user-instruction
        # doc_to_text, an assistant gen_prefix, and a code-block extraction filter
        # (build_predictions_instruct). Plain `humaneval` is completion-style and CANNOT parse a
        # chat model's ```python ...``` reply → scores 0 even for a healthy model.
        "--tasks", "humaneval_instruct",
        "--confirm_run_unsafe_code",          # humaneval executes generated code
        "--batch_size", "1",
        "--output_path", str(out_dir),
    ]
    if apply_chat:
        cmd += ["--apply_chat_template"]
    if limit:
        cmd += ["--limit", str(limit)]
    if gen_max_toks:
        cmd += ["--gen_kwargs", f"max_gen_toks={gen_max_toks}"]
    if log_samples:
        # persist per-problem generations (samples_*.jsonl) so we can inspect what the model
        # actually produced — e.g. whether a damaged `code` model emits garbage (genuine 0) or
        # coherent code truncated by max_gen_toks (artificial 0).
        cmd += ["--log_samples"]
    env = dict(os.environ)
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if token:
        env.setdefault("HF_TOKEN", token)
    print("lm_eval:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    files = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True))
    if not files:
        files = sorted(glob.glob(str(out_dir / "**" / "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no lm_eval results json under {out_dir}")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Original Python HumanEval pass@1 (lm-eval + chat template) on an optionally-damaged model.")
    p.add_argument("--label", required=True, help="condition: code|bottom|random_seed1|original")
    p.add_argument("--model-id", required=True, help="clean original HF model id")
    p.add_argument("--mask-dir", default=None, help="boolean mask dir to zero; omit/none for clean model")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--apply-chat-template", action="store_true", help="format prompts with the model chat template (instruct models)")
    p.add_argument("--gen-max-toks", type=int, default=None, help="cap generation length (collapsed models)")
    p.add_argument("--log-samples", action="store_true", help="save per-problem generations (samples_*.jsonl) next to --output")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("/work/pr_he"))
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

    he_dir = a.work_dir / "he"
    results = run_humaneval(model_dir, he_dir, token, a.apply_chat_template, a.limit or None,
                            a.gen_max_toks, log_samples=a.log_samples)
    result = {
        "label": a.label,
        "model": a.model_id,
        "k": a.k,
        "mask_dir": a.mask_dir if has_mask else None,
        "apply_chat_template": a.apply_chat_template,
        "gen_max_toks": a.gen_max_toks,
        "humaneval": {"pass@1": extract_pass_at_1(results)},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # persist generations beside the metrics so they survive the job-local /work wipe
    if a.log_samples:
        import shutil
        for s in glob.glob(str(he_dir / "**" / "samples_*.jsonl"), recursive=True):
            dst = a.output.parent / Path(s).name
            shutil.copy(s, dst)
            print(f"saved generations -> {dst}", flush=True)
    print(f"\nwrote {a.output}\n{json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
