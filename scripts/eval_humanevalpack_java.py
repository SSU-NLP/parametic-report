#!/usr/bin/env python3
"""One transplant condition, end to end: build the model, then score it.

Runs inside a single GPU job (one VESSL job per condition). Given the base and donor
(coder) HF ids plus their java grad·param score dirs, it:

  1. reconstructs the per-tensor base/coder spots (reusing the mask pipeline helpers),
  2. transplants coder weights into the base for a chosen ``--strategy`` (in-memory),
  3. saves the transplanted base (so BigCode can load it by path),
  4. measures java held-out PPL (reusing evaluate_masked_ppl) and HumanEvalPack java
     pass@1 (BigCode evaluation harness),
  5. writes a single metrics.json.

``--strategy base`` / ``coder`` skip the transplant and score the original model — the
floor and ceiling references.

The mapping itself lives in transplant_mapping.py and is unit-tested without a GPU.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from arch_adapter import layer_pt_files
from create_approx_spot_masks import load_mean_score
from evaluate_masked_ppl import BinDataset, evaluate_loaded_model
from transplant_mapping import STRATEGIES, transplant_param

ORIGINALS = ("base", "coder")


def build_transplanted_model(base_id, donor_id, base_score_dirs, donor_score_dirs, k, strategy, token, dtype, seed):
    """Load base+coder on CPU, transplant per target tensor in place, return base+tok."""
    print(f"loading base={base_id} donor={donor_id} (cpu, {dtype})", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, token=token, torch_dtype=dtype)
    coder = AutoModelForCausalLM.from_pretrained(donor_id, token=token, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)

    base_params = dict(base.named_parameters())
    coder_params = dict(coder.named_parameters())

    files = layer_pt_files(base_score_dirs[0])
    if not files:
        raise SystemExit(f"no target score tensors in {base_score_dirs[0]}")

    applied = 0
    skipped = 0
    with torch.no_grad():
        for idx, score_path in enumerate(files, 1):
            name = score_path.stem
            bp = base_params.get(name)
            cp = coder_params.get(name)
            if bp is None or cp is None:
                print(f"skip {name}: missing in base/coder params", flush=True)
                skipped += 1
                continue
            if bp.shape != cp.shape:
                raise ValueError(f"shape mismatch {name}: base={tuple(bp.shape)} coder={tuple(cp.shape)}")

            # A score tensor may be absent for one seed (deepspeed occasionally yields a
            # None grad for a parameter, so accumulate skips it). Skip that tensor rather
            # than fail the whole run — it stays at the base value.
            base_paths = [d / score_path.name for d in base_score_dirs]
            coder_paths = [d / score_path.name for d in donor_score_dirs]
            if not all(p.exists() for p in base_paths + coder_paths):
                print(f"skip {name}: score missing for some seed", flush=True)
                skipped += 1
                continue
            base_score, _ = load_mean_score(base_paths, torch.device("cpu"))
            coder_score, _ = load_mean_score(coder_paths, torch.device("cpu"))

            new = transplant_param(bp.data, cp.data, base_score, coder_score, k, strategy, name=name, seed=seed)
            bp.data.copy_(new.to(bp.dtype))
            applied += 1
            if idx % 25 == 0 or idx == len(files):
                print(f"transplant {strategy}: {idx}/{len(files)} tensors (applied={applied})", flush=True)

    print(f"transplant {strategy}: applied={applied} skipped={skipped}", flush=True)
    del coder
    return base, tokenizer, applied, skipped


def run_ppl(model_dir, token, dtype, device, data_prefix, max_samples, max_seq_len, batch_size):
    print(f"\n== PPL eval ({model_dir}) ==", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, token=token, torch_dtype=dtype).to(device)
    dataset = BinDataset(data_prefix, max_seq_len)
    indices = list(range(min(max_samples, len(dataset))))
    args = argparse.Namespace(batch_size=batch_size, max_seq_len=max_seq_len, progress_every=32)
    row = evaluate_loaded_model("ppl", model, tokenizer, dataset, indices, args, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _extract_pass_at_1(metric_json):
    """Pull pass@1 out of BigCode's metric json (keyed by task name)."""
    for key, value in metric_json.items():
        if key == "config" or not isinstance(value, dict):
            continue
        for mk, mv in value.items():
            if mk.lower().replace(" ", "").startswith("pass@1"):
                return mv
    return None


def run_humaneval(model_dir, bigcode_dir, token, precision, prompt, max_len, limit, work_dir):
    print(f"\n== HumanEvalPack java ({model_dir}) ==", flush=True)
    gen_path = work_dir / "he_generations.json"
    metric_path = work_dir / "he_metrics.json"
    cmd = [
        sys.executable, str(Path(bigcode_dir) / "main.py"),
        "--model", str(model_dir),
        "--tasks", "humanevalsynthesize-java",
        "--prompt", prompt,
        "--max_length_generation", str(max_len),
        "--temperature", "0.0",
        "--do_sample", "False",
        "--n_samples", "1",
        "--batch_size", "1",
        "--precision", precision,
        "--allow_code_execution",
        "--save_generations",
        "--save_generations_path", str(gen_path),
        "--metric_output_path", str(metric_path),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ)
    if token:
        env.setdefault("HF_TOKEN", token)
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    print("bigcode:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    metric_json = json.loads(metric_path.read_text(encoding="utf-8"))
    return {"pass@1": _extract_pass_at_1(metric_json), "raw": metric_json}


def main():
    p = argparse.ArgumentParser(description="Evaluate one transplant condition (PPL + HumanEvalPack java).")
    p.add_argument("--strategy", required=True, choices=(*ORIGINALS, *STRATEGIES))
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--base-scores", nargs="+", type=Path, default=[])
    p.add_argument("--donor-scores", nargs="+", type=Path, default=[])
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("/work/transplant_model"))
    # PPL
    p.add_argument("--data-prefix", type=Path)
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--skip-ppl", action="store_true")
    # HumanEval
    p.add_argument("--bigcode-dir", type=Path)
    p.add_argument("--he-prompt", default="codellama")
    p.add_argument("--he-max-len", type=int, default=1536)
    p.add_argument("--he-limit", type=int, default=0)
    p.add_argument("--skip-humaneval", action="store_true")
    args = p.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    precision = "bf16" if dtype == torch.bfloat16 else "fp32"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    transplant_info = None
    if args.strategy == "base":
        model_dir = args.base_model
    elif args.strategy == "coder":
        model_dir = args.donor_model
    else:
        if not args.base_scores or not args.donor_scores:
            raise SystemExit("--base-scores and --donor-scores are required for transplant strategies")
        model, tokenizer, applied, skipped = build_transplanted_model(
            args.base_model, args.donor_model, args.base_scores, args.donor_scores,
            args.k, args.strategy, token, dtype, args.seed,
        )
        model_dir = args.work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        transplant_info = {"transplanted_tensors": applied, "skipped_tensors": skipped, "k": args.k, "seed": args.seed}
        model_dir = str(model_dir)

    result = {"strategy": args.strategy, "base_model": args.base_model, "donor_model": args.donor_model}
    if transplant_info:
        result["transplant"] = transplant_info

    if not args.skip_ppl:
        if not args.data_prefix:
            raise SystemExit("--data-prefix required unless --skip-ppl")
        result["ppl"] = run_ppl(model_dir, token, dtype, device, args.data_prefix,
                                args.max_samples, args.max_seq_len, args.batch_size)

    if not args.skip_humaneval:
        if not args.bigcode_dir:
            raise SystemExit("--bigcode-dir required unless --skip-humaneval")
        result["humaneval"] = run_humaneval(model_dir, args.bigcode_dir, token, precision,
                                            args.he_prompt, args.he_max_len, args.he_limit or None, args.work_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the raw generations next to metrics (on /shared) so we can inspect what the
    # model actually produced — BigCode appends the task name to the filename, so glob it.
    import glob as _glob
    import shutil as _shutil
    for g in _glob.glob(str(args.work_dir / "he_generations*")):
        try:
            _shutil.copy(g, args.output.parent / Path(g).name)
        except OSError as exc:
            print(f"could not publish generations {g}: {exc}", flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}\n{json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
