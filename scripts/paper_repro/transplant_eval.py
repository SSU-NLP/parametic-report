#!/usr/bin/env python3
"""Phase C transplant + Python-HumanEval(chat) eval, for the qwen instruct pair.

Applies a per-tensor transplant (donor -> recipient) over a given mask dir, for one strategy,
then evals the transplanted INSTRUCT model on the validated lm-eval `humaneval_instruct` + chat
path (reuses eval_humaneval_python.run_humaneval). Held-out Python, directly comparable to the
Phase B damage table (floor=base-it 0.543, ceiling=coder-it 0.628).

ONE driver covers BOTH families by swapping the mask dir:
  - Spot family   : --mask-base/--mask-coder = code-spot masks (B)
  - Bridge family : --mask-base/--mask-coder = bridge masks (A∖B)

Strategy logic is the SAME per-tensor transplant as scripts/transplant/eval_bridge_cowork.py
(v1/v2/v3/rand/perm/ndhi/ndlo/vhi/vlo/reverse), copied here to avoid the java-eval coupling.
baselines (base/coder) are NOT run here — they are the damage-table original rows.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))          # eval_humaneval_python.py lives in scripts/
from eval_humaneval_python import run_humaneval, extract_pass_at_1  # noqa: E402 (no java/torch deps)

STRATEGIES = {"v1", "v2", "v3", "rand", "perm", "ndhi", "ndlo", "vhi", "vlo", "reverse"}


def build_transplant(base_id, donor_id, mask_base, mask_coder, strategy, token, dtype):
    """Per-tensor donor->recipient transplant over mask dirs; same logic as eval_bridge_cowork.build.
    recv = the model being edited/evaluated; src = value source. reverse swaps them."""
    reverse = strategy == "reverse"
    recv_id, src_id = (donor_id, base_id) if reverse else (base_id, donor_id)
    recv = AutoModelForCausalLM.from_pretrained(recv_id, token=token, torch_dtype=dtype)
    src = AutoModelForCausalLM.from_pretrained(src_id, token=token, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_id, token=token)
    rp = dict(recv.named_parameters()); sp = dict(src.named_parameters())
    recv_mask, src_mask = (mask_coder, mask_base) if reverse else (mask_base, mask_coder)
    applied = selected = 0
    with torch.no_grad():
        for i, mf in enumerate(sorted(Path(recv_mask).glob("*.pt"))):
            name = mf.stem
            smf = Path(src_mask) / mf.name
            if name not in rp or name not in sp or not smf.exists():
                continue
            rm = torch.load(mf, map_location="cpu").bool()
            sm = torch.load(smf, map_location="cpu").bool()
            if tuple(rm.shape) != tuple(rp[name].shape):
                continue
            inter = rm & sm
            n = int(inter.sum())
            g = torch.Generator().manual_seed(20260624 + i)
            if reverse or strategy == "v2":
                sel = inter; vals = sp[name].data[sel]
            elif strategy == "v1":
                sel = sm; vals = sp[name].data[sel]
            elif strategy == "v3":
                sel = rm; vals = sp[name].data[sel]
            elif strategy == "rand":
                flat = torch.zeros(rm.numel(), dtype=torch.bool)
                if n:
                    flat[torch.randperm(rm.numel(), generator=g)[:n]] = True
                sel = flat.reshape(rm.shape); vals = sp[name].data[sel]
            elif strategy == "perm":
                sel = inter
                v = sp[name].data[sel]
                vals = v[torch.randperm(v.numel(), generator=g)] if v.numel() else v
            elif strategy in ("ndhi", "ndlo", "vhi", "vlo"):
                delta = (sp[name].data - rp[name].data).abs().reshape(-1)
                if strategy[0] == "n":
                    cand = ~inter.reshape(-1); k_ = n
                else:
                    cand = inter.reshape(-1); k_ = n // 2
                hi = strategy.endswith("hi")
                d = delta.clone()
                d[~cand] = -1.0 if hi else float("inf")
                idx = torch.topk(d, k_, largest=hi).indices if k_ > 0 else torch.empty(0, dtype=torch.long)
                flat = torch.zeros(rm.numel(), dtype=torch.bool); flat[idx] = True
                sel = flat.reshape(rm.shape); vals = sp[name].data[sel]
            else:
                raise SystemExit(f"unknown strategy {strategy}")
            cnt = int(sel.sum())
            if cnt:
                rp[name].data[sel] = vals.to(rp[name].dtype)
                selected += cnt
            applied += 1
    del src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"transplant {strategy} (reverse={reverse}): applied={applied} selected={selected}", flush=True)
    return recv, tok, applied, selected


def run_humaneval_java(model_dir, bigcode_dir, token, prompt, max_len, limit, work_dir, instruction_tokens=None):
    """In-distribution Java eval via BigCode humanevalsynthesize-java (instruct synthesize),
    matching the prior/doyun harness — to compare in-dist Java vs held-out Python on the SAME model."""
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    metric_path = work_dir / "he_metrics.json"; gen_path = work_dir / "he_generations.json"
    cmd = [
        sys.executable, str(Path(bigcode_dir) / "main.py"),
        "--model", str(model_dir), "--tasks", "humanevalsynthesize-java",
        "--prompt", prompt, "--max_length_generation", str(max_len),
        "--temperature", "0.0", "--do_sample", "False", "--n_samples", "1", "--batch_size", "1",
        "--precision", "bf16", "--allow_code_execution",
        "--save_generations", "--save_generations_path", str(gen_path),
        "--metric_output_path", str(metric_path),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if instruction_tokens:                       # chat delimiters (e.g. Qwen <|im_start|>user...)
        cmd += ["--instruction_tokens", instruction_tokens]
    env = dict(os.environ); env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if token:
        env.setdefault("HF_TOKEN", token)
    print("bigcode:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    mj = json.loads(metric_path.read_text(encoding="utf-8"))
    for k, v in mj.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                if mk.lower().replace(" ", "").startswith("pass@1"):
                    return mv
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=["spot", "bridge"])
    p.add_argument("--strategy", required=True, choices=sorted(STRATEGIES) + ["none"])
    p.add_argument("--eval", choices=["python", "java"], default="python")
    p.add_argument("--bigcode-dir", default=None, help="cloned bigcode harness (java eval)")
    p.add_argument("--he-prompt", default="instruct")
    p.add_argument("--he-max-len", type=int, default=2048)
    p.add_argument("--he-instruction-tokens", default=None,
                   help="bigcode chat delimiters 'user,end,assistant' for instruct-fair java eval")
    p.add_argument("--he-chat", choices=["none", "qwen"], default="none",
                   help="qwen: set instruction_tokens to Qwen chat delimiters (real newlines, no shell escaping)")
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--mask-base", default=None, help="recipient mask dir (code spot OR bridge); omit for strategy none")
    p.add_argument("--mask-coder", default=None, help="donor mask dir")
    p.add_argument("--gen-max-toks", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("/work/pr_tx"))
    a = p.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    a.work_dir.mkdir(parents=True, exist_ok=True)
    applied = selected = 0
    if a.strategy == "none":           # baseline: eval the model as-is (Java/Python floor or ceiling)
        model_dir = a.base_model
    else:
        recv, tok, applied, selected = build_transplant(
            a.base_model, a.donor_model, a.mask_base, a.mask_coder, a.strategy, token, torch.bfloat16)
        mdir = a.work_dir / "model"
        recv.save_pretrained(mdir); tok.save_pretrained(mdir)
        del recv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = str(mdir)

    if a.eval == "java":
        instr = a.he_instruction_tokens
        if a.he_chat == "qwen" and not instr:      # Qwen chat delimiters, real newlines
            instr = "<|im_start|>user\n,<|im_end|>\n,<|im_start|>assistant\n"
        p1 = run_humaneval_java(model_dir, a.bigcode_dir, token, a.he_prompt, a.he_max_len,
                                a.limit or None, a.work_dir / "hejava", instruction_tokens=instr)
    else:
        results = run_humaneval(model_dir, a.work_dir / "he", token, apply_chat=True,
                                limit=a.limit or None, gen_max_toks=a.gen_max_toks)
        p1 = extract_pass_at_1(results)
    out = {
        "family": a.family, "strategy": a.strategy, "eval": a.eval,
        "base": a.base_model, "donor": a.donor_model,
        "applied_tensors": applied, "selected_params": selected,
        "humaneval": {"pass@1": p1},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
