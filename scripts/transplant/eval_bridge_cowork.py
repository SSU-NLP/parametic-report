#!/usr/bin/env python3
"""Bridge transplant + colleague's HumanEval-Java(completion) eval.

Bridge Theory: the model already holds the java ability (its core spot B = |grad·param|
top-k), but base lacks the *bridge* (A∖B = |weight| top-k minus core) that wires it up.
Transplanting coder's bridge into base should activate base's latent ability.

This loads base+coder, reads the paper-spot bridge masks (A∖B) for each, and per strategy:
  v2     base_bridge ∩ coder_bridge  위치에 coder 값  ← 동료/Bridge 핵심
  v1     coder_bridge                위치에 coder 값
  v3     base_bridge                 위치에 coder 값
  base/coder = 원본 (floor/ceiling)
then scores with the colleague's completion harness (eval_humaneval_java_cowork).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_humaneval_java_cowork as cw  # noqa: E402

ORIGINALS = ("base", "coder")


def build(base_id, donor_id, bridge_base, bridge_coder, strategy, token, dtype):
    base = AutoModelForCausalLM.from_pretrained(base_id, token=token, torch_dtype=dtype)
    coder = AutoModelForCausalLM.from_pretrained(donor_id, token=token, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_id, token=token)
    bp = dict(base.named_parameters()); cp = dict(coder.named_parameters())
    applied = selected = 0
    with torch.no_grad():
        for mf in sorted(bridge_base.glob("*.pt")):
            name = mf.stem
            cmf = bridge_coder / mf.name
            if name not in bp or name not in cp or not cmf.exists():
                continue
            bm = torch.load(mf, map_location="cpu").bool()
            cm = torch.load(cmf, map_location="cpu").bool()
            if strategy == "v2":
                sel = bm & cm           # 두 bridge의 교집합
            elif strategy == "v1":
                sel = cm                # coder bridge 위치
            else:                       # v3
                sel = bm                # base bridge 위치
            if tuple(sel.shape) != tuple(bp[name].shape):
                continue
            cnt = int(sel.sum())
            if cnt:
                bp[name].data[sel] = cp[name].data[sel].to(bp[name].dtype)
                selected += cnt
            applied += 1
    del coder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"bridge {strategy}: applied={applied} selected={selected}", flush=True)
    return base, tok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--bridge-base", type=Path)
    p.add_argument("--bridge-coder", type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/bridgemodel"))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if a.strategy == "base":
        model_dir = a.base_model
    elif a.strategy == "coder":
        model_dir = a.donor_model
    else:
        if not a.bridge_base or not a.bridge_coder:
            raise SystemExit("--bridge-base and --bridge-coder required for transplant strategies")
        model, tok = build(a.base_model, a.donor_model, a.bridge_base, a.bridge_coder, a.strategy, token, dtype)
        a.work_dir.mkdir(parents=True, exist_ok=True)
        md = a.work_dir / "model"; md.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(md); tok.save_pretrained(md)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = str(md)

    tasks = cw.load_tasks(a.data, limit=a.limit)
    gen_args = argparse.Namespace(model=model_dir, dtype="bfloat16", device="auto",
                                  max_new_tokens=a.max_new_tokens, temperature=0.0, top_p=0.95,
                                  local_files_only=False)
    rows = cw.generate_completions(gen_args, tasks)
    Path(a.result).mkdir(parents=True, exist_ok=True)
    cw.write_jsonl(str(Path(a.result) / "generations.jsonl"), rows)
    summary = cw.score_results(argparse.Namespace(result=a.result, timeout=10), rows)
    summary["strategy"] = a.strategy
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
