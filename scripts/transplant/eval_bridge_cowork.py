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


# 위치/값 통제 strategies — 전부 v2 교집합(inter)의 per-tensor 개수 n에 매칭.
#   v2      base_bridge ∩ coder_bridge 위치에 coder 값            (기준)
#   v1/v3   coder/base bridge 단독
#   rand    같은 n개 무작위 위치, coder 값                         (위치 특이성: 양 아님)
#   perm    inter 위치, coder 값을 텐서 내 셔플                    (값-위치 정합 깸)
#   ndhi    non-bridge(¬inter) 중 |coder-base| 상위 n             (drift confound 핵심)
#   ndlo    non-bridge 중 |coder-base| 하위 n
#   vhi/vlo inter 내부를 |coder-base| 상/하위 절반(n//2)으로 분리   (drift가 효과 주도?)
#   reverse coder에 base bridge 이식(inter, base 값) → 평가 coder  (인과: 하락해야)
CONTROLS = {"v1", "v2", "v3", "rand", "perm", "ndhi", "ndlo", "vhi", "vlo", "reverse"}


def build(base_id, donor_id, bridge_base, bridge_coder, strategy, token, dtype):
    reverse = strategy == "reverse"
    # recv = 수정 대상 모델(평가됨), src = 값 출처
    recv_id, src_id = (donor_id, base_id) if reverse else (base_id, donor_id)
    recv = AutoModelForCausalLM.from_pretrained(recv_id, token=token, torch_dtype=dtype)
    src = AutoModelForCausalLM.from_pretrained(src_id, token=token, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_id, token=token)
    rp = dict(recv.named_parameters()); sp = dict(src.named_parameters())
    recv_bridge, src_bridge = (bridge_coder, bridge_base) if reverse else (bridge_base, bridge_coder)
    applied = selected = 0
    with torch.no_grad():
        for i, mf in enumerate(sorted(recv_bridge.glob("*.pt"))):
            name = mf.stem
            smf = src_bridge / mf.name
            if name not in rp or name not in sp or not smf.exists():
                continue
            rm = torch.load(mf, map_location="cpu").bool()       # recv 자신의 bridge
            sm = torch.load(smf, map_location="cpu").bool()      # src 자신의 bridge
            if tuple(rm.shape) != tuple(rp[name].shape):
                continue
            inter = rm & sm                                       # v2 교집합 = 매칭 기준
            n = int(inter.sum())
            g = torch.Generator().manual_seed(20260622 + i)
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
                if strategy[0] == "n":               # non-bridge 후보, 전체 n
                    cand = ~inter.reshape(-1); k_ = n
                else:                                 # bridge 내부, 절반
                    cand = inter.reshape(-1); k_ = n // 2
                hi = strategy.endswith("hi")
                d = delta.clone()
                d[~cand] = -1.0 if hi else float("inf")
                idx = torch.topk(d, k_, largest=hi).indices if k_ > 0 else torch.empty(0, dtype=torch.long)
                flat = torch.zeros(rm.numel(), dtype=torch.bool)
                flat[idx] = True
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
    print(f"bridge {strategy} (reverse={reverse}): applied={applied} selected={selected}", flush=True)
    return recv, tok


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
    p.add_argument("--batch-size", type=int, default=32)   # batched greedy; A100 80GB는 32~48 여유
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
                                  batch_size=a.batch_size, local_files_only=False)
    rows = cw.generate_completions(gen_args, tasks)
    Path(a.result).mkdir(parents=True, exist_ok=True)
    cw.write_jsonl(str(Path(a.result) / "generations.jsonl"), rows)
    summary = cw.score_results(argparse.Namespace(result=a.result, timeout=10), rows)
    summary["strategy"] = a.strategy
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
