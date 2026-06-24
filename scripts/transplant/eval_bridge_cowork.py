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
import re
import sys
from pathlib import Path

# torch/transformers are imported lazily inside build()/main() so the pure helpers below
# (keep_tensor/_parse_layers) stay importable for unit tests on a host without torch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_humaneval_java_cowork as cw  # noqa: E402

ORIGINALS = ("base", "coder")

# ── module/layer restriction (FFN-only / attn-only / per-layer) — pure, torch-free ──
_LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)$")


def _parse_param(name):
    """`model.layers.5.mlp.down_proj.weight` -> (5, 'mlp.down_proj.weight'); None for embed/lm_head/model.norm."""
    m = _LAYER_RE.search(name)
    return (int(m.group(1)), m.group(2)) if m else None


def _parse_layers(spec):
    """'all'->None; '12'->{12}; '0-5'->{0..5}; '0-3,12'->{0,1,2,3,12}."""
    if spec is None or spec == "all":
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def keep_tensor(name, modules="all", layers=None):
    """True iff this named_parameter is in-scope for the module/layer restriction.
    modules: all|ffn|attn|attn-vo|attn-qk|attn-qkv|norm. layers: None(all) or set of int indices."""
    pp = _parse_param(name)
    if layers is not None:                       # layer filter: only numbered blocks survive
        if pp is None or pp[0] not in layers:
            return False
    if modules == "all":
        return True
    if pp is None:                               # embed/lm_head/model.norm never match a module class
        return False
    suffix = pp[1]
    if modules == "ffn":
        return suffix.startswith("mlp.")         # Qwen2.5 mlp.{gate,up,down}_proj.weight (no bias)
    if modules == "norm":
        return "input_layernorm" in suffix or "post_attention_layernorm" in suffix
    is_q, is_k = "q_proj" in suffix, "k_proj" in suffix
    is_v, is_o = "v_proj" in suffix, "o_proj" in suffix
    if modules == "attn":
        return is_q or is_k or is_v or is_o
    if modules == "attn-vo":
        return is_v or is_o
    if modules == "attn-qk":
        return is_q or is_k
    if modules == "attn-qkv":
        return is_q or is_k or is_v
    raise SystemExit(f"unknown --modules {modules}")


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


def build(base_id, donor_id, bridge_base, bridge_coder, strategy, token, dtype, alpha=1.0,
          modules="all", layers=None):
    """Transplant donor->recipient over a mask dir. alpha<1 turns the masked OVERWRITE into a
    scaled DELTA-ADD:  θ' = θ_recv + α·(θ_src − θ_recv)  (α=1 == the original replace).
    strategy 'interp' ignores masks and blends the WHOLE model: θ' = (1−α)·base + α·coder
    (positive control: is the base→coder linear weight path even coherent?).
    modules/layers restrict which tensors are touched (FFN-only / attn-only / per-layer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reverse = strategy == "reverse"
    # recv = 수정 대상 모델(평가됨), src = 값 출처
    recv_id, src_id = (donor_id, base_id) if reverse else (base_id, donor_id)
    recv = AutoModelForCausalLM.from_pretrained(recv_id, token=token, torch_dtype=dtype)
    src = AutoModelForCausalLM.from_pretrained(src_id, token=token, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_id, token=token)
    rp = dict(recv.named_parameters()); sp = dict(src.named_parameters())
    if strategy == "interp":                       # global interpolation, no mask
        with torch.no_grad():
            for name, prm in rp.items():
                if not keep_tensor(name, modules, layers):
                    continue
                if name in sp and tuple(sp[name].shape) == tuple(prm.shape):
                    prm.data.copy_((prm.data + alpha * (sp[name].data.to(prm.dtype) - prm.data)).to(prm.dtype))
        del src
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"interp alpha={alpha}: blended {len(rp)} tensors", flush=True)
        return recv, tok
    recv_bridge, src_bridge = (bridge_coder, bridge_base) if reverse else (bridge_base, bridge_coder)
    applied = selected = 0
    with torch.no_grad():
        for i, mf in enumerate(sorted(recv_bridge.glob("*.pt"))):
            name = mf.stem
            if not keep_tensor(name, modules, layers):
                continue
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
                if alpha == 1.0:
                    rp[name].data[sel] = vals.to(rp[name].dtype)
                else:                                  # delta-add: θ_recv + α·(θ_src − θ_recv)
                    cur = rp[name].data[sel]
                    rp[name].data[sel] = (cur + alpha * (vals.to(cur.dtype) - cur)).to(rp[name].dtype)
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
    p.add_argument("--alpha", type=float, default=1.0, help="delta-add scale (1.0=replace); also global interp scale")
    p.add_argument("--modules", default="all",
                   choices=["all", "ffn", "attn", "attn-vo", "attn-qk", "attn-qkv", "norm"],
                   help="restrict transplant/interp to a module group")
    p.add_argument("--layers", default="all", help="restrict to layer indices: all | 12 | 0-5 | 0-3,12")
    a = p.parse_args()

    import torch
    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if a.strategy == "base":
        model_dir = a.base_model
    elif a.strategy == "coder":
        model_dir = a.donor_model
    else:
        if a.strategy != "interp" and (not a.bridge_base or not a.bridge_coder):
            raise SystemExit("--bridge-base and --bridge-coder required for transplant strategies")
        model, tok = build(a.base_model, a.donor_model, a.bridge_base, a.bridge_coder, a.strategy,
                           token, dtype, a.alpha, a.modules, _parse_layers(a.layers))
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
    summary["alpha"] = a.alpha
    summary["modules"] = a.modules
    summary["layers"] = a.layers
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
