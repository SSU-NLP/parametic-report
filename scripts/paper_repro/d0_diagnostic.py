#!/usr/bin/env python3
"""D0 — pre-GPU diagnostic for the Phase C bridge transplant. CPU-only, STOP-if-negative gate.

Question it answers WITHOUT spending GPU: is there a concentrated, aligned coding-drift signal in
the bridge (A∖B) worth transplanting at all? If not, we stop (clean negative).

Pair: recipient R (Qwen2.5-1.5B-Instruct), donor D (Qwen2.5-Coder-1.5B-Instruct), identical shapes.
Per target tensor t:
  theta_R, theta_D : weights (loaded from HF).
  B_R, B_D         : code-spot bool masks (top-kB by summed |grad·param|), from the volume.
  A                : top-kA by |theta|.  bridge = A & ~B.  inter = bridge_R & bridge_D.
  delta_j = theta_D - theta_R  (drift; |delta| magnitude, sign = direction).

Reports (aggregated over tensors, size-weighted):
  D0-a  drift enrichment of bridge vs a MATCHED-NULL (per-tensor, |theta_D| magnitude-binned random
        subset of the same size, drawn from non-B coords) and vs B and vs rest. + signed mean.
        --> the gate: if bridge drift is NOT enriched over matched-null, STOP.
  D0-b  representation-alignment proxy: Spearman corr(I_R, I_D) of summed importance (overall +
        within bridge). low => coordinate-wise transplant assumption weak.
  D0-c  module/layer distribution of bridge / B / inter (embed/attn/mlp/norm) to rule out a generic
        distribution shift masquerading as coding-specific.

Pure helpers (top_k_bool, matched_null_bool, enrichment, module_of, spearman) are unit-tested in
tests/test_d0_diagnostic.py (no torch model load needed).
"""
import argparse
import json
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]   # repo root (scripts/paper_repro/<this>)
import sys
sys.path.insert(0, str(REPO / "scripts"))
from arch_adapter import is_target  # noqa: E402


# ---------- pure helpers (unit-tested) ----------
def top_k_bool(score, k):
    """Bool mask of the top-k fraction by value (largest). Deterministic."""
    n = score.numel()
    cnt = int(k * n)
    out = torch.zeros(n, dtype=torch.bool)
    if cnt <= 0:
        return out
    idx = torch.topk(score.reshape(-1), min(cnt, n), largest=True, sorted=False).indices
    out[idx] = True
    return out


def matched_null_bool(weight_abs, exclude_mask, count, nbins=20, seed=0):
    """Sample `count` positions from coords NOT in exclude_mask, matched to the |theta| distribution
    of... the caller passes weight_abs already restricted in spirit; here we bin by weight_abs over
    the *candidate* (~exclude) coords and draw proportionally to a target histogram == the excluded
    set's histogram. Returns a bool mask. Deterministic via seed."""
    wa = weight_abs.reshape(-1)
    n = wa.numel()
    cand = (~exclude_mask.reshape(-1))
    out = torch.zeros(n, dtype=torch.bool)
    if count <= 0 or cand.sum() == 0:
        return out
    # bins over the full tensor's weight magnitude
    qs = torch.linspace(0, 1, nbins + 1)
    edges = torch.quantile(wa.float(), qs)
    edges[0] = -float("inf"); edges[-1] = float("inf")
    g = torch.Generator().manual_seed(seed)
    chosen = []
    for b in range(nbins):
        in_bin = (wa >= edges[b]) & (wa < edges[b + 1]) & cand
        bin_idx = torch.nonzero(in_bin, as_tuple=False).flatten()
        if bin_idx.numel() == 0:
            continue
        # draw proportional to bin's share of the candidate pool
        share = bin_idx.numel() / int(cand.sum())
        take = min(bin_idx.numel(), max(1, round(count * share)))
        perm = bin_idx[torch.randperm(bin_idx.numel(), generator=g)[:take]]
        chosen.append(perm)
    if not chosen:
        return out
    sel = torch.cat(chosen)[:count]
    out[sel] = True
    return out


def enrichment(delta_abs, mask, baseline_mask):
    """mean(|delta| over mask) / mean(|delta| over baseline_mask). >1 == enriched."""
    d = delta_abs.reshape(-1)
    m = mask.reshape(-1); b = baseline_mask.reshape(-1)
    if m.sum() == 0 or b.sum() == 0:
        return None
    mb = d[b].mean().item()
    return (d[m].mean().item() / mb) if mb > 0 else None


def module_of(name):
    if "embed" in name or "lm_head" in name:
        return "embed"
    if "norm" in name:
        return "norm"
    if any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn")):
        return "attn"
    if any(x in name for x in ("gate_proj", "up_proj", "down_proj", "mlp")):
        return "mlp"
    return "other"


def spearman(a, b):
    """Spearman rank correlation of two 1-D tensors (no scipy)."""
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    if a.numel() < 2:
        return None
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    return (ra @ rb / denom).item() if denom > 0 else None


# ---------- score loading (reuse spot pipeline idea) ----------
def mean_importance(score_root, langs, sample, name):
    """mean over languages of |grad·param| for one tensor; None if absent."""
    acc = None; cnt = 0
    for l in langs:
        p = Path(score_root) / l / f"grad-mul-param_checkpoint_{sample}" / f"{name}.pt"
        if not p.exists():
            continue
        t = torch.load(p, map_location="cpu").abs().float().reshape(-1)
        acc = t if acc is None else acc + t
        cnt += 1
    return (acc / cnt) if cnt else None


def load_weights(model_id, token):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, token=token)
    return {n: p.detach().reshape(-1) for n, p in m.named_parameters()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipient-id", required=True)
    p.add_argument("--donor-id", required=True)
    p.add_argument("--recipient-score-root", required=True)  # scores/<m>/seed_<s>
    p.add_argument("--donor-score-root", required=True)
    p.add_argument("--recipient-spot", required=True)        # masks/<m>/top<kB>/code
    p.add_argument("--donor-spot", required=True)
    p.add_argument("--langs", default="bash,csharp,cpp,go,java,javascript,julia,ruby,rust,typescript")
    p.add_argument("--sample", default="10000")
    p.add_argument("--ka", default="0.01,0.05")
    p.add_argument("--corr-sample", type=int, default=40,
                   help="D0-b importance corr is computed on ~N evenly-spaced tensors only "
                        "(the score reads off /shared are the I/O bottleneck); 0 = all. "
                        "The D0-a gate uses ALL tensors and needs no scores.")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    langs = [x for x in a.langs.split(",") if x]
    kas = [float(x) for x in a.ka.split(",")]
    print("loading weights (CPU)...", flush=True)
    wR = load_weights(a.recipient_id, token)
    wD = load_weights(a.donor_id, token)

    # tensors to analyze = those present as spot masks (the target set) on the recipient side
    names = sorted(q.stem for q in Path(a.recipient_spot).glob("*.pt"))
    per_ka = {f"{ka}": {"enrich_bridge_vs_null": [], "enrich_B_vs_null": [], "signed_bridge": [],
                        "module": {}, "weights": []} for ka in kas}
    corr_overall = []; corr_bridge = []
    # D0-b corr reads per-language scores off /shared (the I/O bottleneck) — sample only ~N tensors.
    # D0-a (the gate) below uses ALL tensors and needs no scores.
    stride = max(1, len(names) // a.corr_sample) if a.corr_sample else 1

    for i, name in enumerate(names):
        if name not in wR or name not in wD or not is_target(name):
            continue
        tR, tD = wR[name], wD[name]
        if tR.shape != tD.shape:
            continue
        delta = (tD - tR)
        delta_abs = delta.abs()
        bR = torch.load(Path(a.recipient_spot) / f"{name}.pt", map_location="cpu").bool().reshape(-1)
        dspot = Path(a.donor_spot) / f"{name}.pt"
        bD = torch.load(dspot, map_location="cpu").bool().reshape(-1) if dspot.exists() else bR
        # D0-b importance corr — only on the sampled tensors (score reads are slow)
        IR = ID = None
        if a.corr_sample == 0 or (i % stride == 0):
            IR = mean_importance(a.recipient_score_root, langs, a.sample, name)
            ID = mean_importance(a.donor_score_root, langs, a.sample, name)
            if IR is not None and ID is not None:
                c = spearman(IR, ID)
                if c is not None:
                    corr_overall.append((c, tR.numel()))
        for ka in kas:
            key = f"{ka}"
            A_R = top_k_bool(tR.abs(), ka)
            A_D = top_k_bool(tD.abs(), ka)
            bridge_R = A_R & ~bR
            bridge_D = A_D & ~bD
            inter = bridge_R & bridge_D
            null = matched_null_bool(tD.abs(), bD, int(bridge_D.sum()), seed=1234 + i)
            e_bridge = enrichment(delta_abs, bridge_D, null)
            e_B = enrichment(delta_abs, bD, null)
            if e_bridge is not None:
                per_ka[key]["enrich_bridge_vs_null"].append((e_bridge, tR.numel()))
            if e_B is not None:
                per_ka[key]["enrich_B_vs_null"].append((e_B, tR.numel()))
            sgn = delta[bridge_D].mean().item() if bridge_D.sum() else 0.0
            per_ka[key]["signed_bridge"].append((sgn, int(bridge_D.sum())))
            mod = module_of(name)
            md = per_ka[key]["module"].setdefault(mod, {"bridge": 0, "B": 0, "inter": 0})
            md["bridge"] += int(bridge_D.sum()); md["B"] += int(bD.sum()); md["inter"] += int(inter.sum())
            if IR is not None and ID is not None and bridge_D.sum() > 1:
                cb = spearman(IR[bridge_D], ID[bridge_D])
                if cb is not None:
                    corr_bridge.append((cb, int(bridge_D.sum())))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(names)} tensors", flush=True)

    def wmean(pairs):
        tot = sum(w for _, w in pairs)
        return (sum(v * w for v, w in pairs) / tot) if tot else None

    result = {
        "pair": {"recipient": a.recipient_id, "donor": a.donor_id},
        "kB_spot": str(a.recipient_spot), "ka": kas, "n_tensors": len(names),
        "D0b_corr_I_overall": wmean(corr_overall),
        "D0b_corr_I_within_bridge": wmean(corr_bridge),
        "per_ka": {},
    }
    for ka in kas:
        k = f"{ka}"
        d = per_ka[k]
        result["per_ka"][k] = {
            "D0a_enrich_bridge_vs_matched_null": wmean(d["enrich_bridge_vs_null"]),
            "D0a_enrich_B_vs_matched_null": wmean(d["enrich_B_vs_null"]),
            "D0a_signed_bridge_drift_mean": wmean(d["signed_bridge"]),
            "D0c_module_counts": d["module"],
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    # gate hint
    for ka in kas:
        e = result["per_ka"][f"{ka}"]["D0a_enrich_bridge_vs_matched_null"]
        print(f"[GATE] kA={ka}: bridge drift enrichment vs matched-null = {e} "
              f"({'PASS(>1.2?)' if (e or 0) > 1.2 else 'WEAK -> consider STOP'})", flush=True)


if __name__ == "__main__":
    main()
