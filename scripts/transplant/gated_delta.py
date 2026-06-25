#!/usr/bin/env python3
"""Compatibility-gated static surgery — the last learning-free transplant test.

Per layer-tensor, with recipient code-grad stats G=Σg, A=Σ|g|, F=Σg² (from grad_calib.py)
and donor delta Δ=θ_D−θ_R:
  ρ = |G|/(A+ε)              delta-conditioned sign-consistency  (≡ |ΣgΔ|/Σ|gΔ|)
  gain = max(0, −G·Δ)        first-order descent contribution (>0 only)
  risk = F·Δ²                diagonal-Fisher 2nd-order risk
  T = gain/(risk+ε)·ρ        (NB: gain/risk = −G/(FΔ) → tiny |Δ| inflates T; |Δ| logged)
mask = per-tensor top-k% by score; θ' = θ_R + α·M⊙Δ. Mask modes isolate what matters:
  T            full score
  gain         gradient-alignment only (no Fisher/ρ)
  descent-rand random among descent coords (−GΔ>0)  → value of *ranking* beyond the gate
  matched-rand random among all coords             → value of the gate itself
Eval = java MultiPL-E completion pass@1. Logs predicted ΔL = −α·gain + ½α²·risk (must be <0).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import eval_humaneval_java_cowork as cw          # noqa: E402
from eval_bridge_cowork import keep_tensor, _parse_layers  # noqa: E402 (torch-free)


def gated_mask(G, A, F, delta, k, mode, eps=1e-8, seed=0, theta=None):
    """Return (bool mask over flattened tensor, stats dict). Pure torch.
    Saliency modes (signed/abs/signed-and-abs/abs-not-signed) need theta=recipient weight:
      S^signed=|θ|·|G| (≡|Σ g·θ|), S^abs=|θ|·A (≡Σ|g·θ|) — no descent gate, top-k% by score."""
    import torch
    G = G.float(); A = A.float(); F = F.float(); delta = delta.float()
    n = delta.numel()
    gain_all = (-G * delta).clamp_min(0.0)             # descent contribution, >0 only
    risk_all = F * delta * delta
    eligible = gain_all > 0
    count = max(0, min(n, int(round(k * n))))
    mask = torch.zeros(n, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    if mode in ("T", "gain"):
        if mode == "T":
            rho = G.abs() / (A + eps)
            score = gain_all / (risk_all + eps) * rho
        else:
            score = gain_all.clone()
        score = score.masked_fill(~eligible, float("-inf"))
        c = min(count, int(eligible.sum()))
        if c > 0:
            mask[torch.topk(score, c).indices] = True
    elif mode == "descent-rand":
        idx = torch.nonzero(eligible).flatten()
        c = min(count, idx.numel())
        if c > 0:
            mask[idx[torch.randperm(idx.numel(), generator=g)[:c]]] = True
    elif mode == "matched-rand":
        c = min(count, n)
        if c > 0:
            mask[torch.randperm(n, generator=g)[:c]] = True
    elif mode in ("signed", "abs", "signed-and-abs", "abs-not-signed"):
        if theta is None:
            raise SystemExit(f"mode {mode} needs theta")
        th = theta.float().reshape(-1).abs()
        s_signed = th * G.abs()                         # |θ|·|G| = |Σ g·θ|
        s_abs = th * A                                  # |θ|·A   = Σ|g·θ|
        def _topk(score):
            m = torch.zeros(n, dtype=torch.bool)
            c = min(count, n)
            if c > 0:
                m[torch.topk(score, c).indices] = True
            return m
        if mode == "signed":
            mask = _topk(s_signed)
        elif mode == "abs":
            mask = _topk(s_abs)
        elif mode == "signed-and-abs":
            mask = _topk(s_signed) & _topk(s_abs)
        else:                                           # abs-not-signed (control)
            mask = _topk(s_abs) & ~_topk(s_signed)
    else:
        raise SystemExit(f"unknown mask-mode {mode}")
    rho_full = G.abs() / (A + eps)
    sel = mask
    stats = {
        "count": int(sel.sum()), "eligible": int(eligible.sum()),
        "gain": float((-G * delta)[sel].sum()),          # signed Σ−GΔ over selected
        "risk": float(risk_all[sel].sum()),
        "abs_delta": float(delta[sel].abs().sum()),
        "rho_sum": float(rho_full[sel].sum()),
    }
    return mask, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)
    p.add_argument("--grad-dir", required=True, type=Path, help="dir with G/ A/ F/ *.pt")
    p.add_argument("--mode", default="T",
                   choices=["T", "gain", "descent-rand", "matched-rand",
                            "signed", "abs", "signed-and-abs", "abs-not-signed"])
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--modules", default="ffn")
    p.add_argument("--layers", default="2-25")
    p.add_argument("--data", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base_model, token=token)
    base = AutoModelForCausalLM.from_pretrained(a.base_model, token=token, torch_dtype=dtype).to(device).eval()
    donor = AutoModelForCausalLM.from_pretrained(a.donor_model, token=token, torch_dtype=dtype).to(device).eval()
    dp = dict(donor.named_parameters())
    layers = _parse_layers(a.layers)

    agg = {"count": 0, "eligible": 0, "gain": 0.0, "risk": 0.0, "abs_delta": 0.0, "rho_sum": 0.0, "tensors": 0}
    if a.alpha != 0.0:
        with torch.no_grad():
            for name, prm in base.named_parameters():
                if not keep_tensor(name, a.modules, layers) or name not in dp:
                    continue
                gf = a.grad_dir / "G" / f"{name}.pt"
                if not gf.exists():
                    continue
                G = torch.load(gf, map_location="cpu")
                A = torch.load(a.grad_dir / "A" / f"{name}.pt", map_location="cpu")
                F = torch.load(a.grad_dir / "F" / f"{name}.pt", map_location="cpu")
                delta = (dp[name].data.float().cpu() - prm.data.float().cpu()).reshape(-1)
                theta = prm.data.detach().float().cpu().reshape(-1)   # recipient weight for saliency modes
                mask, st = gated_mask(G.reshape(-1), A.reshape(-1), F.reshape(-1), delta,
                                      a.k, a.mode, seed=20260624 + agg["tensors"], theta=theta)
                if st["count"]:
                    sel_vals = (prm.data.reshape(-1)[mask].float()
                                + a.alpha * delta[mask].to(device)).to(prm.dtype)
                    prm.data.reshape(-1)[mask] = sel_vals
                for kk in ("count", "eligible", "gain", "risk", "abs_delta", "rho_sum"):
                    agg[kk] += st[kk]
                agg["tensors"] += 1
    pred_dL = -a.alpha * agg["gain"] + 0.5 * a.alpha ** 2 * agg["risk"]
    mean_absd = agg["abs_delta"] / max(1, agg["count"])
    mean_rho = agg["rho_sum"] / max(1, agg["count"])
    print(f"[gated] mode={a.mode} k={a.k} alpha={a.alpha} mods={a.modules} L={a.layers} "
          f"selected={agg['count']} eligible={agg['eligible']} gain={agg['gain']:.3e} "
          f"risk={agg['risk']:.3e} pred_dL={pred_dL:.3e} mean|d|={mean_absd:.3e} mean_rho={mean_rho:.3f}",
          flush=True)
    if a.mode in ("T", "gain") and a.alpha != 0.0 and pred_dL > 0:
        print(f"[gated][WARN] predicted ΔL>0 ({pred_dL:.3e}) — 1st-order says this should not help", flush=True)

    tasks = cw.load_tasks(a.data, limit=a.limit)
    gen_args = argparse.Namespace(model=a.base_model, dtype="bfloat16", device="auto",
                                  max_new_tokens=512, temperature=0.0, top_p=0.95,
                                  batch_size=a.batch_size, local_files_only=False)
    rows = cw.generate_completions(gen_args, tasks, model=base, tokenizer=tok)
    Path(a.result).mkdir(parents=True, exist_ok=True)
    cw.write_jsonl(str(Path(a.result) / "generations.jsonl"), rows)
    summary = cw.score_results(argparse.Namespace(result=a.result, timeout=10), rows)
    summary.update({"mode": a.mode, "k": a.k, "alpha": a.alpha, "modules": a.modules, "layers": a.layers,
                    "selected": agg["count"], "eligible": agg["eligible"], "pred_dL": pred_dL,
                    "mean_abs_delta": mean_absd, "mean_rho": mean_rho,
                    "gain_total": agg["gain"], "risk_total": agg["risk"]})
    # score_results already wrote summary.json with only the score fields; re-write with the
    # merged dict so selected/pred_dL/mode persist to disk (not just stdout).
    (Path(a.result) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
