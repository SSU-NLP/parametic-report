#!/usr/bin/env python3
"""Residual-gated donor-MLP injection (output-space mechanism test, no training).

For layers l in a chosen set (default middle 2-25, boundary excluded), mix the recipient
MLP output with the donor MLP output computed on the SAME recipient input h_l:

    y'_l = y^R_l + β·(y^D_l(h_l) − y^R_l)          (β=0 identity, β=1 full donor output)

Weights are NOT merged — both models are loaded and a forward hook on the recipient's MLP
replaces its output with the blend (donor MLP run inside the hook). This isolates whether
the donor FFN *output direction* helps at all and how depth-wise accumulation scales with β,
WITHOUT the nonlinear instability of weight interpolation. Eval = java MultiPL-E completion
(eval_humaneval_java_cowork) → per-problem pass@1. base=0.272 / coder=0.386.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import eval_humaneval_java_cowork as cw          # noqa: E402
from eval_bridge_cowork import _parse_layers      # noqa: E402 (torch-free helper)


def blend(out, yd, beta):
    """Linear output-space mix: β=0 → out (recipient), β=1 → yd (donor)."""
    return out + beta * (yd - out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)         # recipient
    p.add_argument("--donor-model", required=True)         # donor
    p.add_argument("--data", required=True)                # multipl-java parquet
    p.add_argument("--result", required=True)
    p.add_argument("--layers", default="2-25")             # boundary-excluded middle by default
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
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

    layers = _parse_layers(a.layers)
    L = len(base.model.layers)
    target = set(range(L)) if layers is None else {i for i in layers if 0 <= i < L}
    handles = []
    if a.beta != 0.0:                                      # β=0 → identity, no hooks (sanity baseline)
        for i in sorted(target):
            dmlp = donor.model.layers[i].mlp
            def mk(dmlp):
                def hook(mod, inp, out):
                    with torch.no_grad():
                        yd = dmlp(inp[0])
                    return blend(out, yd, a.beta)
                return hook
            handles.append(base.model.layers[i].mlp.register_forward_hook(mk(dmlp)))
    print(f"residual-gate: beta={a.beta} layers={sorted(target)} hooks={len(handles)}", flush=True)

    tasks = cw.load_tasks(a.data, limit=a.limit)
    gen_args = argparse.Namespace(model=a.base_model, dtype="bfloat16", device="auto",
                                  max_new_tokens=a.max_new_tokens, temperature=0.0, top_p=0.95,
                                  batch_size=a.batch_size, local_files_only=False)
    rows = cw.generate_completions(gen_args, tasks, model=base, tokenizer=tok)
    for h in handles:
        h.remove()
    Path(a.result).mkdir(parents=True, exist_ok=True)
    cw.write_jsonl(str(Path(a.result) / "generations.jsonl"), rows)
    summary = cw.score_results(argparse.Namespace(result=a.result, timeout=10), rows)
    summary["beta"] = a.beta
    summary["layers"] = a.layers
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
