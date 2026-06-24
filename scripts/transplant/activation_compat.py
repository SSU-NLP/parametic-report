#!/usr/bin/env python3
"""Transplant-compatibility diagnostic (forward-only, no generation, no full eval).

Step 1 — activation compatibility map: hook each recipient (base) block's MLP to capture its
  input residual h_l and output y^R_l on calibration code tokens, then push h_l through the
  DONOR (coder) MLP_l to get y^hyb_l and compare per (non-pad) token:
    norm_ratio = ‖y^hyb‖/‖y^R‖,  cosine(y^hyb,y^R),  rel_delta = ‖y^hyb−y^R‖/‖h‖.
  → which layers' donor MLP is geometrically compatible with recipient activations.

Step 2 — single-module NLL micro-screen: teacher-forced NLL of base, of coder, and of base
  with ONE layer's MLP swapped to the donor's (in place, restored after). Δnll_l vs base.
  → does swapping a single module wreck the LM loss, or is some layer tolerable.

Both decide the next route (gated sparse delta vs adapter vs LoRA distillation) WITHOUT
spending on HumanEval. BinDataset/collate copied from scripts/evaluate_ppl.py (avoids that
module's top-level python-dotenv import in the runner image).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


class BinDataset(Dataset):                         # copied from scripts/evaluate_ppl.py:19
    def __init__(self, prefix, seq_length):
        prefix = Path(prefix)
        idx_path = prefix.with_suffix(".idx"); bin_path = prefix.with_suffix(".bin")
        self.total_sample = idx_path.stat().st_size // 10
        with idx_path.open("rb") as f:
            self.starts = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).copy()
            self.lengths = np.frombuffer(f.read(self.total_sample * 2), dtype=np.uint16).copy()
        self.bin = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_length = seq_length

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        start = int(self.starts[idx]); length = min(int(self.lengths[idx]), self.seq_length)
        return torch.as_tensor(self.bin[start:start + length].tolist(), dtype=torch.long)


def collate(batch, pad_id):                        # adapted from scripts/evaluate_ppl.py:52
    input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    labels = input_ids.clone()
    attention_mask = input_ids.ne(pad_id)
    first_false = (~attention_mask).cumsum(dim=1) == 1
    attention_mask[first_false] = True
    labels[~attention_mask] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def nll(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch, use_cache=False).loss.detach().cpu()))
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)        # recipient
    p.add_argument("--donor-model", required=True)        # donor
    p.add_argument("--data-prefix", required=True, help="tokenized java .bin/.idx prefix (no ext)")
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    import os
    token = os.getenv("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=token)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(a.base_model, token=token, torch_dtype=dtype).to(device).eval()
    donor = AutoModelForCausalLM.from_pretrained(a.donor_model, token=token, torch_dtype=dtype).to(device).eval()

    ds = BinDataset(a.data_prefix, a.seq_len)
    idx = list(range(min(a.max_samples, len(ds))))
    loader = DataLoader(Subset(ds, idx), batch_size=a.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id))

    blayers = base.model.layers
    dlayers = donor.model.layers
    L = len(blayers)
    assert L == len(dlayers), f"layer count mismatch {L} vs {len(dlayers)}"

    # ── Step 1: activation compatibility map ──
    cap = {}                                          # layer -> {'in':h, 'out':y}
    handles = []
    for i, layer in enumerate(blayers):
        def mk(i):
            def hook(mod, inp, out):
                cap[i] = (inp[0].detach(), out.detach())
            return hook
        handles.append(layer.mlp.register_forward_hook(mk(i)))

    acc = {i: {"nr": 0.0, "cos": 0.0, "rd": 0.0, "n": 0} for i in range(L)}
    selfcheck = None
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            cap.clear()
            base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            m = batch["attention_mask"].bool().reshape(-1)             # [B*T]
            for i in range(L):
                h, yR = cap[i]
                yhyb = dlayers[i].mlp(h)                                # donor MLP on recipient activation
                hf = h.reshape(-1, h.shape[-1])[m].float()
                yRf = yR.reshape(-1, yR.shape[-1])[m].float()
                yhf = yhyb.reshape(-1, yhyb.shape[-1])[m].float()
                nR = yRf.norm(dim=-1).clamp_min(1e-6)
                acc[i]["nr"] += float((yhf.norm(dim=-1) / nR).sum())
                acc[i]["cos"] += float(F.cosine_similarity(yhf, yRf, dim=-1).sum())
                acc[i]["rd"] += float(((yhf - yRf).norm(dim=-1) / hf.norm(dim=-1).clamp_min(1e-6)).sum())
                acc[i]["n"] += int(m.sum())
                if bi == 0 and i == 0:                                  # self-check: base MLP recomputes y^R
                    ys = blayers[0].mlp(h).reshape(-1, h.shape[-1])[m].float()
                    selfcheck = float(F.cosine_similarity(ys, yRf, dim=-1).mean())
    for hd in handles:
        hd.remove()
    assert selfcheck is not None and selfcheck > 0.999, f"hook self-check failed: cos={selfcheck}"
    print(f"[selfcheck] base-MLP recompute cosine={selfcheck:.5f} (expect ~1.0)", flush=True)

    # ── Step 2: single-module NLL micro-screen ──
    nll_base = nll(base, loader, device)
    nll_coder = nll(donor, loader, device)
    print(f"[nll] base={nll_base:.4f} coder={nll_coder:.4f}", flush=True)
    dnll = {}
    for i in range(L):
        backup = {k: v.detach().clone() for k, v in blayers[i].mlp.state_dict().items()}
        blayers[i].mlp.load_state_dict(dlayers[i].mlp.state_dict())
        nll_h = nll(base, loader, device)
        blayers[i].mlp.load_state_dict(backup)                          # restore
        dnll[i] = nll_h - nll_base
        print(f"[layer {i:2d}] nll_hybrid={nll_h:.4f} dnll={dnll[i]:+.4f}", flush=True)

    # ── write CSV ──
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["meta", f"nll_base={nll_base:.4f}", f"nll_coder={nll_coder:.4f}",
                    f"layers={L}", f"samples={len(idx)}", f"selfcheck_cos={selfcheck:.5f}"])
        w.writerow(["layer", "norm_ratio", "cosine", "rel_delta", "nll_hybrid", "dnll_vs_base"])
        for i in range(L):
            n = max(1, acc[i]["n"])
            w.writerow([i, round(acc[i]["nr"] / n, 4), round(acc[i]["cos"] / n, 4),
                        round(acc[i]["rd"] / n, 4), round(nll_base + dnll[i], 4), round(dnll[i], 4)])
    print(f"wrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
