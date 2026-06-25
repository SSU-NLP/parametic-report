#!/usr/bin/env python3
"""Recipient code-gradient statistics for compatibility-gated static surgery.

Per-sample (batch=1) backward of the recipient on java code tokens, accumulating per
layer-parameter (model.layers.* only):
  G_j = Σ_n g_{n,j}      (signed sum  → first-order direction)
  S_j = Σ_n |g_{n,j}|    (abs sum     → sign-consistency denominator ρ_j=|G_j|/S_j)
  F_j = Σ_n g_{n,j}^2    (diagonal Fisher → curvature/2nd-order risk)
These feed T_j = max(0,-G_jΔ_j)/(F_jΔ_j^2+ε)·ρ_j in gated_delta.py. No optimizer / no
parameter update — this is "static" (we compute gradients but never train weights).

BinDataset/collate copied from scripts/evaluate_ppl.py (avoid that module's dotenv import).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


class BinDataset(Dataset):
    def __init__(self, prefix, seq_length):
        prefix = Path(prefix)
        self.total = prefix.with_suffix(".idx").stat().st_size // 10
        with prefix.with_suffix(".idx").open("rb") as f:
            self.starts = np.frombuffer(f.read(self.total * 8), dtype=np.uint64).copy()
            self.lengths = np.frombuffer(f.read(self.total * 2), dtype=np.uint16).copy()
        self.bin = np.memmap(prefix.with_suffix(".bin"), dtype=np.uint32, mode="r")
        self.seq_length = seq_length

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        s = int(self.starts[idx]); n = min(int(self.lengths[idx]), self.seq_length)
        return torch.as_tensor(self.bin[s:s + n].tolist(), dtype=torch.long)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data-prefix", required=True)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()

    token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model, token=token)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # grads need fp32 for stable accumulation; layer params only require grad
    model = AutoModelForCausalLM.from_pretrained(a.model, token=token, torch_dtype=torch.float32).to(device)
    model.eval()                                   # no dropout; we still backprop
    for name, prm in model.named_parameters():
        prm.requires_grad_("layers." in name)

    ds = BinDataset(a.data_prefix, a.seq_len)
    n = min(a.max_samples, len(ds))
    G = {}; S = {}; F = {}
    for name, prm in model.named_parameters():
        if "layers." in name:
            G[name] = torch.zeros_like(prm, dtype=torch.float32, device="cpu")
            S[name] = torch.zeros_like(prm, dtype=torch.float32, device="cpu")
            F[name] = torch.zeros_like(prm, dtype=torch.float32, device="cpu")

    for i in range(n):
        ids = ds[i].unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if prm.grad is None or "layers." not in name:
                    continue
                g = prm.grad.detach().float().cpu()
                G[name] += g; S[name] += g.abs(); F[name] += g * g
        if (i + 1) % 32 == 0:
            print(f"[grad] {i+1}/{n}", flush=True)
    model.zero_grad(set_to_none=True)

    for tag, D in (("G", G), ("A", S), ("F", F)):   # A = Σ|g| (sign-consistency denom)
        d = a.output_dir / tag; d.mkdir(parents=True, exist_ok=True)
        for name, t in D.items():
            torch.save(t.bfloat16(), d / f"{name}.pt")
    print(f"[grad] saved G/S/F for {len(G)} tensors over {n} samples -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
