#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100


class BinDataset(Dataset):
    def __init__(self, prefix, seq_length):
        prefix = Path(prefix)
        idx_path = prefix.with_suffix(".idx")
        bin_path = prefix.with_suffix(".bin")
        file_size = idx_path.stat().st_size
        if file_size % 10 != 0:
            raise ValueError(f"Bad idx size: {idx_path}")
        self.total_sample = file_size // 10
        with idx_path.open("rb") as f:
            self.starts = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).copy()
            self.lengths = np.frombuffer(f.read(self.total_sample * 2), dtype=np.uint16).copy()
        self.bin = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_length = seq_length

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        length = min(int(self.lengths[idx]), self.seq_length)
        return torch.as_tensor(self.bin[start : start + length].tolist(), dtype=torch.long)


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model entries must use label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Model entries must use label=path")
    return label, path


def collate(batch, pad_id, max_seq_len):
    input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    if input_ids.size(1) < max_seq_len:
        pad = torch.full((input_ids.size(0), max_seq_len - input_ids.size(1)), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    labels = input_ids.clone()
    attention_mask = input_ids.ne(pad_id)
    first_false = (~attention_mask).cumsum(dim=1) == 1
    attention_mask[first_false] = True
    labels[~attention_mask] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def evaluate_model(label, model_path, dataset, indices, args, device, dtype, token):
    print(f"\n== evaluating {label} ==", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.eos_token_id, args.max_seq_len),
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, torch_dtype=dtype)
    model.to(device)
    model.eval()

    losses = []
    token_counts = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            losses.append(float(outputs.loss.detach().cpu()))
            token_counts += int((batch["labels"] != IGNORE_INDEX).sum().item())
            if args.progress_every and step % args.progress_every == 0:
                print(f"{label}: {step}/{len(loader)}", flush=True)

    loss = sum(losses) / len(losses)
    ppl = math.exp(loss) if loss < 50 else float("inf")
    row = {"model": label, "loss": loss, "ppl": ppl, "tokens": token_counts}
    print(json.dumps(row), flush=True)

    del model, tokenizer, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate causal LM loss/perplexity on preprocessed .bin/.idx data.")
    parser.add_argument("--data-prefix", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=32)
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dataset = BinDataset(args.data_prefix, args.max_seq_len)
    indices = list(range(min(args.max_samples, len(dataset))))

    print(f"eval_samples={len(indices)} total_samples={len(dataset)} max_seq_len={args.max_seq_len}", flush=True)
    print(f"device={device} dtype={dtype}", flush=True)

    rows = []
    for label, model_path in args.model:
        rows.append(evaluate_model(label, model_path, dataset, indices, args, device, dtype, token))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
