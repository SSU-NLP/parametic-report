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
        with idx_path.open("rb") as handle:
            self.starts = np.frombuffer(handle.read(self.total_sample * 8), dtype=np.uint64).copy()
            self.lengths = np.frombuffer(handle.read(self.total_sample * 2), dtype=np.uint16).copy()
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


def parse_mask(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Mask entries must use label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Mask entries must use label=path")
    return label, Path(path)


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


def apply_mask(model, mask_dir, device):
    mask_files = sorted(mask_dir.glob("*.pt"))
    if not mask_files:
        raise SystemExit(f"No mask files found in {mask_dir}")
    params = dict(model.named_parameters())
    applied = 0
    selected = 0
    with torch.no_grad():
        for idx, mask_path in enumerate(mask_files, 1):
            name = mask_path.name[:-3] if mask_path.name.endswith(".pt") else mask_path.stem
            param = params.get(name)
            if param is None:
                print(f"mask skipped, no matching parameter: {name}", flush=True)
                continue
            mask = torch.load(mask_path, map_location="cpu").bool()
            if tuple(mask.shape) != tuple(param.shape):
                raise ValueError(f"Shape mismatch for {name}: mask={tuple(mask.shape)} param={tuple(param.shape)}")
            count = int(mask.sum().item())
            if count:
                param.data[mask.to(device)] = 0
            applied += 1
            selected += count
            del mask
            if idx % 25 == 0 or idx == len(mask_files):
                print(f"applied mask {idx}/{len(mask_files)} selected={selected}", flush=True)
    return {"mask_tensors": applied, "zeroed_params": selected}


def evaluate_loaded_model(label, model, tokenizer, dataset, indices, args, device):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_id, args.max_seq_len),
    )
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
    return {"model": label, "loss": loss, "ppl": ppl, "tokens": token_counts}


def evaluate_condition(label, model_path, mask_dir, dataset, indices, args, device, dtype, token):
    print(f"\n== evaluating {label} ==", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, torch_dtype=dtype)
    model.to(device)
    mask_info = None
    if mask_dir is not None:
        mask_info = apply_mask(model, mask_dir, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row = evaluate_loaded_model(label, model, tokenizer, dataset, indices, args, device)
    if mask_info:
        row.update(mask_info)
    print(json.dumps(row), flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate causal LM PPL after applying zero-out masks in memory.")
    parser.add_argument("--data-prefix", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=str)
    parser.add_argument("--mask", action="append", type=parse_mask, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--skip-original", action="store_true")
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
    if not args.skip_original:
        rows.append(evaluate_condition("original", args.base_model, None, dataset, indices, args, device, dtype, token))
    for label, mask_dir in args.mask:
        rows.append(evaluate_condition(label, args.base_model, mask_dir, dataset, indices, args, device, dtype, token))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
