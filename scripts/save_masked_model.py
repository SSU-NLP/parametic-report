#!/usr/bin/env python3
import argparse
import gc
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_name(name):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def apply_mask(model, mask_dir):
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
                param.data[mask.to(param.device)] = 0
            applied += 1
            selected += count
            del mask
            if idx % 25 == 0 or idx == len(mask_files):
                print(f"applied mask {idx}/{len(mask_files)} selected={selected}", flush=True)
            gc.collect()
    return applied, selected


def main():
    parser = argparse.ArgumentParser(description="Save a Hugging Face model after zeroing parameters selected by a mask directory.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    dtype = dtype_from_name(args.dtype)
    print(f"loading {args.base_model} dtype={args.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=token, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    applied, selected = apply_mask(model, args.mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving model to {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.base_model,
        "mask": str(args.mask),
        "dtype": args.dtype,
        "mask_tensors": applied,
        "zeroed_params": selected,
    }
    (args.output_dir / "mask_damage_meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(meta, flush=True)


if __name__ == "__main__":
    main()
