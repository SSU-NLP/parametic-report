#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path


COPY_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "LICENSE",
    "README.md",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find overlapping parameter masks from two coding-spot directories and "
            "copy donor model values into the base model only at those positions."
        )
    )
    parser.add_argument("--base-model", required=True, help="Weak/base HF model directory")
    parser.add_argument("--donor-model", required=True, help="Strong/donor HF model directory")
    parser.add_argument("--base-mask-dir", required=True, help="Base model boolean .pt spot mask directory")
    parser.add_argument("--donor-mask-dir", required=True, help="Donor model boolean .pt spot mask directory")
    parser.add_argument("--intersection-dir", required=True, help="Where to save AND/intersection masks")
    parser.add_argument("--output", required=True, help="Output HF model directory")
    return parser.parse_args()


def copy_model_files(base_model: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = base_model / name
        if src.exists():
            shutil.copy2(src, output / name)


def load_safetensors_metadata(path: Path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata()


def create_intersection_masks(base_mask_dir: Path, donor_mask_dir: Path, intersection_dir: Path):
    import torch
    from tqdm import tqdm

    base_mask_paths = sorted(base_mask_dir.glob("*.pt"))
    if not base_mask_paths:
        raise FileNotFoundError(f"No .pt mask files found in {base_mask_dir}")

    intersection_dir.mkdir(parents=True, exist_ok=True)

    total_parameters = 0
    base_mask_parameters = 0
    donor_mask_parameters = 0
    intersection_parameters = 0
    written_tensors = 0

    for base_mask_path in tqdm(base_mask_paths, desc="intersecting", file=sys.stdout):
        donor_mask_path = donor_mask_dir / base_mask_path.name
        if not donor_mask_path.exists():
            raise FileNotFoundError(f"Matching donor mask not found: {donor_mask_path}")

        base_mask = torch.load(base_mask_path, map_location="cpu").bool()
        donor_mask = torch.load(donor_mask_path, map_location="cpu").bool()
        if tuple(base_mask.shape) != tuple(donor_mask.shape):
            raise ValueError(
                f"Mask shape mismatch for {base_mask_path.name}: "
                f"{tuple(base_mask.shape)} vs {tuple(donor_mask.shape)}"
            )

        intersection_mask = torch.logical_and(base_mask, donor_mask)
        torch.save(intersection_mask, intersection_dir / base_mask_path.name)

        total_parameters += base_mask.numel()
        base_mask_parameters += int(base_mask.sum().item())
        donor_mask_parameters += int(donor_mask.sum().item())
        intersection_parameters += int(intersection_mask.sum().item())
        written_tensors += 1

    summary = {
        "base_mask_dir": str(base_mask_dir),
        "donor_mask_dir": str(donor_mask_dir),
        "intersection_dir": str(intersection_dir),
        "tensors": written_tensors,
        "total_parameters": total_parameters,
        "base_mask_parameters": base_mask_parameters,
        "donor_mask_parameters": donor_mask_parameters,
        "intersection_parameters": intersection_parameters,
        "intersection_ratio_total": (
            intersection_parameters / total_parameters if total_parameters else 0.0
        ),
        "intersection_ratio_base_mask": (
            intersection_parameters / base_mask_parameters if base_mask_parameters else 0.0
        ),
        "intersection_ratio_donor_mask": (
            intersection_parameters / donor_mask_parameters if donor_mask_parameters else 0.0
        ),
    }

    summary_path = intersection_dir.parent / f"{intersection_dir.name}.summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, summary_path


def transplant_intersection(base_model: Path, donor_model: Path, intersection_dir: Path, output: Path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tqdm import tqdm

    base_weights = base_model / "model.safetensors"
    donor_weights = donor_model / "model.safetensors"
    output_weights = output / "model.safetensors"

    if not base_weights.exists():
        raise FileNotFoundError(base_weights)
    if not donor_weights.exists():
        raise FileNotFoundError(donor_weights)
    if output_weights.exists():
        raise FileExistsError(f"{output_weights} already exists. Move it away before rerunning.")

    copy_model_files(base_model, output)

    metadata = load_safetensors_metadata(base_weights) or {"format": "pt"}
    transplanted = {}
    masked_tensors = 0
    selected_parameters = 0

    with safe_open(base_weights, framework="pt", device="cpu") as base_handle:
        base_keys = list(base_handle.keys())
        with safe_open(donor_weights, framework="pt", device="cpu") as donor_handle:
            donor_keys = set(donor_handle.keys())

            for name in tqdm(base_keys, desc="transplanting", file=sys.stdout):
                tensor = base_handle.get_tensor(name)
                mask_path = intersection_dir / f"{name}.pt"

                if mask_path.exists():
                    if name not in donor_keys:
                        raise KeyError(f"{name} not found in donor model")

                    donor_tensor = donor_handle.get_tensor(name)
                    if tensor.shape != donor_tensor.shape:
                        raise ValueError(
                            f"Tensor shape mismatch for {name}: "
                            f"{tuple(tensor.shape)} vs {tuple(donor_tensor.shape)}"
                        )

                    mask = torch.load(mask_path, map_location="cpu").bool()
                    if tuple(mask.shape) != tuple(tensor.shape):
                        raise ValueError(
                            f"Mask shape mismatch for {name}: "
                            f"{tuple(mask.shape)} vs {tuple(tensor.shape)}"
                        )

                    count = int(mask.sum().item())
                    if count:
                        tensor = tensor.clone()
                        tensor[mask] = donor_tensor[mask].to(dtype=tensor.dtype)
                        selected_parameters += count
                    masked_tensors += 1

                transplanted[name] = tensor

    save_file(transplanted, str(output_weights), metadata=metadata)
    return {
        "base_model": str(base_model),
        "donor_model": str(donor_model),
        "intersection_dir": str(intersection_dir),
        "output": str(output),
        "masked_tensors": masked_tensors,
        "selected_parameters": selected_parameters,
        "weight_file": str(output_weights),
    }


def main():
    args = parse_args()
    base_model = Path(args.base_model).resolve()
    donor_model = Path(args.donor_model).resolve()
    base_mask_dir = Path(args.base_mask_dir).resolve()
    donor_mask_dir = Path(args.donor_mask_dir).resolve()
    intersection_dir = Path(args.intersection_dir).resolve()
    output = Path(args.output).resolve()

    print("[1/4] Creating intersection masks", flush=True)
    intersection_summary, intersection_summary_path = create_intersection_masks(
        base_mask_dir, donor_mask_dir, intersection_dir
    )

    print("[2/4] Transplanting donor parameters into base model", flush=True)
    transplant_summary = transplant_intersection(base_model, donor_model, intersection_dir, output)

    print("[3/4] Writing summaries", flush=True)
    shutil.copy2(intersection_summary_path, output / intersection_summary_path.name)
    full_summary = {
        "mode": "intersection_pure_replacement",
        "formula": "W_new[intersection_mask] = W_donor[intersection_mask]; W_new[~intersection_mask] = W_base[~intersection_mask]",
        "intersection": intersection_summary,
        "transplant": transplant_summary,
    }
    with (output / "intersect_and_transplant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    print("[4/4] Done", flush=True)
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
