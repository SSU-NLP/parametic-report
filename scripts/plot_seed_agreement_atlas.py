#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PARAM_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.pt$")
MODULE_ORDER = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]
MODULE_LABELS = {
    "self_attn.q_proj.weight": "q",
    "self_attn.k_proj.weight": "k",
    "self_attn.v_proj.weight": "v",
    "self_attn.o_proj.weight": "o",
    "self_attn.q_norm.weight": "q_norm",
    "self_attn.k_norm.weight": "k_norm",
    "mlp.gate_proj.weight": "gate",
    "mlp.up_proj.weight": "up",
    "mlp.down_proj.weight": "down",
    "input_layernorm.weight": "in_norm",
    "post_attention_layernorm.weight": "post_norm",
}


MODULE_GROUPS = {
    "all": None,
    "attention-proj": {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    },
}


def module_filter(module_group):
    if module_group not in MODULE_GROUPS:
        raise ValueError(f"Unknown module group: {module_group}")
    allowed = MODULE_GROUPS[module_group]
    if allowed is None:
        return lambda module: True
    return lambda module: module in allowed


def format_k(k):
    return f"{float(k):g}"


def parse_name(path):
    match = PARAM_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def stable_top_indices(score, count):
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    if count >= score.numel():
        return torch.arange(score.numel(), dtype=torch.long, device=score.device)
    values = torch.topk(score, count, sorted=False).values
    threshold = values.min()
    above = torch.nonzero(score > threshold, as_tuple=False).flatten()
    needed = count - above.numel()
    if needed <= 0:
        return above[:count]
    equal = torch.nonzero(score == threshold, as_tuple=False).flatten().sort().values[:needed]
    return torch.cat([above, equal])


def block_reduce_2d(tensor, rows, cols, reduce="mean"):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    height, width = tensor.shape
    rows = min(rows, height)
    cols = min(cols, width)
    cropped_h = height - (height % rows)
    cropped_w = width - (width % cols)
    tensor = tensor[:cropped_h, :cropped_w]
    tensor = tensor.reshape(rows, cropped_h // rows, cols, cropped_w // cols)
    if reduce == "sum":
        return tensor.sum(dim=(1, 3)).numpy()
    if reduce == "max":
        return tensor.amax(dim=(1, 3)).numpy()
    return tensor.mean(dim=(1, 3)).numpy()


def build_items(seed_a_dir, seed_b_dir, k, tile_size, device, module_group="all"):
    items = {}
    files = sorted(seed_a_dir.glob("model.layers.*.pt"))
    if not files:
        raise SystemExit(f"No layer tensors found in {seed_a_dir}")
    keep = module_filter(module_group)
    processed = 0
    for idx, path_a in enumerate(files, 1):
        parsed = parse_name(path_a)
        if parsed is None:
            continue
        layer, module = parsed
        if not keep(module):
            continue
        processed += 1
        path_b = seed_b_dir / path_a.name
        if not path_b.exists():
            raise FileNotFoundError(path_b)

        tensor_a = torch.load(path_a, map_location="cpu")
        tensor_b = torch.load(path_b, map_location="cpu")
        if tuple(tensor_a.shape) != tuple(tensor_b.shape):
            raise ValueError(f"Shape mismatch for {path_a.name}: {tuple(tensor_a.shape)} != {tuple(tensor_b.shape)}")
        shape = tuple(tensor_a.shape)
        numel = tensor_a.numel()
        count = int(k * numel)

        score_a = tensor_a.abs().float().reshape(-1).to(device)
        score_b = tensor_b.abs().float().reshape(-1).to(device)
        del tensor_a, tensor_b

        idx_a = stable_top_indices(score_a, count)
        idx_b = stable_top_indices(score_b, count)
        del score_a, score_b

        mask_a = torch.zeros(numel, dtype=torch.bool, device=device)
        mask_b = torch.zeros(numel, dtype=torch.bool, device=device)
        mask_a[idx_a] = True
        mask_b[idx_b] = True
        intersection = mask_a & mask_b
        union = mask_a | mask_b
        xor = mask_a ^ mask_b

        selected_a = int(mask_a.sum().item())
        selected_b = int(mask_b.sum().item())
        inter = int(intersection.sum().item())
        union_count = int(union.sum().item())
        jaccard = inter / union_count if union_count else np.nan

        intersection_tile = block_reduce_2d(intersection.reshape(shape), tile_size, tile_size, reduce="sum")
        union_tile = block_reduce_2d(union.reshape(shape), tile_size, tile_size, reduce="sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            agreement_tile = intersection_tile / union_tile
        agreement_tile[union_tile == 0] = np.nan
        disagreement_tile = block_reduce_2d(xor.reshape(shape), tile_size, tile_size, reduce="mean")
        union_density_tile = block_reduce_2d(union.reshape(shape), tile_size, tile_size, reduce="mean")

        items[(layer, module)] = {
            "agreement_tile": agreement_tile,
            "disagreement_tile": disagreement_tile,
            "union_density_tile": union_density_tile,
            "jaccard": jaccard,
            "selected_a": selected_a,
            "selected_b": selected_b,
            "intersection": inter,
            "union": union_count,
        }

        del idx_a, idx_b, mask_a, mask_b, intersection, union, xor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if processed % 25 == 0:
            print(f"processed {processed} tensors", flush=True)
    return items


def ordered_modules(items):
    present = {module for _, module in items}
    modules = [module for module in MODULE_ORDER if module in present]
    modules.extend(module for module in sorted(present) if module not in modules)
    return modules


def plot_tile_atlas(items, value_key, output, title, cmap, vmin=None, vmax=None):
    layers = sorted({layer for layer, _ in items})
    modules = ordered_modules(items)
    fig, axes = plt.subplots(
        len(layers),
        len(modules),
        figsize=(len(modules) * 1.28, len(layers) * 0.58),
        squeeze=False,
    )
    image = None
    for row_idx, layer in enumerate(layers):
        for col_idx, module in enumerate(modules):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            item = items.get((layer, module))
            if item is None:
                ax.axis("off")
                continue
            image = ax.imshow(item[value_key], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if col_idx == 0:
                ax.set_ylabel(f"L{layer:02d}", rotation=0, labelpad=16, fontsize=7, va="center")
            if row_idx == 0:
                ax.set_title(MODULE_LABELS.get(module, module), fontsize=7, pad=4)
    fig.suptitle(title, fontsize=14, y=0.995)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.55, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bubble(items, output, title):
    layers = sorted({layer for layer, _ in items})
    modules = ordered_modules(items)
    xs, ys, sizes, colors = [], [], [], []
    for layer_idx, layer in enumerate(layers):
        for module_idx, module in enumerate(modules):
            item = items.get((layer, module))
            if item is None:
                continue
            instability = 1.0 - item["jaccard"]
            xs.append(module_idx)
            ys.append(layer_idx)
            colors.append(instability)
            sizes.append(35 + 850 * min(instability / 0.08, 1.0))
    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="magma", vmin=0.0, vmax=max(max(colors), 0.02), alpha=0.9, edgecolor="#111827", linewidth=0.25)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([MODULE_LABELS.get(module, module) for module in modules], rotation=35, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer:02d}" for layer in layers])
    ax.invert_yaxis()
    ax.set_xlabel("module")
    ax.set_ylabel("layer")
    ax.set_title(title)
    ax.grid(True, alpha=0.18)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("instability = 1 - Jaccard")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create MRI-style seed agreement atlases from two grad*param checkpoints.")
    parser.add_argument("--seed-a-checkpoint", required=True, type=Path)
    parser.add_argument("--seed-b-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--title-prefix", default="Seed agreement")
    parser.add_argument("--module-group", choices=sorted(MODULE_GROUPS), default="all")
    args = parser.parse_args()

    device = resolve_device(args.device)
    items = build_items(args.seed_a_checkpoint, args.seed_b_checkpoint, args.k, args.tile_size, device, args.module_group)
    if not items:
        raise SystemExit("No parsed tensors found")

    all_disagreement = np.concatenate([item["disagreement_tile"].reshape(-1) for item in items.values()])
    dis_vmax = max(float(np.nanpercentile(all_disagreement, 99)), args.k * 0.2, 1e-6)
    plot_tile_atlas(
        items,
        "agreement_tile",
        args.output_dir / f"seed_agreement_atlas_k{format_k(args.k)}.png",
        f"{args.title_prefix}: local top-k agreement ({args.module_group}), k={format_k(args.k)}",
        "viridis",
        0.90,
        1.0,
    )
    plot_tile_atlas(
        items,
        "disagreement_tile",
        args.output_dir / f"seed_disagreement_atlas_k{format_k(args.k)}.png",
        f"{args.title_prefix}: local top-k disagreement density ({args.module_group}), k={format_k(args.k)}",
        "magma",
        0.0,
        dis_vmax,
    )
    plot_bubble(
        items,
        args.output_dir / f"seed_instability_bubble_map_k{format_k(args.k)}.png",
        f"{args.title_prefix}: per-tensor instability ({args.module_group}), k={format_k(args.k)}",
    )
    print(f"wrote seed agreement atlases for {len(items)} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
