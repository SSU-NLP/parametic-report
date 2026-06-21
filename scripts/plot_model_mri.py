#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from arch_adapter import parse_param

MODULE_ORDER = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
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
    "mlp.gate_proj.weight": "gate",
    "mlp.up_proj.weight": "up",
    "mlp.down_proj.weight": "down",
    "input_layernorm.weight": "in_norm",
    "post_attention_layernorm.weight": "post_norm",
}


def parse_name(path):
    return parse_param(path.stem)


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
    if reduce == "max":
        return tensor.amax(dim=(1, 3)).numpy()
    return tensor.mean(dim=(1, 3)).numpy()


def build_atlas(checkpoint, mask_dir, tile_size):
    items = {}
    importances = []
    mask_densities = []
    for grad_path in sorted(checkpoint.glob("*.pt")):
        parsed = parse_name(grad_path)
        if parsed is None:
            continue
        layer, module = parsed
        grad = torch.load(grad_path, map_location="cpu").abs()
        imp_tile = block_reduce_2d(torch.log10(grad.float() + 1e-30), tile_size, tile_size)
        importance_sum = float(grad.float().sum().item())

        mask_path = mask_dir / grad_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, map_location="cpu").bool()
            mask_tile = block_reduce_2d(mask.float(), tile_size, tile_size)
            mask_density = float(mask.float().mean().item())
        else:
            mask_tile = np.full((tile_size, tile_size), np.nan)
            mask_density = np.nan

        items[(layer, module)] = {
            "importance_tile": imp_tile,
            "mask_tile": mask_tile,
            "importance_sum": importance_sum,
            "mask_density": mask_density,
        }
        importances.append(importance_sum)
        if not np.isnan(mask_density):
            mask_densities.append(mask_density)
    return items, importances, mask_densities


def plot_tile_atlas(items, value_key, output, title, cmap, vmin=None, vmax=None):
    layers = sorted({layer for layer, _ in items})
    modules = [module for module in MODULE_ORDER if any((layer, module) in items for layer in layers)]
    fig, axes = plt.subplots(
        len(layers),
        len(modules),
        figsize=(len(modules) * 1.45, len(layers) * 0.82),
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
                ax.set_ylabel(f"L{layer:02d}", rotation=0, labelpad=18, fontsize=8, va="center")
            if row_idx == 0:
                ax.set_title(MODULE_LABELS.get(module, module), fontsize=8, pad=4)
    fig.suptitle(title, fontsize=14, y=0.995)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.55, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bubble(items, output):
    layers = sorted({layer for layer, _ in items})
    modules = [module for module in MODULE_ORDER if any((layer, module) in items for layer in layers)]
    sums = np.array([item["importance_sum"] for item in items.values()])
    min_sum = float(sums.min())
    max_sum = float(sums.max())

    xs, ys, sizes, colors = [], [], [], []
    for layer_idx, layer in enumerate(layers):
        for module_idx, module in enumerate(modules):
            item = items.get((layer, module))
            if item is None:
                continue
            score = item["importance_sum"]
            norm = (np.log10(score + 1e-30) - np.log10(min_sum + 1e-30)) / (
                np.log10(max_sum + 1e-30) - np.log10(min_sum + 1e-30) + 1e-12
            )
            xs.append(module_idx)
            ys.append(layer_idx)
            sizes.append(20 + 520 * norm)
            colors.append(np.log10(score + 1e-30))

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="magma", alpha=0.88, edgecolor="#111827", linewidth=0.25)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([MODULE_LABELS.get(module, module) for module in modules], rotation=35, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer:02d}" for layer in layers])
    ax.invert_yaxis()
    ax.set_xlabel("module")
    ax.set_ylabel("layer")
    ax.set_title("Model MRI Bubble Map: total abs(grad * param)")
    ax.grid(True, alpha=0.18)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10 total importance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create MRI-style model atlases for code-region importance and masks.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()

    items, importances, _ = build_atlas(args.checkpoint, args.mask, args.tile_size)
    if not items:
        raise SystemExit(f"No layer tensors found under {args.checkpoint}")

    all_importance_tiles = np.concatenate([item["importance_tile"].reshape(-1) for item in items.values()])
    imp_vmin, imp_vmax = np.nanpercentile(all_importance_tiles, [2, 99])
    plot_tile_atlas(
        items,
        "importance_tile",
        args.output / "model_mri_importance_atlas.png",
        "Model MRI: downsampled log10 abs(grad * param)",
        "magma",
        imp_vmin,
        imp_vmax,
    )
    plot_tile_atlas(
        items,
        "mask_tile",
        args.output / "model_mri_mask_atlas.png",
        "Model MRI: local density of selected code-region mask",
        "viridis",
        0.0,
        max(0.05, max(float(np.nanmax(item["mask_tile"])) for item in items.values())),
    )
    plot_bubble(items, args.output / "model_mri_bubble_map.png")
    print(f"wrote MRI plots for {len(items)} tensors to {args.output}")


if __name__ == "__main__":
    main()
