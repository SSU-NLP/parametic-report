#!/usr/bin/env python3
import argparse
import csv
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


def format_k(k):
    return f"{float(k):g}"


def parse_name(path):
    match = PARAM_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def module_filter(module_group):
    allowed = MODULE_GROUPS[module_group]
    if allowed is None:
        return lambda module: True
    return lambda module: module in allowed


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
    if reduce == "max":
        return tensor.amax(dim=(1, 3)).numpy()
    if reduce == "sum":
        return tensor.sum(dim=(1, 3)).numpy()
    return tensor.mean(dim=(1, 3)).numpy()


def load_mean_score(paths):
    score = None
    shape = None
    for path in paths:
        tensor = torch.load(path, map_location="cpu")
        if shape is None:
            shape = tuple(tensor.shape)
            score = tensor.abs().float()
        else:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Shape mismatch for {path.name}: {tuple(tensor.shape)} != {shape}")
            score += tensor.abs().float()
        del tensor
    score /= len(paths)
    return score, shape


def build_items(checkpoint_dirs, k, tile_size, device, module_group):
    keep = module_filter(module_group)
    first_dir = checkpoint_dirs[0]
    files = sorted(first_dir.glob("model.layers.*.pt"))
    if not files:
        raise SystemExit(f"No layer tensors found in {first_dir}")

    items = {}
    rows = []
    module_acc = {}
    processed = 0
    for first_path in files:
        parsed = parse_name(first_path)
        if parsed is None:
            continue
        layer, module = parsed
        if not keep(module):
            continue
        paths = [checkpoint / first_path.name for checkpoint in checkpoint_dirs]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

        score_cpu, shape = load_mean_score(paths)
        numel = score_cpu.numel()
        count = int(k * numel)
        score_flat = score_cpu.reshape(-1).to(device)
        top = stable_top_indices(score_flat, count)
        mask_flat = torch.zeros(numel, dtype=torch.bool, device=device)
        if top.numel() > 0:
            mask_flat[top] = True
        mask = mask_flat.reshape(shape)

        importance_sum = float(score_cpu.sum().item())
        importance_mean = float(score_cpu.mean().item())
        importance_max = float(score_cpu.max().item())
        mask_density = float(mask_flat.float().mean().item())
        masked_importance_mean = float(score_flat[top].mean().item()) if top.numel() > 0 else np.nan

        importance_tile = block_reduce_2d(torch.log10(score_cpu + 1e-30), tile_size, tile_size)
        mask_tile = block_reduce_2d(mask.float(), tile_size, tile_size)
        items[(layer, module)] = {
            "importance_tile": importance_tile,
            "mask_tile": mask_tile,
            "importance_sum": importance_sum,
            "importance_mean": importance_mean,
            "importance_max": importance_max,
            "mask_density": mask_density,
            "masked_importance_mean": masked_importance_mean,
            "selected": int(top.numel()),
            "numel": int(numel),
        }
        rows.append({
            "layer": layer,
            "module": module,
            "numel": int(numel),
            "selected": int(top.numel()),
            "importance_sum": importance_sum,
            "importance_mean": importance_mean,
            "importance_max": importance_max,
            "mask_density": mask_density,
            "masked_importance_mean": masked_importance_mean,
        })
        acc = module_acc.setdefault(module, {"module": module, "numel": 0, "selected": 0, "importance_sum": 0.0})
        acc["numel"] += int(numel)
        acc["selected"] += int(top.numel())
        acc["importance_sum"] += importance_sum

        del score_cpu, score_flat, top, mask_flat, mask
        if device.type == "cuda":
            torch.cuda.empty_cache()
        processed += 1
        if processed % 25 == 0:
            print(f"processed {processed} tensors", flush=True)

    total_importance = sum(row["importance_sum"] for row in rows)
    module_rows = []
    for row in module_acc.values():
        item = dict(row)
        item["importance_share"] = item["importance_sum"] / total_importance if total_importance else 0.0
        item["importance_mean"] = item["importance_sum"] / item["numel"] if item["numel"] else 0.0
        item["mask_density"] = item["selected"] / item["numel"] if item["numel"] else 0.0
        module_rows.append(item)
    module_rows.sort(key=lambda row: row["importance_sum"], reverse=True)
    return items, rows, module_rows


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


def matrix_from_rows(rows, value_key):
    layers = sorted({row["layer"] for row in rows})
    modules = [module for module in MODULE_ORDER if any(row["module"] == module for row in rows)]
    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)
    layer_idx = {layer: idx for idx, layer in enumerate(layers)}
    module_idx = {module: idx for idx, module in enumerate(modules)}
    for row in rows:
        if row["module"] in module_idx:
            matrix[layer_idx[row["layer"]], module_idx[row["module"]]] = row[value_key]
    return matrix, layers, modules


def plot_layer_module_heatmap(rows, value_key, output, title, log_scale=False):
    matrix, layers, modules = matrix_from_rows(rows, value_key)
    plot_matrix = matrix.copy()
    if log_scale:
        plot_matrix = np.log10(np.maximum(plot_matrix, 1e-30))
        label = f"log10({value_key})"
    else:
        label = value_key
    fig, ax = plt.subplots(figsize=(max(8, len(modules) * 1.1), max(7, len(layers) * 0.28)))
    image = ax.imshow(plot_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([MODULE_LABELS.get(module, module) for module in modules], rotation=35, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer:02d}" for layer in layers])
    ax.set_xlabel("module")
    ax.set_ylabel("layer")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_bubble(items, output, title):
    layers = sorted({layer for layer, _ in items})
    modules = ordered_modules(items)
    sums = np.array([item["importance_sum"] for item in items.values()], dtype=float)
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
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="magma", alpha=0.9, edgecolor="#111827", linewidth=0.25)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([MODULE_LABELS.get(module, module) for module in modules], rotation=35, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer:02d}" for layer in layers])
    ax.invert_yaxis()
    ax.set_xlabel("module")
    ax.set_ylabel("layer")
    ax.set_title(title)
    ax.grid(True, alpha=0.18)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10 total mean abs(grad * param)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_csv(rows, output, fields):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description="Plot approximate Coding Spot location from one or more grad*param checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--module-group", choices=sorted(MODULE_GROUPS), default="all")
    parser.add_argument("--title-prefix", default="Approximate Coding Spot")
    args = parser.parse_args()

    device = resolve_device(args.device)
    items, rows, module_rows = build_items(args.checkpoints, args.k, args.tile_size, device, args.module_group)
    if not items:
        raise SystemExit("No tensors matched the requested module group")

    all_importance_tiles = np.concatenate([item["importance_tile"].reshape(-1) for item in items.values()])
    imp_vmin, imp_vmax = np.nanpercentile(all_importance_tiles, [2, 99])
    all_mask_tiles = np.concatenate([item["mask_tile"].reshape(-1) for item in items.values()])
    mask_vmax = max(float(np.nanpercentile(all_mask_tiles, 99)), args.k, 1e-6)

    group_label = args.module_group
    title = f"{args.title_prefix} ({group_label}), k={format_k(args.k)}"
    plot_tile_atlas(
        items,
        "importance_tile",
        args.output_dir / f"spot_importance_atlas_k{format_k(args.k)}.png",
        f"{title}: downsampled log10 mean abs(grad * param)",
        "magma",
        imp_vmin,
        imp_vmax,
    )
    plot_tile_atlas(
        items,
        "mask_tile",
        args.output_dir / f"spot_mask_atlas_k{format_k(args.k)}.png",
        f"{title}: local density of approximate top-k mask",
        "viridis",
        0.0,
        mask_vmax,
    )
    plot_layer_module_heatmap(
        rows,
        "importance_sum",
        args.output_dir / "spot_layer_module_importance_sum.png",
        f"{title}: layer/module total mean abs(grad * param)",
        log_scale=True,
    )
    plot_layer_module_heatmap(
        rows,
        "importance_mean",
        args.output_dir / "spot_layer_module_importance_mean.png",
        f"{title}: layer/module mean abs(grad * param)",
        log_scale=True,
    )
    plot_bubble(items, args.output_dir / f"spot_importance_bubble_map_k{format_k(args.k)}.png", title)

    fields = [
        "layer", "module", "numel", "selected", "importance_sum", "importance_mean",
        "importance_max", "mask_density", "masked_importance_mean",
    ]
    write_csv(rows, args.output_dir / "spot_parameter_summary.csv", fields)
    write_csv(
        module_rows,
        args.output_dir / "spot_module_summary.csv",
        ["module", "numel", "selected", "importance_sum", "importance_share", "importance_mean", "mask_density"],
    )
    manifest = [
        "# Approximate Spot Location Figures",
        "",
        f"- k: {format_k(args.k)}",
        f"- module_group: {args.module_group}",
        f"- score: mean abs(grad * param) across {len(args.checkpoints)} checkpoint(s)",
        "- top-k tie mode: stable cutoff, lower index first",
        "- note: this is a sample-based approximate Java Taylor region, not a full 10k damage-validated Coding Spot unless the checkpoints are full-run checkpoints.",
        "",
        "## Inputs",
    ]
    manifest.extend(f"- {checkpoint}" for checkpoint in args.checkpoints)
    manifest.extend([
        "",
        "## Figures",
        f"- spot_importance_atlas_k{format_k(args.k)}.png: tensor-internal importance intensity.",
        f"- spot_mask_atlas_k{format_k(args.k)}.png: tensor-internal top-k mask density; this is the closest figure to 'where the spot is'.",
        "- spot_layer_module_importance_sum.png: layer/module total importance, size-sensitive.",
        "- spot_layer_module_importance_mean.png: layer/module average importance, size-normalized.",
        f"- spot_importance_bubble_map_k{format_k(args.k)}.png: layer/module total-importance bubble map.",
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"wrote approximate spot figures for {len(items)} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
