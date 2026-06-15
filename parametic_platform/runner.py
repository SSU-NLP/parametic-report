from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_python(repo_root: Path) -> str:
    configured = os.getenv("PARAMETIC_RUNNER_PYTHON") or os.getenv("PARAMETIC_PYTHON_BIN") or os.getenv("PYTHON_BIN")
    if configured:
        return configured
    repo_python = repo_root / ".venv/bin/python"
    if repo_python.exists() and not bool_env("PARAMETIC_IGNORE_REPO_VENV"):
        return str(repo_python)
    return sys.executable


def resolve_deepspeed() -> str:
    configured = os.getenv("PARAMETIC_DEEPSPEED_BIN") or os.getenv("DEEPSPEED_BIN")
    if configured:
        return configured
    return shutil.which("deepspeed") or "deepspeed"


def write_error_breadcrumb(manifest: dict[str, Any], stage: str, message: str) -> None:
    """Persist a failure summary so the worker can surface which stage broke.

    A successful run writes manifest.json; on failure we instead drop error.json
    (manifest.json stays the success signal the worker checks for)."""
    root = manifest.get("_artifact_root")
    if not root:
        return
    snapshot = {key: value for key, value in manifest.items() if key != "_artifact_root"}
    snapshot["status"] = "failed"
    snapshot["error"] = {"stage": stage, "message": message[-4000:]}
    snapshot["finished_at"] = datetime.utcnow().isoformat()
    try:
        Path(root, "error.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except OSError:
        pass


@contextmanager
def stage(name: str, manifest: dict[str, Any]):
    """Record a timed stage in the manifest, dropping an error breadcrumb on failure.

    Use for in-process work (e.g. publishing artifacts); run_step wraps it for
    subprocess steps."""
    print(f"\n== {name} ==", flush=True)
    manifest["stages"].append({"name": name, "status": "running", "started_at": datetime.utcnow().isoformat()})
    try:
        yield
    except Exception as exc:
        manifest["stages"][-1]["status"] = "failed"
        manifest["stages"][-1]["finished_at"] = datetime.utcnow().isoformat()
        print(f"\n!! stage '{name}' failed: {exc}", file=sys.stderr, flush=True)
        write_error_breadcrumb(manifest, name, str(exc))
        raise
    manifest["stages"][-1]["status"] = "succeeded"
    manifest["stages"][-1]["finished_at"] = datetime.utcnow().isoformat()


def run_step(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str] | None, manifest: dict[str, Any]) -> None:
    with stage(name, manifest):
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_cuda(repo_root: Path, python_bin: str, manifest: dict[str, Any]) -> None:
    run_step(
        "cuda_check",
        [
            python_bin,
            "-B",
            "-c",
            "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); "
            "raise SystemExit(0 if torch.cuda.is_available() else 2)",
        ],
        cwd=repo_root,
        env=os.environ.copy(),
        manifest=manifest,
    )


def path_exists_all(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def format_k(k: float) -> str:
    """Match the figure filename suffix used by the plotting scripts."""
    return f"{float(k):g}"


def write_report(artifact_root: Path, manifest: dict[str, Any], ppl_path: Path | None, k: float) -> None:
    kf = format_k(k)
    lines = [
        "# Spot Discovery Report",
        "",
        f"- request_id: `{manifest['request_id']}`",
        f"- job_id: `{manifest['job_id']}`",
        f"- cache_key: `{manifest['cache_key']}`",
        f"- model: `{manifest['analysis']['model']['hf_model_id']}`",
        f"- area: `{manifest['analysis']['area']['id']}`",
        f"- mode: `{manifest['analysis']['mode']['id']}`",
        f"- method: `1024-sample approximate grad * parameter`",
        f"- seeds: `{', '.join(str(seed) for seed in manifest['analysis']['mode']['seeds'])}`",
        f"- k: `{manifest['analysis']['k']}`",
    ]

    # Embed each figure inline as a markdown image with a caption, but only if the
    # plotting stage actually produced the file (so a skipped/failed plot does not
    # leave a broken image in the report). Paths track the requested k.
    figures = [
        (
            f"figures/approx_spot/spot_mask_atlas_k{kf}.png",
            f"Top-{k:g} mask density per tensor — the closest view of where the coding spot lives.",
        ),
        (
            f"figures/approx_spot/spot_importance_atlas_k{kf}.png",
            "Tensor-internal grad * parameter importance intensity.",
        ),
        (
            f"figures/seed_agreement/seed_agreement_atlas_k{kf}.png",
            "Per-tensor agreement between the two calibration seeds (spot stability).",
        ),
        (
            f"figures/seed_agreement/seed_disagreement_atlas_k{kf}.png",
            "Per-tensor seed disagreement (where the approximate spot is least stable).",
        ),
    ]
    present = [(rel, caption) for rel, caption in figures if (artifact_root / rel).exists()]
    if present:
        lines.extend(["", "## Coding Spot Figures", ""])
        for rel, caption in present:
            title = Path(rel).stem
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{caption}]({rel})")
            lines.append("")
            lines.append(f"_{caption}_")
            lines.append("")

    if ppl_path and ppl_path.exists():
        rows = json.loads(ppl_path.read_text(encoding="utf-8"))
        lines.extend(["", "## PPL Damage Summary", ""])
        for row in rows:
            lines.append(f"- {row['model']}: ppl={row['ppl']:.4g}, loss={row['loss']:.4f}")
    (artifact_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Parametic Report analysis job inside an ephemeral container.")
    parser.add_argument("--job-spec", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.job_spec.read_text(encoding="utf-8"))
    analysis = spec["analysis"]
    model = analysis["model"]
    area = analysis["area"]
    mode = analysis["mode"]
    k = float(analysis["k"])
    k_label = f"top{k:g}"

    repo_root = Path(spec["paths"]["repo_root"])
    artifact_root = Path(spec["paths"]["artifact_root"])
    scratch_root = Path(spec["paths"]["scratch_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "figures").mkdir(exist_ok=True)
    (artifact_root / "metrics").mkdir(exist_ok=True)
    (artifact_root / "error.json").unlink(missing_ok=True)

    # Masks are ~8GB/run of raw .pt tensors the UI never renders. Compute them on the
    # fast cluster SSD (scratch_root=/work) so create_masks writes and the 4×
    # evaluate_ppl_damage reads don't hit the slow S3-backed artifact_root (/shared) —
    # that IO was the whole gap vs local runs. Only publish them to artifact_root (and
    # thus download to the host) when explicitly asked, via PARAMETIC_PUBLISH_MASKS=1 or
    # a mode `publish_masks: true`.
    publish_masks = bool_env("PARAMETIC_PUBLISH_MASKS", False) or bool(mode.get("publish_masks", False))
    mask_root = scratch_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "request_id": spec["request_id"],
        "job_id": spec["job_id"],
        "cache_key": spec["cache_key"],
        "analysis": analysis,
        "created_at": datetime.utcnow().isoformat(),
        "status": "running",
        "stages": [],
        "artifacts": {},
        "_artifact_root": str(artifact_root),
    }

    python_bin = resolve_python(repo_root)
    deepspeed_bin = resolve_deepspeed()

    env = os.environ.copy()
    env["PARAMETIC_PYTHON_BIN"] = python_bin
    env["PYTHON_BIN"] = python_bin
    env["PARAMETIC_DEEPSPEED_BIN"] = deepspeed_bin
    env["DEEPSPEED_BIN"] = deepspeed_bin
    env["PARAMETIC_IGNORE_REPO_VENV"] = env.get("PARAMETIC_IGNORE_REPO_VENV", "1")
    env["CONFIG_PATH"] = str(repo_root / model["config_path"])
    env["CALIBRATION_OUTPUT_ROOT"] = str(scratch_root / "calibration")
    env["CALIBRATION_SEEDS"] = " ".join(str(seed) for seed in mode["seeds"])
    env["CALIBRATION_SAVE_SAMPLES"] = str(mode["sample_size"])
    env["CALIBRATION_EXPECTED_TENSORS"] = str(model["expected_tensors"])

    ensure_cuda(repo_root, python_bin, manifest)

    dataset_root = repo_root / "data_preprocess/dataset" / area["dataset_name"]
    jsonl_paths = [
        dataset_root / "train" / f"{area['language']}.jsonl",
        dataset_root / "test" / f"{area['language']}.jsonl",
    ]
    if not path_exists_all(jsonl_paths):
        run_step(
            "dataset_load",
            [
                python_bin,
                str(repo_root / "data_preprocess/create_code_dataset.py"),
                f"--hf_dataset_name={area['hf_dataset_name']}",
                f"--output_dataset_name={area['dataset_name']}",
                f"--total_examples={area['total_examples']}",
                f"--train_ratio={area['train_ratio']}",
                f"--languages={area['language']}",
                f"--config_path={repo_root / model['config_path']}",
            ],
            cwd=repo_root,
            env=env,
            manifest=manifest,
        )
    else:
        manifest["stages"].append({"name": "dataset_load", "status": "cached"})

    tokenizer_name = Path(model["tokenizer_path"]).name
    preprocessed_prefix = dataset_root / "preprocessed" / tokenizer_name
    preprocessed_paths = [
        preprocessed_prefix / "train" / area["language"] / "train.bin",
        preprocessed_prefix / "train" / area["language"] / "train.idx",
        preprocessed_prefix / "test" / area["language"] / "test.bin",
        preprocessed_prefix / "test" / area["language"] / "test.idx",
    ]
    if not path_exists_all(preprocessed_paths):
        run_step(
            "preprocess",
            [
                "bash",
                str(repo_root / "data_preprocess/run_preprocess.sh"),
                area["dataset_name"],
                area["language"],
                model["tokenizer_path"],
            ],
            cwd=repo_root,
            env=env,
            manifest=manifest,
        )
    else:
        manifest["stages"].append({"name": "preprocess", "status": "cached"})

    run_step(
        "accumulate_grad_mul_param",
        [
            "bash",
            str(repo_root / "training/further_training/run_java_sample_calibration.sh"),
            area["dataset_name"],
            tokenizer_name,
            model["hf_model_id"],
            area["language"],
        ],
        cwd=repo_root,
        env=env,
        manifest=manifest,
    )

    checkpoints = [
        scratch_root
        / "calibration"
        / model["model_output_name"]
        / f"seed_{seed}"
        / area["language"]
        / f"grad-mul-param_checkpoint_{mode['sample_size']}"
        for seed in mode["seeds"]
    ]
    code_mask = mask_root / "code-region" / model["id"] / k_label
    control_root = mask_root / "control-region" / model["id"]
    run_step(
        "create_masks",
        [
            python_bin,
            str(repo_root / "scripts/create_approx_spot_masks.py"),
            "--checkpoints",
            *[str(path) for path in checkpoints],
            "--code-output",
            str(code_mask),
            "--control-output-root",
            str(control_root),
            "--k",
            str(k),
            "--k-label",
            k_label,
            "--random-seeds",
            *[str(seed) for seed in mode["random_seeds"]],
            "--device",
            "auto",
        ],
        cwd=repo_root,
        env=env,
        manifest=manifest,
    )

    run_step(
        "plot_approx_spot",
        [
            python_bin,
            str(repo_root / "scripts/plot_approx_spot_location.py"),
            "--checkpoints",
            *[str(path) for path in checkpoints],
            "--output-dir",
            str(artifact_root / "figures" / "approx_spot"),
            "--k",
            str(k),
            "--tile-size",
            str(mode["tile_size"]),
            "--device",
            "auto",
            "--title-prefix",
            f"{model['display_name']} {area['display_name']}",
        ],
        cwd=repo_root,
        env=env,
        manifest=manifest,
    )

    run_step(
        "plot_seed_agreement",
        [
            python_bin,
            str(repo_root / "scripts/plot_seed_agreement_atlas.py"),
            "--seed-a-checkpoint",
            str(checkpoints[0]),
            "--seed-b-checkpoint",
            str(checkpoints[1]),
            "--output-dir",
            str(artifact_root / "figures" / "seed_agreement"),
            "--k",
            str(k),
            "--tile-size",
            str(mode["tile_size"]),
            "--device",
            "auto",
            "--title-prefix",
            f"{model['display_name']} {area['display_name']}",
        ],
        cwd=repo_root,
        env=env,
        manifest=manifest,
    )

    ppl_path = artifact_root / "metrics" / "ppl_damage.json"
    data_prefix = preprocessed_prefix / "test" / area["language"] / "test"
    ppl_cmd = [
        python_bin,
        str(repo_root / "scripts/evaluate_masked_ppl.py"),
        "--data-prefix",
        str(data_prefix),
        "--base-model",
        model["hf_model_id"],
        "--mask",
        f"code_{k_label}={code_mask}",
        "--mask",
        f"bottom_{k_label}={control_root / 'bottom' / k_label}",
    ]
    for seed in mode["random_seeds"]:
        ppl_cmd.extend([
            "--mask",
            f"random_seed{seed}_{k_label}={control_root / f'random_seed{seed}' / k_label}",
        ])
    ppl_cmd.extend([
        "--output",
        str(ppl_path),
        "--max-samples",
        str(mode["ppl_samples"]),
        "--max-seq-len",
        "1024",
        "--batch-size",
        "1",
    ])
    run_step(
        "evaluate_ppl_damage",
        ppl_cmd,
        cwd=repo_root,
        env=env,
        manifest=manifest,
    )

    if publish_masks:
        with stage("publish_masks", manifest):
            shutil.copytree(mask_root, artifact_root / "masks", dirs_exist_ok=True)

    write_report(artifact_root, manifest, ppl_path, k)
    manifest.pop("_artifact_root", None)
    manifest["status"] = "succeeded"
    manifest["finished_at"] = datetime.utcnow().isoformat()
    manifest["artifacts"] = {
        "report": "report.md",
        "manifest": "manifest.json",
        "ppl_damage": "metrics/ppl_damage.json",
        "figures": "figures",
    }
    if publish_masks:
        manifest["artifacts"]["masks"] = "masks"
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {artifact_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
