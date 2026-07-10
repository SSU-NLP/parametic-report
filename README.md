# Parametic Report

Parametic Report is a research pipeline and local platform for finding coding-related regions in LLM parameters. The core experiment accumulates `gradient * parameter` tensors on code data, converts high-importance positions into parameter masks, damages selected weights, and compares the behavioral effect against matched controls.

The repository currently focuses on Java Coding Spot experiments for Llama 3.2 3B Instruct and Qwen3-8B.

## Repository Layout

- `data_preprocess/`: dataset download, language splitting, and tokenizer-based preprocessing.
- `training/further_training/`: DeepSpeed scripts that accumulate `gradient * parameter` tensors.
- `training/utils/`: shared model, data, DeepSpeed, LoRA, and utility code.
- `region_selection/`: scripts that convert accumulated tensors into top-k boolean masks.
- `damage/`: scripts that zero masked model weights and save damaged Hugging Face models.
- `scripts/`: calibration, mask generation, PPL evaluation, plotting, and reporting helpers.
- `parametic_platform/`: FastAPI API, Postgres job queue integration, Docker worker, runner, and static web UI.
- `docs/`: platform notes and runbooks.
- `reports/`: generated experiment reports and figures. This directory is ignored for new generated outputs.

## Current Status

As of 2026-06-02, the main completed research result is the Java Coding Spot experiment:

- Base Llama model: `meta-llama/Llama-3.2-3B-Instruct`
- Base Qwen model: `Qwen/Qwen3-8B`
- Dataset: `nampdn-ai/tiny-codes`, Java split as `tiny-codes-java-full`
- Llama full Java checkpoint: `training/further_training_java_full/Llama-3.2-3B-Instruct/java/grad-mul-param_checkpoint_10000`
- Llama Java masks: `region_selection_java_full/code-region/llama-3.2-3b/top0.005`, `top0.01`, `top0.03`, `top0.05`
- Llama damaged models: `damage/damaged_models/llama-3.2-3b/java-full/`
- Main Llama report: `reports/java_full_spot_report.md`
- Qwen3-8B sample calibration report: `reports/qwen3_8b_sample_calibration/sample_calibration_report.md`
- Qwen3-8B approximate damage report: `reports/qwen3_8b_approx_s1024_damage_comparison/damage_report.md`

Key finding: Java code-region damage at top-1% collapses Java PPL and limited Java code synthesis benchmarks, while matched random and bottom controls remain close to the original model. For Qwen3-8B, the sample=1024 approximate Java top-1% mask also causes a large Java PPL collapse while controls remain near baseline.

See `handoff.md` for the previous Java full experiment handoff and `docs/platform_mvp.md` for the platform MVP runbook.

## Platform MVP

The platform MVP is a **single-tenant managed service operated by us**. Customers use the web UI at `/app/` to request the 1024-sample approximate Spot analysis and inspect report, metrics, and figures. The API, Postgres database, host worker, Docker runner, GPU access, Hugging Face credentials, and raw artifacts are operational components managed by us.

Deployment shape:

- FastAPI API without Docker socket access.
- Postgres-backed request and job tables.
- Host worker running under `systemd` with host Docker access.
- GPU runner containers launched as sibling containers, not Docker-in-Docker.
- Local artifact and scratch volumes.
- Basic auth for customer-facing routes; `/health` remains unauthenticated.

Supported MVP catalog:

- Models: `llama-3.2-3b`, `qwen3-8b`
- Areas: `java-code`
- Customer mode: `approx-1024`
- Internal validation mode: `approx-smoke`

Customer-visible artifacts are limited to `report.md`, `metrics/*.json`, and `figures/**/*`. Raw masks, logs, job specs, scratch checkpoints, and manifests stay internal.

Full 10k checkpoint discovery is research-only and is not part of the managed MVP. `approx-smoke` is retained only for internal lifecycle checks.

See `docs/platform_mvp.md` for the managed-service runbook.

## Manual Pipeline

Run commands from the directory noted in each command.

Download and split the configured code dataset:

```bash
python data_preprocess/create_code_dataset.py
```

Tokenize one language:

```bash
cd data_preprocess
bash run_preprocess.sh tiny-codes-java-full java tokenizers/llama-3.2
```

Accumulate Java `gradient * parameter` tensors for Llama:

```bash
cd training/further_training
bash code_train_core-10000.sh tiny-codes-java-full llama-3.2 meta-llama/Llama-3.2-3B-Instruct java
```

Run sample calibration accumulation:

```bash
bash training/further_training/run_java_sample_calibration.sh tiny-codes-java-full llama-3.2 meta-llama/Llama-3.2-3B-Instruct java
```

Create approximate spot masks from calibration checkpoints:

```bash
python scripts/create_approx_spot_masks.py \
  --checkpoints <seed-1234-checkpoint> <seed-5678-checkpoint> \
  --code-output <code-mask-dir> \
  --control-output-root <control-mask-root> \
  --k 0.01 \
  --random-seeds 1 2 3 \
  --device auto
```

Evaluate in-memory masked PPL:

```bash
python scripts/evaluate_masked_ppl.py \
  --data-prefix <preprocessed-test-prefix> \
  --base-model <hf-model-id> \
  --mask code_top0.01=<mask-dir> \
  --output reports/ppl_damage.json \
  --max-samples 128 \
  --max-seq-len 1024 \
  --batch-size 1
```

Save a fully damaged Hugging Face model:

```bash
cd damage
python damage_model.py \
  --weights_folder <mask-dir> \
  --original_model <hf-model> \
  --output_dir <out-dir>
```

## Configuration

Runtime defaults live in:

- `config.json` for Llama 3.2 3B Java experiments.
- `config.qwen3-8b.json` for Qwen3-8B Java experiments.

Avoid hard-coded absolute paths in new scripts. Prefer explicit CLI arguments or values from the config files.

## Generated Artifacts

Do not commit downloaded datasets, checkpoints, masks, damaged models, or platform scratch data. These paths are ignored:

- `data_preprocess/dataset/`
- `training/further_training_java_full/`
- `training/sample_calibration_*/`
- `region_selection_java_full/`
- `damage/damaged_models/`
- `platform_artifacts/`
- `platform_scratch/`
- `*.pt`, `*.bin`, `*.idx`, `*.dis`

Keep Hugging Face credentials in `.env` or the environment:

```bash
export HF_TOKEN=<token>
```

## Verification

No dedicated automated test suite is configured. For code changes, run the smallest relevant check:

```bash
python -m py_compile <changed-file.py>
```

For tensor-producing changes, verify expected outputs exist and can be loaded:

```bash
python - <<'PY2'
import torch
path = "<artifact.pt>"
tensor = torch.load(path, map_location="cpu")
print(path, tuple(tensor.shape), tensor.dtype)
PY2
```

For platform changes, smoke-test:

```bash
python -m py_compile parametic_platform/*.py
curl http://localhost:8000/health
curl http://localhost:8000/models
```
