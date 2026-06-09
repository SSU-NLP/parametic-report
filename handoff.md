# Handoff: Java Coding Spot Experiment

## Current Claim

As of 2026-05-30, the Java-derived top-1% parameter region can be described as a Java coding spot for `meta-llama/Llama-3.2-3B-Instruct`, with scope limits. The claim is supported by matched damage controls on Java PPL and a limited BigCode HumanEvalPack Java synthesize benchmark.

Safe wording: the Java full-data top-1% region is causally important for Java next-token prediction and Java code synthesis on the tested subset. Do not claim full paper reproduction, cross-language generalization, or non-code preservation yet.

## Repository State

- Branch: `benchmark/bigcode-eval`.
- BigCode Evaluation Harness clone: `/tmp/bigcode-evaluation-harness`.
- JDK inside container: `javac 21.0.11`.
- Hugging Face access is expected through `HF_TOKEN` in `.env`; do not commit `.env`.
- Main report: `reports/java_full_spot_report.md`.

## Completed Work

1. Created Java full dataset and tokenizer preprocessing for `tiny-codes-java-full`.
2. Accumulated `gradient * parameter` tensors for Java full training data at 10,000 examples.
3. Generated code-region masks for `top0.005`, `top0.01`, `top0.03`, and `top0.05`.
4. Generated matched control masks: `bottom`, `random_seed1`, `random_seed2`, `random_seed3`.
5. Damaged original model weights by zeroing selected mask positions for code and control regions.
6. Evaluated Java test-subset PPL/loss across all k values and controls.
7. Built visualization/report assets for module importance, model MRI-style maps, overview, and damage curves.
8. Installed and smoke-tested BigCode Evaluation Harness with HumanEvalPack Java synthesize.
9. Ran limited benchmark comparison on `humanevalsynthesize-java --limit 20` for original, code top0.01, random seed1 top0.01, and bottom top0.01.

## Key Results

Java PPL on the held-out subset:

| Model | PPL |
|---|---:|
| Original | 3.93 |
| Bottom top0.01 | 3.98 |
| Random top0.01 seed 1 | 4.07 |
| Random top0.01 seed 2 | 4.10 |
| Random top0.01 seed 3 | 4.11 |
| Code top0.01 | 115,096.54 |

BigCode HumanEvalPack Java synthesize, `--limit 20`, deterministic decoding:

| Model | pass@1 |
|---|---:|
| Original | 0.80 |
| Bottom top0.01 | 0.80 |
| Random top0.01 seed 1 | 0.75 |
| Code top0.01 | 0.00 |

Primary files:

- `reports/java_full_all_k_control_ppl.json`
- `reports/java_full_all_k_control_comparison/all_k_control_summary.csv`
- `reports/java_full_all_k_control_comparison/all_k_control_curves.png`
- `reports/bigcode_humanevalsynth_java_limit20_summary.csv`
- `reports/bigcode_humanevalsynth_java_limit20_pass1.png`
- `reports/java_full_model_mri_top0.01/`
- `reports/java_full_spot_report.md`

## Next Experiments

1. Run full `humanevalsynthesize-java` without `--limit` for the four top0.01 models: original, code, random_seed1, bottom.
2. Repeat benchmark evaluation for `top0.005`, `top0.03`, and `top0.05`.
3. Add more random seeds to benchmark evaluation, at least seeds 2 and 3 for `top0.01`.
4. Evaluate a non-code or general-language PPL set to check whether damage is code-specific.
5. Evaluate at least one other language from the paper setup to avoid overclaiming Java-only behavior.
6. Inspect generated benchmark outputs for failure mode: repeated tokens, syntax failures, runtime failures, or wrong answers.
7. If reporting externally, include confidence limits: one model, Java-only, HumanEvalPack subset unless the full run is complete.

## Full Benchmark Command Template

Run from repository root. Replace the model path and output label per condition.

```bash
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2-) \
.venv/bin/python /tmp/bigcode-evaluation-harness/main.py \
  --model damage/damaged_models/llama-3.2-3b/java-full/code/top0.01 \
  --tasks humanevalsynthesize-java \
  --prompt codellama \
  --max_length_generation 1536 \
  --temperature 0.0 \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --precision bf16 \
  --use_auth_token \
  --allow_code_execution \
  --save_generations \
  --save_generations_path reports/bigcode_humanevalsynth_java_code_top0.01_full_generations.json \
  --metric_output_path reports/bigcode_humanevalsynth_java_code_top0.01_full_metrics.json
```

Note: BigCode appends the task name to the generation output filename.

## Verification Checklist

- Confirm the starting model is the clean original model before each damage run.
- Confirm each damaged model has `config.json`, tokenizer files, and `model.safetensors`.
- Confirm masks have the same tensor count and selected-position count as expected for each k.
- Confirm `HF_TOKEN` is loaded for every dataset/model download path.
- Confirm benchmark commands use `--allow_code_execution` only in the intended environment.
- Confirm report figures use global normalization for claims and reserve per-tensor normalization for inspection only.

## Known Limits and Risks

- Full HumanEvalPack Java has not been completed yet.
- Current benchmark evidence is `--limit 20`, so report it as a subset result.
- `multiple-java` raw completion was unsuitable for this instruct model; `humanevalsynthesize-java --prompt codellama` produced meaningful smoke-test results.
- Sandbox `bwrap` errors may occur for local tooling; rerun necessary commands with approved escalation when they are blocked by namespace creation.
- Generated datasets, checkpoints, masks, and damaged models are large and should remain uncommitted.

# Handoff: Spot Discovery Platform MVP

## Product Goal

The productization goal is to provide Coding Spot discovery as a single-tenant managed service operated by us. Customers use the web UI to submit catalog-based analyses and inspect report, metrics, and figures. Docker, GPU runners, Hugging Face tokens, raw masks, logs, scratch cleanup, and artifact storage are internal operational concerns.

## Current Platform State

- API package: `parametic_platform/`.
- Static UI: `parametic_platform/web/`.
- Managed-service runbook: `docs/platform_mvp.md`.
- Runner image: `Dockerfile.runner`.
- Host API dependencies: `requirements-platform.txt`.
- GPU runner dependencies: `requirements-runner.txt`.
- Job storage: Postgres via Compose or direct `docker run`.
- Current catalog: models `llama-3.2-3b`, `qwen3-8b`; area `java-code`; customer mode `approx-1024`; internal validation mode `approx-smoke`.

The platform flow is API request -> deterministic spec/cache key -> DB request/job -> host worker claim -> Docker runner -> standardized artifact directory -> customer-safe artifact view.

## Latest Platform Changes

- Added Basic auth support controlled by `PARAMETIC_BASIC_AUTH_USER` and `PARAMETIC_BASIC_AUTH_PASSWORD`.
- Restricted public artifact listing and file access to `report.md`, `metrics/*.json`, and `figures/**/*`.
- Hid internal artifact root and manifest paths from public analysis responses.
- Added `approx-smoke` mode for fast lifecycle validation.
- Updated UI to display customer-visible report, metrics, and figures instead of raw artifact roots.
- Added runner dependency manifest in `requirements-runner.txt`.
- Updated `Dockerfile.runner` to install platform and research-pipeline dependencies.
- Updated runner and existing bash scripts to avoid hard-coded host `.venv` use inside runner containers.

## Next Platform Work

1. Build `parametic-runner:latest` and confirm dependency imports.
2. Run API auth smoke tests with Basic auth env enabled.
3. Run host worker `--once` against a queued `approx-smoke` request and verify `job_spec.json`, `logs/worker.log`, and DB state transitions.
4. Run `approx-smoke` end to end and confirm customer-safe artifacts render in the UI.
5. Keep custom model/dataset input out of MVP until catalog-first lifecycle is stable.

## Known Platform Risks

- Full runner Docker build has not been verified after adding `requirements-runner.txt`.
- End-to-end worker execution still depends on Docker, NVIDIA runtime, CUDA availability, Hugging Face access, and correct host paths.
- The service is single-tenant MVP; shared multi-tenant auth, quota, billing, and object storage are intentionally out of scope for now.

# Handoff: Web→Result Loop + Researcher "Spot Story" UI + TDD (2026-06-09)

## What shipped this session (branch `experiment/qwen3-8b-calibration`)

Commits (newest first): `c88fb87` Spot Story UI · `a7ed8e6` transparency + /spec · `ac39696` web loop + pytest harness · `9f6d35a` java-code-smoke area · `486ff20` inline figures + failure surfacing.

1. **Full Docker E2E proven** on GPU: web submit → host worker → sibling runner container → real spot discovery → artifacts → `succeeded`. Validated with `llama-3.2-3b` + `approx-smoke`.
2. **Web request→result loop wired**: added `GET /analyses` (list, newest-first, `?limit`); `web/app.js boot()` loads the list; results auto-load when a row reaches `succeeded` (no manual click). Closed the "submit → stuck queued, report never shows" gaps.
3. **Persistent worker daemon** is what makes UI submissions actually run on GPU (run it, do not use `--once`).
4. **Full-transparency artifacts** (researcher direction): `is_listable_artifact` exposes the whole tree (manifest/masks/logs/CSV); masks summarized by region in the listing (756 files → 3 groups of 252) but each still downloadable. New `GET /analyses/{id}/spec` returns structured reproducibility spec from the manifest.
5. **Researcher "Spot Story" UI**: single-run hero that narrates a spot in four acts — Where (mask atlas) → What (module importance-share ranking from CSV + layer/module figures) → Stable (seed agreement) → Causal (PPL damage: spot vs equal-size controls), led by a headline collapse ratio. Figures matched by name substring (k-agnostic), click-to-zoom. Right panel = reproducibility spec card + full artifact download list.
6. **First automated tests**: `tests/` (pytest + httpx TestClient, SQLite — no Postgres/Docker/GPU). Runner faked by monkeypatching `worker.subprocess.Popen`. 16 passing: catalog/auth, list endpoint, artifact transparency, worker lifecycle, `/spec`, full web-path integration. Run: `/opt/conda/bin/python -m pytest tests/ -q`. Deps: `requirements-dev.txt`.

## OPEN — UI needs a major rework

The Spot Story is a first pass; the user wants a **대대적 (substantial) UI redesign**. Treat the current `web/` as a working baseline / data-contract proof, not the final design. Revisit layout, IA, and visual design next.

## Operational restart (this DooD container)

- Python with platform deps: **`/opt/conda/bin/python`** (the bare `/usr/bin/python3` has no pip). Runner image already built: `parametic-runner:latest`. `docker` CLI installed; socket mounted.
- **DooD path rule (critical)**: all `PARAMETIC_*` roots must use the host-identical path `/home/ssunlp/workspace/seonghyeon/parametic-report/...` (host `/home/ssunlp/workspace` is bind-mounted at the same path here). The `/workspace` alias does NOT exist on the host daemon.
- Env file: `/tmp/parametic.env` (DATABASE_URL points at gateway `172.17.0.1:5432`, NOT localhost; Basic auth `demo`/`demo`; `PARAMETIC_ALLOW_INTERNAL_MODES=1`; HF_TOKEN loaded from `.env`).
- Restart sequence:
  1. `docker start parametic-postgres` (container stopped, data preserved).
  2. `set -a; source /tmp/parametic.env; set +a`
  3. API: `setsid /opt/conda/bin/python -m uvicorn parametic_platform.api:app --host 0.0.0.0 --port 8000 &` (host:8000 is published).
  4. Worker daemon: `setsid /opt/conda/bin/python -m parametic_platform.worker --init-db --poll-seconds 5 &`
- UI: `http://<host>:8000/app/` (demo/demo). Existing succeeded runs persist in Postgres (e.g. `272fd225…` java k=0.03 shows PPL 3.31→187,905, ×56,751 — the most dramatic).
- Fast smoke run for the daemon: submit `{model_id:llama-3.2-3b, area_id:java-code-smoke, mode:approx-smoke, k:<new>}` (~4 min on GPU). Vary `k` to avoid the cache hit.

## Roadmap (agreed out-of-scope this session)

Multi-run **comparison** view · **intervention** actions (the professor's "부분 극대/극소": ablate/amplify/export a selected layer/module spot — module rows are already structured as the selectable unit) · user **model upload** / self-serve · more languages.

