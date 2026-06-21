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

# 2026-06-15 — GPU dispatch migrated to VESSL Cloud

## What changed

GPU dispatch moved from **Docker-out-of-Docker (DooD)** — a sibling runner container on the host
Docker daemon — to **VESSL Cloud batch jobs**, reusing the lab's vendored harness in
`scripts/vessl/` (`config.sh`, `push.sh`, `submit.sh`, `watch.sh`, wrapping `vesslctl`). The worker
no longer builds/runs a Docker image; `docker_command()` is gone. GPU dispatch is now isolated in
`worker.py` in four functions: `write_job_spec()`, `vessl_submit()` (shells out to
`scripts/vessl/submit.sh`, parses the `job-...` slug), `vessl_wait()` (polls `vesslctl job show`
until terminal, captures `vesslctl job logs`), and `vessl_fetch_artifacts()`
(`vesslctl volume download`).

Flow: `push.sh` uploads repo code to `/shared/${NS}/code` (S3 object volume); `submit.sh` syncs it
to fast SSD `/work/${NS}/code` and runs the runner there. Container paths:
`artifact_root=/shared/${NS}/results/<cache_key>`, `scratch_root=/work/${NS}/scratch/<job_id>`,
HF cache `=/shared/${NS}/hf-cache`. After the job reaches a terminal state, the worker downloads
`/shared/${NS}/results/<cache_key>` back to the host-local artifact_root, so the **API still serves
artifacts from local disk, unchanged**. `HF_TOKEN` is passed to the job via submit.sh's
`--env HF_TOKEN=...` (the code push excludes `.env`).

**Success contract unchanged:** success ⇔ VESSL job state `succeeded` AND `manifest.json` present
in the (downloaded) artifact_root; failure ⇔ `error.json` breadcrumb `{stage, message}`.

**KEY BENEFIT:** the old DooD path-aliasing gotcha (host-identical paths, must run from
`/home/ssunlp/workspace/...`) is **eliminated** — VESSL volume mounts are clean, no host-real-path
requirement.

## VESSL conventions (from `scripts/vessl/config.sh`)

| Setting | Value |
|---|---|
| Org / team | `SSU-NLPLab` / `Default` |
| GPU spec | `resourcespec-a100x1` (A100 SXM 80GB, $1.55/hr, betelgeuse cluster) |
| Image | stock `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` + `pip install -r requirements-runner.txt` at job start (no custom registry image) |
| Object volume | `objvol-gsvyr0eu87wt` → `/shared` (S3-backed, cross-cluster, downloadable) |
| Cluster volume | `clustervol-r922i766wr02` → `/work` (fast SSD, betelgeuse-local) |
| Namespace | `VESSL_NS=seonghyeon/parametic` (set in repo-root `.vesslrc`) |

## Operational runbook

- After code changes, push to the object volume: `bash scripts/vessl/push.sh`.
- The worker daemon then **submits / watches / downloads automatically** for each queued job — no
  local Docker or GPU needed on the host.
- To debug a single job manually:

  ```bash
  bash scripts/vessl/submit.sh --name <n> --gpus 1 \
    --pip "-r requirements-runner.txt" \
    --env HF_TOKEN=... \
    --cmd "..."
  bash scripts/vessl/watch.sh <job-slug>
  ```

- **Prerequisites:** `vesslctl auth status` must be valid, and `.vesslrc` must be present at the
  repo root (defines `VESSL_NS`).

## Status as of this session (2026-06-15)

**Code: DONE.** All migration code is written and unit-tested; nothing committed yet (branch
`experiment/qwen3-8b-calibration`).

- `parametic_platform/config.py` — DooD settings replaced with VESSL settings; reads `.vesslrc`
  (same env names as `scripts/vessl/config.sh`) so the shell harness and Python stay single-sourced.
- `parametic_platform/worker.py` — `docker_command()` removed; `vessl_submit()` / `vessl_wait()` /
  `vessl_fetch_artifacts()` added; `run_job()` wraps dispatch in try/except (any dispatch error ⇒
  job failed). `vessl_fetch_artifacts()` downloads into a temp dir, locates the result root by its
  `manifest.json`/`error.json` marker, then materializes it at the host-local artifact_root.
- `tests/conftest.py` — `fake_runner` now monkeypatches the three `vessl_*` seams (not
  `subprocess.Popen`), writing the artifact tree into the local artifact_root. **16 tests green**
  (`/opt/conda/bin/python -m pytest tests/ -q`), no GPU/Docker/Postgres/VESSL needed.
- `scripts/vessl/{config,push,submit,watch}.sh` vendored from the `vessl-job` plugin; `submit.sh`
  extended with a repeatable `--env KEY=VALUE` passthrough (for `HF_TOKEN`). `.vesslrc` added.

**Live plumbing smoke: GREEN** (validated end-to-end on VESSL, no GPU):
1. `bash scripts/vessl/push.sh` → uploaded 90 files / 11.9 MB to `objvol-...:seonghyeon/parametic/code`.
2. CPU job (`ubuntu:22.04`, `resourcespec-a100cpu`) succeeded: `/shared` + `/work` mounts present,
   code synced `/shared`→`/work`, sentinel `manifest.json` written to `/shared/.../results/`.
3. `vesslctl volume download ... --remote-prefix <p>` lands files **flat** under the local dir
   (prefix stripped) — matches `vessl_fetch_artifacts()` + `_locate_result_root()`.

**Permissions:** `.claude/settings.local.json` (gitignored) allows `Bash(bash scripts/vessl/*)`
and `Bash(vesslctl *)` so the harness runs without the auto-mode classifier gating it.

**REMAINING — real GPU E2E (not yet run):** an `approx-smoke` analysis on A100×1 through the full
web→worker→download path (run API+worker against a **SQLite** `DATABASE_URL` here, since this host
has no local Docker/Postgres). **Watch the cold-start risk:** the job does `pip install -r
requirements-runner.txt` at start, which includes **deepspeed** — slow/possibly flaky to build on
the stock image. If it bites, the fix is a prebuilt venv on the volume or a custom image (roadmap).

# 2026-06-15 (later) — real GPU E2E PASSED ✅ (4 bugs fixed; uncommitted)

The full web→worker→VESSL-A100→download→API-serve loop is **proven green** with
`llama-3.2-3b` + `java-code-smoke` + `approx-smoke` (k=0.0151):

| Model | PPL |
|---|---:|
| original | 3.31 |
| **code spot top0.0151** | **203,927.76** (×61,600) |
| bottom top0.0151 | 3.35 |
| random_seed1 top0.0151 | 3.38 |

11 figures + 756 masks + metrics + manifest round-tripped to the host-local `artifact_root`;
API serves a figure (HTTP 200 PNG 2247×3205) and `/spec`; DB request → `succeeded`. Ran API +
worker daemon against **SQLite** (`DATABASE_URL=sqlite:////tmp/parametic_e2e.db`,
`PARAMETIC_ALLOW_INTERNAL_MODES=1`, basic auth demo/demo) — no Postgres/Docker on host. Env saved
at `/tmp/parametic_e2e.env`. The cold-start fear was a non-issue: **deepspeed pip install ~50s, fine.**

**The staged de-risk caught 4 real bugs that full-send would have buried.** All fixes are
host-side or container-code and **not yet committed** (branch `experiment/qwen3-8b-calibration`):

1. **Stale base image.** Team-default `VESSL_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`
   (torch 2.3) is too old for the *unpinned* latest `deepspeed`/`transformers` in
   `requirements-runner.txt` (transformers jumped to **5.x**, needs torch ≥2.4; deepspeed import
   hit `torch.library.custom_op` missing). **Fix:** pin `VESSL_IMAGE=` to the DooD-validated
   `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` (same as `Dockerfile.runner`) in **`.vesslrc`**
   — single-sources it for both `config.sh` (sourced) and `config.py` (`vget`).
2. **`submit.sh` pip + tag.** The torch-2.12 image is Debian **PEP-668 externally-managed** → pip
   needs `--break-system-packages` (added to the harness pip line). Also its default `--tag` was
   hardcoded `omni-cons` (vendoring leftover) → now derives from `VESSL_NS` (`${VESSL_NS##*/}`).
   *(Real worker jobs already pass `--tag parametic` explicitly via `worker.py`.)*
3. **`worker.py vessl_submit` slug parse.** `re.search(r"job-[a-z0-9]+", out)` greedily matched
   **`job-spec`** from the echoed `--job-spec` arg, so the worker polled a nonexistent job →
   marked the request false-failed while the **real GPU job ran orphaned**. **Fix:** anchor on
   submit.sh's authoritative `slug: job-...` line (`re.findall(r"slug:\s*(job-[a-z0-9]+)")[-1]`).
4. **HF cache on `/shared` breaks symlinks.** `worker.py vessl_container_paths` put `hf_cache` on
   the S3-backed object volume `/shared`. HuggingFace stores files as `blobs/<sha>` **symlinked**
   into `snapshots/<rev>/`, and the S3 FUSE mount can't resolve symlinks → snapshot entries are
   dead **0-byte files** → `OSError: config.json not valid JSON`. **Fix:** move `hf_cache` to
   `/work` (real SSD fs, symlinks work, betelgeuse-local so still persists across jobs).

**Files touched (uncommitted):** `.vesslrc`, `scripts/vessl/submit.sh`, `parametic_platform/worker.py`.
16 unit tests still green. **Leftover cleanup (optional):** corrupt `~6GB` HF cache at
`/shared/seonghyeon/parametic/hf-cache` (0-byte snapshots + real blobs) is now unused — safe to
delete on the volume; a few `failed` test rows in the SQLite DB are throwaway.

**Operational note:** to restart the host loop here — `set -a; source /tmp/parametic_e2e.env; set +a`,
then `setsid /opt/conda/bin/python -m parametic_platform.worker --init-db --poll-seconds 5 &` and
`setsid /opt/conda/bin/python -m uvicorn parametic_platform.api:app --host 0.0.0.0 --port 8000 &`.
Submit with a **fresh `k`** each time to dodge the cache (e.g. `k=0.0161`).

## IO optimization — masks off the slow S3 volume (verified, 2.3× faster)

The first green VESSL run took ~20 min vs ~4.3 min for the prior local Pro-6000 run. Cause
(from `manifest.json` stage timings, NOT the GPU): the ~8GB of raw mask `.pt` files were written
by `create_masks` to and re-read 4× by `evaluate_ppl_damage` from the **S3-backed `/shared`**
artifact_root. GPU compute (`accumulate`) was identical (~130s) — A100 vs Pro-6000 doesn't matter
for this tiny workload; it was all mask IO.

**Fix (`parametic_platform/runner.py`):** masks now compute on `scratch_root` (`/work` SSD) and
`evaluate_ppl_damage` reads them there. By **default they are NOT published** to artifact_root (the
UI renders only figures/metrics/report) — flip on with `PARAMETIC_PUBLISH_MASKS=1` or a mode
`publish_masks: true`, which runs a `publish_masks` stage that copies them to artifact_root. Added a
`stage()` context manager (times the publish step + drops the failure breadcrumb); `run_step` reuses it.

**Measured (approx-smoke, warm model cache):**

| stage | before (/shared) | after (/work) |
|---|---:|---:|
| accumulate_grad_mul_param | 130s | 129s |
| create_masks | 329s | 206s |
| evaluate_ppl_damage | **606s** | **60s** |
| **TOTAL (stages)** | **1189s (~20m)** | **518s (~8.6m)** |

Host download dropped **7.9GB → 1.8MB** (masks no longer fetched by default). Result unchanged:
code-spot ×66,900 collapse, controls flat. Remaining gap vs local (~257s) is `create_masks` writing
8GB to the cluster SSD (~38MB/s); pushing masks to container-local `/tmp` could shave it but risks
space — not worth it. 16 unit tests green.

# 2026-06-15 (later) — Frontend dev workflow decided + seed bundle shipped

**Motivation:** the user works over SSH on the server, where computer-use (visual/browser
iteration) isn't available, so UI work is hard. Decision: do UI work **locally** against real
data; backend (`api.py`/`worker.py`/`runner.py`) + GPU/DB stay on the server; only `web/` flows
back via git.

**Decisions (user-confirmed):**
- **Dev env:** seed bundle — server tars real run data → user downloads → local `uvicorn` renders
  real-data UI → local Claude edits `web/` (computer-use works) → `git push` → server pulls.
- **Stack for the redesign:** **buildless reactive** (Preact+htm or Alpine via CDN/esm.sh; scatter
  as SVG/canvas). No bundler/`dist/` — FastAPI keeps serving `web/` static. Deploy stays `git pull`.
- **Sequencing:** **full IA/layout redesign first** (from existing data); the K%-slider scatter
  spot-finder (`docs/image.png`) is a follow-up because it needs a new artifact (below).

**Shipped this session:**
- `parametic-frontend-seed.tar.gz` (2.9MB) delivered to the user. Contents: `seed/parametic.db`
  (4 real rows: 2 succeeded `573f63d8`/`f537dfa9`, 2 failed), `seed/platform_artifacts/<key>/`
  (figures+metrics+report, **masks stripped**), `seed/seed_db.py` (rewrites the stored server-
  absolute `artifact_root`/`manifest_path` to the bundle's own local paths — stdlib only,
  idempotent; **required** because `api.py` serves from the stored absolute path), `seed/RUN_LOCALLY.md`.
  Verified end-to-end on the server: seeded API serves `/analyses`, figure (HTTP 200 PNG), `/spec`,
  `/app/` all 200. Local recipe: `python seed/seed_db.py` then `uvicorn` with
  `DATABASE_URL=sqlite:///$PWD/seed/parametic.db`, `PARAMETIC_ALLOW_INTERNAL_MODES=1`, demo/demo.
- `docs/frontend_dev.md` (committed `60713c6`) — local loop, stack decision, full API data contract
  (endpoints, row/artifact shapes, figure-name matching, masks now opt-in/empty-by-default).

**FOLLOW-UP — scatter spot-finder (not started, needs server+GPU):** the existing CSVs are
aggregated per (layer, module); there is **no per-individual-parameter sample** anywhere. The
scatter needs each point = a sampled param at (x=value, y=gradient), blue if score |grad·param| in
top-K%, with a live client-side K% slider. Plan: emit a new `metrics/param_scatter.json` (sample of
N params {value, gradient, score} + global score quantiles for threshold(K)) from the runner near
`scripts/create_approx_spot_masks.py`, one GPU run to produce it; slider then re-colors client-side
(no GPU). Do after the IA redesign settles.

## Session git/runtime state (read before resuming)

- Branch `experiment/qwen3-8b-calibration` is **12 commits ahead of `origin` — NOTHING PUSHED** (the
  entire DooD→VESSL migration `96f5cda` onward, plus this session's `c7caab1` E2E fixes, `5d6548e`
  IO perf, `60713c6` frontend guide). Working tree clean. **Ask before pushing.**
- All server processes are **stopped** (API + worker daemon killed). No VESSL jobs running (all
  terminal). To restart the host loop, see the "Operational note" above (SQLite env at
  `/tmp/parametic_e2e.env`).
- Leftover on volumes (optional cleanup): the corrupt ~6GB HF cache at
  `/shared/seonghyeon/parametic/hf-cache` (0-byte snapshots from the symlink bug) is unused now that
  the cache lives on `/work` — safe to delete. The good `/work` HF cache is warm (model cached).

# 2026-06-15 (later) — UI redesign shipped + experiment handoff

The frontend redesign is **done** on branch `ui/redesign`. Done locally against the seed bundle
(`seed/`, see `seed/RUN_LOCALLY.md`); backend untouched except one authorized line (routing, below).

**Shipped (commits `1a1d2ce`…`34b9af7`, routing `a5567e0`):**
- Buildless **Preact + htm** rewrite of `web/` (importmap via esm.sh, no build step). Split into
  `api.js` / `format.js` / `store.js` / `spotdata.js` / `ui/{common,gallery,report,acts,details,viz}.js`.
- **IA = two routes:** Gallery (`#/`) and Report (`#/a/:id`), hash-routed.
- **Report is tabbed** (not scroll): a pinned hero = the layer×module importance **heatmap with a
  top-K% threshold slider** (dims cells below the cut) + by-depth profile — all drawn in-browser
  from `spot_parameter_summary.csv`. Tabs: What it is · What removing it does · Reproducibility ·
  Artifacts. The "Is it stable?" seed-agreement figures were **dropped** (near-uniform rasters, no
  numeric artifact to redraw them).
- New-analysis request opens in a **modal**. Gallery is a workspace (top bar + "Analyses" toolbar),
  no marketing hero.
- **Landing page at `/`** (`web/landing.html` + `landing.css`), Artificial-Analysis-style: pill nav
  (`Analyses · Paper`), serif hero (Spectral, matches the "Parametic Report" wordmark logo), LATEST/
  METHOD cards, Highlights cards (causal collapse / module concentration / footprint heatmap, real
  numbers), action cards + Recent. CTAs → `/app/`.
- **Routing (the one backend line, authorized):** `api.py` `root()` now serves `landing.html`
  instead of redirecting to `/app/`. So **`/` = landing, `/app/` = app**. Landing is still behind
  basic auth (middleware guards all but `/health`).

**Stack/identity decisions locked:** buildless Preact+htm; serif display (Spectral) is the landing
identity only — the app stays sans. Logo = "Parametic Report" wordmark (the user's artwork is the
real logo; the landing currently approximates it as serif text — swap for the SVG when provided).
"Coding Spot" naming dropped.

**Remaining work to hand to the experiment/data side (UI is ready and waiting):**
1. **per-parameter scatter dump** — emit `metrics/param_scatter.json` (sample of N params
   `{value, gradient, score}` + score quantiles) from the runner near
   `scripts/create_approx_spot_masks.py`; **one GPU run**. Then the K%-slider hero can become the
   real per-parameter scatter (`docs/image.png`) instead of the (layer×module) heatmap stand-in —
   UI side is ~one `spotdata.js` matcher + a scatter component.
2. **Public-model spot gallery (landing "Analyses")** — searchable card list of pre-computed spots
   across many public models; needs the spot computations + a public list endpoint. Reuses
   `ui/gallery.js` cards. Plan only.
3. **Landing auth** — public marketing page would need `/` and its `/app/*` assets exempted from the
   basic-auth middleware; left behind auth for now (the user will revisit).
4. **Real-data smoke check** — verify on a real server run (not seed) that figure-name substring
   matching, empty masks (default), spec fields, and `spot_parameter_summary.csv` still render.
5. **CDN reachability** — Spectral (`fonts.googleapis.com`) + Preact/htm (`esm.sh`) load from CDNs;
   confirm the demo environment can reach them, else self-host.

**Git state:** `ui/redesign` pushed to `origin`. (It carries the prior unpushed
`experiment/qwen3-8b-calibration` commits in its history.)

# 2026-06-15 (later) — UI absorbed into experiment + new-model feature designed

**UI merge.** Fast-forwarded `experiment/qwen3-8b-calibration` ← `origin/ui/redesign` (14 UI
commits: buildless Preact+htm rewrite, landing at `/`, app at `/app/`, heatmap K%-slider hero,
modal request, tabbed report). Backend touch = `api.py` 7 lines (`/` serves `landing.html` instead
of redirecting). **16 unit tests pass.** Working tree clean. `experiment` now == `origin/ui/redesign`
tip and is **14 ahead of `origin/experiment` — still unpushed** (no GitHub creds on the server; push
from an authed local: `git push origin experiment/qwen3-8b-calibration`, a pointer-only update since
the objects are already on `origin/ui/redesign`).

**ENV CHANGE (important):** `/opt/conda` is GONE on this server. Use system `python3` (3.12) with
`pip install --break-system-packages -r requirements-platform.txt -r requirements-dev.txt`
(PEP-668 externally-managed). Run tests with `python3 -m pytest tests/ -q`. CLAUDE.md still says
`/opt/conda/bin/python` — stale.

**Preview app with real data (no GPU/Postgres).** `/tmp/*.db` was wiped, but the real run artifacts
remain under `platform_artifacts/`. Helper `platform_scratch/seed_preview_db.py` (gitignored)
rebuilds a SQLite from the on-disk `manifest.json`/`error.json` (6 rows: 4 succeeded + 2 failed,
llama+qwen, smoke+1024). Launch: `PYTHONPATH=$PWD DATABASE_URL=sqlite:///$PWD/platform_scratch/preview.db
PARAMETIC_ALLOW_INTERNAL_MODES=1 PARAMETIC_BASIC_AUTH_USER=demo PARAMETIC_BASIC_AUTH_PASSWORD=demo
python3 -m uvicorn parametic_platform.api:app --host 0.0.0.0 --port 8000`. UI verified serving
(`/`,`/app/`,`/analyses`,figures all 200).

## NEXT FEATURE — register/run new HuggingFace models (designed, not started)

Goal/vision (user): a **region-detection tool platform**. Phase 1 = **we (operators) register HF
models and accumulate a library**; later = users upload code+weights or pull private HF models.
Architecture support = **broad, via an abstraction** (user picked "더 넓게").

Key finding: most of "download from HF" already works — models load via `hf_model_id`, and
`tokenizer_path` already accepts HF ids (`qwen3-8b` uses `Qwen/Qwen3-8B`; `run_preprocess.sh` passes
HF ids through). The arch-specific surface is just **two spots**: `scripts/create_approx_spot_masks.py`
(~line 93, `startswith("model.layers.")` filter) and `scripts/plot_approx_spot_location.py`
(line 12 `PARAM_RE = model\.layers\.(\d+)\.(.+)` + `MODULE_GROUPS`/`MODULE_ORDER`). The
calibration grad×param dump is already arch-agnostic (HF param names).

Designed approach + recommended build order:
1. **Architecture adapter** (the abstraction): `param_name → (layer_idx, module)` + `is_target(name)`,
   generic over `.layers.N.` / `.h.N.` / `.blocks.N.`; wire into the two scripts, drop the hardcoded
   filter; derive `expected_tensors` by runtime tensor count (no user input). **Start here +
   regression-check that llama-3.2-3b & qwen3-8b produce identical output.**
2. **`resolve_model(hf_id, revision)`** — fetch only `config.json` from HF → detect `model_type` →
   A100-fit size check → derive ModelSpec (display/output name, `tokenizer=hf_id`, size-appropriate
   `config_path`) + compatibility report (supported/needs-review/unsupported). Then a
   `registered_models` DB table so registrations accumulate alongside the curated `catalog.py`
   entries; resolved spec feeds the existing spec/cache_key builder unchanged. Validate with one new
   HF model end-to-end on GPU.
3. **UI** operator "Add model" flow (HF id + revision → resolve → preview report → register), gated
   for now; self-serve (upload / private HF + auth/quota/storage) reuses this path later.

## Session end state
- All server processes **stopped** (preview uvicorn killed). No VESSL jobs running. `/shared` corrupt
  HF cache already deleted; good cache on `/work`.
- Branch `experiment/qwen3-8b-calibration`: UI + VESSL E2E + IO-opt all merged, **unpushed to
  `origin/experiment`** (safe on `origin/ui/redesign`).

# 2026-06-21 — New-model feature: Steps 1+2 shipped + new-model GPU E2E PASSED ✅

Built the **register/run arbitrary HF models** feature (steps 1–2 of the design above) and
**validated it end-to-end on GPU with a brand-new model** (`Qwen/Qwen2.5-1.5B`, never in the
catalog). Two commits on `experiment/qwen3-8b-calibration` (still unpushed to `origin/experiment`):

- `b1306d8` **Step 1 — architecture adapter.** New `scripts/arch_adapter.py`
  (`parse_param`/`is_target`/`layer_pt_files`) generalizes the per-script hardcoded llama/qwen
  `model.layers.N.` parsing over `layers.N`/`h.N`/`blocks.N`/`decoder.layers.N` (llama, qwen, opt,
  gpt2, gptj, falcon, gpt-neox, mpt). Wired into the 3 runner-path scripts
  (`create_approx_spot_masks`, `plot_approx_spot_location`, `plot_seed_agreement_atlas`) + 2 research
  scripts (`compare_seed_stability`, `plot_model_mri`). `plot_spot_overview`'s PARAM_RE was dead code.
  Sibling import resolves because the runner runs scripts by full path (sys.path[0]=scripts/).
- `1c3eb6b` **Step 2 — resolve & register.** `parametic_platform/resolve.py` `resolve_model(hf_id,
  revision)`: fetch only `config.json` → detect arch (model_type/architectures) → estimate params for
  an A100×1 fit gate → derive ModelSpec → compatibility report (supported/needs-review/unsupported).
  **`expected_tensors` derived from config** (exact for llama=254/qwen3=399; skip-safe overestimate +
  `expected_tensors_estimated` flag for un-tabulated families). Network fetch isolated in
  `fetch_hf_config` (pure, unit-testable). `registry.py` + `registered_models` table (db.py);
  `resolve_model_spec()` = catalog-first-then-registry single lookup; `create_analysis` falls back to
  it. API: `POST /models/resolve` (preview), `POST /models/register` (persist), both gated by
  `PARAMETIC_ALLOW_MODEL_REGISTRATION`; `/models` lists catalog + registered. **Worker/runner
  unchanged** — worker reads the baked `request.spec`; registered models reuse the shared `config.json`
  because model-specific values (hf_model_id, tokenizer) flow through runner *args*.

**KEY DESIGN POINT:** a registered model needs **no per-model config file** — only `tokenizer_path`
and `model_name_or_path` differ per model, and both are passed as runner args that override
`config.json`. So `config_path="config.json"` for every registered model; `model_output_name` =
`hf_id.split('/')[-1]` (matches the calibration shell's `${MODEL##*/}`).

**Tests:** `tests/test_arch_adapter.py` (16: identical-to-old-logic on llama/qwen + generalization)
and `tests/test_resolve.py` (14) + `tests/test_registry_api.py` (13). **Full suite 59 green**
(`python3 -m pytest tests/ -q`). conftest now sets `PARAMETIC_ALLOW_MODEL_REGISTRATION=1` and wipes
`RegisteredModel` between tests. `huggingface-hub` added to `requirements-platform.txt` (host now
fetches configs). NOTE: this was **test-after, not TDD** — the user asked for real TDD going forward.

**GPU E2E (the real validation) — PASSED.** Registered `Qwen/Qwen2.5-1.5B` (model_type `qwen2` —
qkv-bias, no qk-norm, *different* from both validated families) via the live API, submitted
`java-code-smoke`+`approx-smoke`, ran on VESSL A100 (~8 min):

| Model | PPL |
|---|---:|
| original | 2.95 |
| **code spot top0.01** | **16,745,231** (×5.68M) |
| bottom top0.01 | 2.97 |
| random_seed1 top0.01 | 3.00 |

`mask_tensors: 336` = exactly qwen2.5-1.5b's layer-tensor count (28 layers × 12) — **proves the arch
adapter parsed qwen2 correctly** (excluded embed/norm/lm_head). All 8 stages succeeded; 11 figures +
2 CSVs (incl `spot_parameter_summary.csv`) round-tripped to host-local artifact_root; API serves a
figure (HTTP 200 PNG 134KB) and the 15-entry artifact listing. The `expected_tensors=366` overestimate
was skip-safe (no false skip). cache_key `4227553136718d9791f12f7d88e1dbd5`, request
`0aa8a59f-7d1d-4505-8db8-0bae893c92a5` in the SQLite E2E DB.

**Operational (this run):** SQLite env at `/tmp/parametic_qwen25_e2e.env`
(`DATABASE_URL=sqlite:////tmp/parametic_qwen25_e2e.db`, registration enabled, demo/demo). Recipe:
`bash scripts/vessl/push.sh` (after code changes — the runner imports `scripts/arch_adapter.py` from
the pushed volume), then `set -a; source /tmp/parametic_qwen25_e2e.env; set +a`, then API
(`python3 -m uvicorn parametic_platform.api:app --port 8000`) + worker
(`python3 -m parametic_platform.worker --init-db --poll-seconds 5`). Both **stopped** at session end.

**Step 3 — operator "Add model" UI: DONE** (commit `f5136c2`, first real TDD pass on this repo).
Backend contract written test-first (red→green): `/models` entries tagged `source`
(catalog|registered), registered entries carry `model_type`+compatibility `status`; new
`GET /capabilities` → `{model_registration}` so the UI gates the flow on the server's
`PARAMETIC_ALLOW_MODEL_REGISTRATION`. UI (`web/`): gallery toolbar "＋ Add model" launcher
(shown only when capable) — HF id+revision → `/models/resolve` preview (compat badge, model_type,
est. params, derived id/expected_tensors, notes) → `/models/register` → `catalog.refresh()` so the
new model shows in the picker immediately. `api.js` gains resolveModel/registerModel/getCapabilities;
`useCatalog` loads capabilities + exposes refresh. `tests/test_models_contract.py` (4); **full suite
63 green**. JS syntax-checked (`node --check`) + backend smoke (capabilities/source/`/app/`/gallery.js
all 200) — but **UI visual rendering not verified headless**; eyeball the modal locally.

**REMAINING:**
- The whole new-model feature (steps 1–3) is **5 commits unpushed to `origin/experiment`**
  (`b1306d8`→`f5136c2`; push from an authed local — no GitHub creds on the server).
- Visual UI check of the Add-model modal (local, per the frontend dev workflow).
- Optional: register `Qwen/Qwen2.5-1.5B` etc. in the *curated* catalog with exact `expected_tensors`
  (336+2=338) if it becomes a standing offering, so the cross-k calibration skip fires.
- Later phases: user self-serve upload / private HF + auth/quota/storage (reuses this path).

