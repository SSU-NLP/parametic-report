# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two layers over one research idea — the **Coding Spot**: the small top-k% of an LLM's
parameters (ranked by `gradient * parameter` importance on code data) that, when zeroed,
collapses coding ability (PPL ×10^4) while equal-size random/bottom controls barely move.

1. **Research pipeline** (`data_preprocess/`, `training/`, `region_selection/`, `damage/`,
   `scripts/`) — the manual, GPU-heavy steps that discover and causally validate a spot.
   Documented in `README.md` / `AGENTS.md`.
2. **`parametic_platform/`** — a web platform that wraps a fast, approximate version of that
   pipeline (sample-calibrated instead of 10k-example) into a one-click analysis with a job
   queue, GPU runner, artifacts, and a researcher-facing UI. This is the active EMNLP-demo
   surface; most current work happens here.

`handoff.md` is the running session log + operational runbook — **read it first** when
resuming platform work. `DESIGN.md` is the web UI design system.

## Platform architecture (the part that needs reading multiple files)

Request → result flows through four processes that share a Postgres DB and a local filesystem:

- **`api.py`** (FastAPI) — `POST /analyses` builds a deterministic **spec** (`spec.py`) from
  the catalog selection and hashes it to a 32-char **`cache_key`**. Same model+area+mode+k →
  same key → request is **deduplicated** (cache hit, no new job). Serves `GET /analyses`,
  `/analyses/{id}/artifacts`, `/analyses/{id}/spec`, and artifact file downloads. A global
  basic-auth middleware guards everything except `/health`.
- **`catalog.py`** — the only source of truth for what's runnable: `MODEL_CATALOG`,
  `AREA_CATALOG`, `MODE_CATALOG`. Each spec has a `public` flag; `public_*()` filter it so
  internal modes (`approx-smoke`, `approx-2048`, `java-code-smoke`) stay hidden unless
  `PARAMETIC_ALLOW_INTERNAL_MODES=1`. **Add new models/areas/modes here**, not inline.
- **`worker.py`** — daemon loop. `claim_job()` takes the oldest `queued` job (FIFO), then
  `run_job()` dispatches it and blocks on completion. **All GPU dispatch is isolated in four
  functions**: `write_job_spec()` (writes `job_spec.json`), `vessl_submit()` (shells out to
  `scripts/vessl/submit.sh` and parses the `job-...` slug), `vessl_wait()` (polls
  `vesslctl job show` until a terminal state, captures `vesslctl job logs`), and
  `vessl_fetch_artifacts()` (`vesslctl volume download` of the results back to the local
  artifact_root). The GPU job runs on **VESSL Cloud** (see "VESSL dispatch" below), not as a
  local Docker container — the contract below is fixed.
- **`runner.py`** — runs *inside* the GPU container, one job per invocation
  (`python -m parametic_platform.runner --job-spec <path>`). Executes the staged pipeline
  (cuda_check → dataset_load → preprocess → accumulate_grad_mul_param → create_masks →
  plot_approx_spot → plot_seed_agreement → evaluate_ppl_damage → report) by shelling out to the
  `scripts/` and `training/` programs. Writes all outputs under `artifact_root`.

**Worker↔runner contract (do not break):** success ⇔ exit code 0 **and** `manifest.json` exists
in `artifact_root`. Failure ⇔ any stage writes `error.json` (`{stage, message}` breadcrumb) and
exits non-zero. The worker polls nothing mid-run; it reads these two files after the process exits.

**VESSL dispatch:** the worker submits the GPU job to **VESSL Cloud** via the vendored harness in
`scripts/vessl/` (`config.sh`/`push.sh`/`submit.sh`/`watch.sh`, wrapping `vesslctl`). Repo code is
pushed to the S3-backed object volume (`/shared/${NS}/code`) and synced to fast SSD
(`/work/${NS}/code`) before the runner executes there; the job writes `artifact_root` to
`/shared/${NS}/results/<cache_key>`. After the job reaches a terminal state,
`vessl_fetch_artifacts()` downloads those results back to the **host-local** artifact_root, so the
API still serves artifacts from local disk **unchanged**. The old DooD host-identical-path
constraint is gone — VESSL volume mounts are clean and require no host-real path.

## Common commands

```bash
# Tests — no GPU/Docker/Postgres needed (SQLite tmp DB, fake runner via monkeypatched Popen).
# Use the conda python; the repo's /usr/bin venv has no pip.
/opt/conda/bin/python -m pytest tests/ -q
/opt/conda/bin/python -m pytest tests/test_spec.py -q                      # one file
/opt/conda/bin/python -m pytest tests/test_artifacts.py::test_name -q      # one test

# Run the platform locally (see handoff.md for the full VESSL restart recipe)
docker compose -f docker-compose.platform.yml up -d postgres   # Postgres only
bash scripts/vessl/push.sh                                      # push code to the object volume after code changes
uvicorn parametic_platform.api:app --host 0.0.0.0 --port 8000   # API; UI at /app/
python -m parametic_platform.worker --init-db                   # worker daemon (--once for one job)
# The worker submits GPU jobs to VESSL Cloud — no local Docker/GPU needed on the host.

# Smoke checks
python -m py_compile parametic_platform/*.py
curl -u demo:demo http://localhost:8000/health
```

Dependencies: `requirements-platform.txt` (host API/worker), `requirements-dev.txt` (pytest),
`requirements-runner.txt` (`pip install`ed into the stock VESSL image at job start; torch comes
from the base image).

For the manual research pipeline commands (dataset → preprocess → accumulate → masks → damage),
see `README.md` — each script states the directory to run from.

## Conventions & gotchas specific to this repo

- **Python interpreter:** use `/opt/conda/bin/python` on the host (has pip 24.2). The bare
  `/usr/bin/python3` venv lacks pip. Inside the runner image, python is `/usr/bin/python`.
- **Postgres** runs locally; the host API and worker reach it at `localhost:5432`. Default creds
  `parametic/parametic`; basic-auth demo creds `demo/demo`. (The old DooD bridge-gateway
  `172.17.0.1` note no longer applies — the runner now runs on VESSL, not in a sibling container,
  and never talks to Postgres directly.)
- **Backgrounding processes:** `pkill` has killed the controlling shell here (exit 144). Prefer
  `setsid ... & disown` and a targeted `kill <PID>`; match PIDs with
  `ps -eo pid,cmd | grep <x> | grep -v grep` (a bare `pgrep -f` self-matches).
- **Secrets:** `HF_TOKEN` lives in `.env` (gitignored). Never print or commit its value; if you
  need to confirm it's set, have the user run `! grep '^HF_TOKEN=' .env`.
- **Generated dirs are gitignored** and must not be committed: `platform_artifacts/`,
  `platform_scratch/`, datasets, checkpoints, `*.pt/*.bin/*.idx/*.dis`, damaged models.
- **Commits:** only when asked. End commit messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Tests

`tests/conftest.py` is the harness: sets env at import (SQLite tmp DB, `demo/demo`,
`PARAMETIC_ALLOW_INTERNAL_MODES=1`), provides an authed Starlette `TestClient`, and a
`fake_runner` fixture that **monkeypatches `worker.subprocess.Popen`** to write a success tree
(or `error.json`) instead of launching Docker. This lets the full web→worker→artifact loop be
tested with no GPU. When you add an artifact the UI consumes, also add it to the success tree in
`conftest.py` so the data contract stays locked.
