# Parametic Report Managed Platform MVP

## Summary

The MVP is a single-tenant managed Spot Discovery service operated by us. Customers use the web UI to request catalog-based analyses and inspect report, metrics, and figures. The API, Postgres database, host worker, Docker runner, GPU access, Hugging Face credentials, scratch cleanup, and raw artifacts are operational concerns managed by us.

The deployment topology is:

```text
FastAPI API, no docker.sock
Postgres job DB
host worker under systemd, with host Docker access
GPU runner containers launched as sibling containers
local artifact and scratch volumes
```

Do not use Docker-in-Docker for the MVP. The worker uses the host Docker daemon; the API must not receive Docker socket access.

## Runbook

Install API dependencies on the API/worker environment:

```bash
python -m pip install -r requirements-platform.txt
```

Build the GPU runner image:

```bash
DOCKER_API_VERSION=1.43 docker build -f Dockerfile.runner -t parametic-runner:latest .
```

Check runner imports:

```bash
DOCKER_API_VERSION=1.43 docker run --rm parametic-runner:latest   python -c "import torch, transformers, datasets, deepspeed, fire; print('runner ok')"
```

Start Postgres. If Compose is available:

```bash
docker compose -f docker-compose.platform.yml up -d postgres
```

If only Docker CLI is available:

```bash
DOCKER_API_VERSION=1.43 docker run -d --name parametic-postgres   -e POSTGRES_DB=parametic   -e POSTGRES_USER=parametic   -e POSTGRES_PASSWORD=parametic   -p 5432:5432   postgres:16
```

Start the API with Basic auth enabled:

```bash
DATABASE_URL=postgresql+psycopg://parametic:parametic@localhost:5432/parametic PARAMETIC_BASIC_AUTH_USER=<user> PARAMETIC_BASIC_AUTH_PASSWORD=<password> PARAMETIC_ARTIFACT_ROOT=/srv/parametic/artifacts PARAMETIC_SCRATCH_ROOT=/srv/parametic/scratch uvicorn parametic_platform.api:app --host 0.0.0.0 --port 8000
```

Open the customer UI:

```text
http://<service-host>:8000/app/
```

Submit an authenticated analysis:

```bash
curl -u <user>:<password> -X POST http://localhost:8000/analyses   -H 'content-type: application/json'   -d '{"model_id":"qwen3-8b","area_id":"java-code","mode":"approx-smoke"}'
```

## Worker Service

Run the worker on the GPU host. `PARAMETIC_REPO_ROOT`, `PARAMETIC_ARTIFACT_ROOT`, and `PARAMETIC_SCRATCH_ROOT` are host paths because Docker bind mounts are resolved by the host daemon.

Example systemd environment:

```ini
[Service]
WorkingDirectory=/srv/parametic/repo
Environment=DATABASE_URL=postgresql+psycopg://parametic:parametic@localhost:5432/parametic
Environment=PARAMETIC_REPO_ROOT=/srv/parametic/repo
Environment=PARAMETIC_ARTIFACT_ROOT=/srv/parametic/artifacts
Environment=PARAMETIC_SCRATCH_ROOT=/srv/parametic/scratch
Environment=PARAMETIC_HF_CACHE=/srv/parametic/hf-cache
Environment=PARAMETIC_RUNNER_IMAGE=parametic-runner:latest
Environment=PARAMETIC_GPU_DEVICE=0
Environment=DOCKER_API_VERSION=1.43
Environment=HF_TOKEN=<token>
ExecStart=/srv/parametic/repo/.venv/bin/python -m parametic_platform.worker --init-db
Restart=always
RestartSec=5
```

## Product Contract

Customer-facing requests are represented by a deterministic analysis spec:

- `model_id`: catalog model, currently `llama-3.2-3b` or `qwen3-8b`.
- `area_id`: target domain/language, currently `java-code`.
- `mode`: analysis mode. Customer-facing requests use `approx-1024`; `approx-smoke` is internal validation only.
- `k`: optional top-k fraction; defaults to the mode value.

The API returns request/job status, cache key, and error text. Internal filesystem paths and raw manifests are not exposed in public responses.

Full 10k checkpoint discovery and 2048-sample modes are not customer-facing MVP features.

## Customer-Visible Artifacts

The runner still writes the full internal artifact tree:

```text
manifest.json
report.md
job_spec.json
figures/
metrics/
masks/
logs/
```

The customer API/UI only exposes:

```text
report.md
metrics/*.json
figures/**/*
```

Raw masks, logs, job specs, scratch checkpoints, and manifests remain internal operational artifacts.

## Smoke Test

Use `approx-smoke` internally for lifecycle validation. Start the API with `PARAMETIC_ALLOW_INTERNAL_MODES=1` before submitting this mode:

```bash
curl -u <user>:<password> -X POST http://localhost:8000/analyses   -H 'content-type: application/json'   -d '{"model_id":"llama-3.2-3b","area_id":"java-code","mode":"approx-smoke"}'
```

Acceptance checks:

- `/health` returns 200 without auth.
- `/models`, `/areas`, `/modes`, `/analyses*`, and `/app/` require Basic auth.
- Duplicate queued/running specs reuse the active request.
- Worker creates `job_spec.json` and `logs/worker.log`.
- Successful runner creates `manifest.json`, `report.md`, `metrics/`, `figures/`, and `masks/`.
- Customer artifact endpoint hides `masks/`, `logs/`, `job_spec.json`, and `manifest.json`.
