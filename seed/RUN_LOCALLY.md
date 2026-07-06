# Parametic — local frontend dev seed

Run the platform UI **locally with real data** so you can iterate on `web/`
with visual feedback (computer-use), no GPU / Postgres / VESSL needed.

This bundle is **data only** (2 succeeded + 2 failed real runs + a SQLite DB).
The code (`parametic_platform/`, including `web/`) comes from the git repo.

## One-time setup

```bash
# 1. From your local clone of the repo (branch experiment/qwen3-8b-calibration),
#    extract this bundle so ./seed/ sits at the repo root:
#       tar xzf parametic-frontend-seed.tar.gz      # creates ./seed/

# 2. Point the bundled DB at your local paths (idempotent; re-run anytime):
python seed/seed_db.py

# 3. Python deps for the host API (no torch/GPU needed):
#    use any python with: pip install -r requirements-platform.txt
```

## Run the UI

```bash
export DATABASE_URL="sqlite:///$PWD/seed/parametic.db"
export PARAMETIC_ALLOW_INTERNAL_MODES=1          # exposes the *-smoke catalog entries
export PARAMETIC_BASIC_AUTH_USER=demo
export PARAMETIC_BASIC_AUTH_PASSWORD=demo
uvicorn parametic_platform.api:app --port 8000
```

Open **http://localhost:8000/app/** (login `demo` / `demo`).

You should see 4 analyses in the list (2 `succeeded`, 2 `failed`). Selecting a
succeeded one renders the Spot Story (figures / metrics / report) from the
bundled artifacts. Selecting a failed one shows the error breadcrumb.

> Submitting a NEW analysis from the local UI will queue a row but never run
> (no worker/GPU here). That's expected — this seed is for UI work, not runs.

## The dev loop

- Edit only `parametic_platform/web/` (`index.html`, `app.js`, `styles.css`).
- Refresh the browser — FastAPI serves the static files directly, no build step.
- Commit `web/` and `git push`; the server picks it up on pull.
- Backend (`api.py` / `worker.py` / `runner.py`) stays on the server, untouched.

See `docs/frontend_dev.md` in the repo for the API data contract, the chosen
stack (buildless reactive), and the IA redesign starting point.
