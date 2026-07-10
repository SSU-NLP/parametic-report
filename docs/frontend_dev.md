# Frontend dev guide

How to work on the platform UI (`parametic_platform/web/`) **locally** with real
data, the chosen stack for the redesign, and the API data contract the UI consumes.

## Why local

The platform runs on a server reached over SSH, where browser/visual iteration
(computer-use) isn't available. So UI work happens on a laptop against a **seed
bundle** of real run data; the backend (`api.py` / `worker.py` / `runner.py`) and
GPU/DB stay on the server. Only `web/` changes flow back via git.

## Local dev loop

1. Get the seed bundle (`parametic-frontend-seed.tar.gz`, produced on the server)
   and extract `seed/` at the repo root. See `seed/RUN_LOCALLY.md`.
2. `python seed/seed_db.py` then run `uvicorn parametic_platform.api:app --port 8000`
   with `DATABASE_URL=sqlite:///$PWD/seed/parametic.db` + `PARAMETIC_ALLOW_INTERNAL_MODES=1`
   + basic-auth `demo`/`demo`.
3. Open `http://localhost:8000/app/`, edit `web/{index.html,app.js,styles.css}`,
   refresh (no build step — FastAPI serves the files directly).
4. Commit `web/` and push. Server picks it up on pull. Backend untouched.

> A NEW analysis submitted locally just queues (no worker/GPU) — expected.

## Stack decision (redesign)

**Buildless reactive.** Bring in a tiny reactive layer via CDN/esm.sh — Preact +
htm (JSX-like, no build) is the recommended pick — and render the parameter
scatter (see below) with plain SVG/canvas. No bundler, no `dist/`: FastAPI keeps
serving `web/` as static files (`api.py` mounts `/app/`). This keeps deploy a
single `git pull`. Avoid a build step unless the app outgrows this.

Start the IA/layout redesign from the current feature inventory in `app.js`:
catalog selectors + new-analysis form, analyses list (polling), the 4-act "Spot
Story" hero, the reproducibility spec card, the raw-artifact list (masks
summarized by region), and the click-to-zoom figure overlay. Reuse the existing
`api()` / `apiText()` helpers and the artifact `kind`-based grouping.

The eventual hero is the **K%-slider parameter scatter** (`docs/image.png`):
points = sampled parameters at (x = value, y = gradient), blue when their score
|grad·param| is in the top-K%, with a live slider. **This needs a new artifact**
(a per-parameter sample dump) the pipeline doesn't emit yet — a follow-up that
requires a runner change + one GPU run. Lay out a placeholder for it; build it
after the IA settles.

## API data contract

All endpoints are behind HTTP basic auth except `/health`. JSON unless noted.
The current `web/app.js` is the working reference consumer.

| Endpoint | Returns |
|---|---|
| `GET /models`,`/areas`,`/modes` | `[{id, display_name, ...}]` — catalog for the selectors |
| `GET /analyses?limit=` | list of rows, newest-first (see row shape) |
| `POST /analyses` | body `{model_id, area_id, mode, k?}` → a row (`cache_hit` tells if deduped) |
| `GET /analyses/{id}` | one row |
| `GET /analyses/{id}/artifacts` | `{items: [...], masks: [...]}` (see below) |
| `GET /analyses/{id}/spec` | `{analysis, status, created_at, finished_at, stages: [{name,...}]}` |
| `GET <item.url>` | the artifact file itself (PNG / JSON / CSV / text) |

**Row** (`/analyses` items and `/analyses/{id}`):
`{request_id, status, cache_key, error, model_id, area_id, mode, created_at, ...}`.
`status` ∈ `queued|running|succeeded|failed`. Poll while `queued|running`
(app.js polls every 5s).

**Artifacts** `{items, masks}`:
- `items: [{path, url, kind}]` — `kind` ∈ `report|spec|metric|table|figure|log|artifact`.
  Match figures by name substring (k-agnostic), e.g. `spot_mask_atlas`,
  `spot_importance_atlas`, `seed_agreement_atlas`, `layer_module_importance_sum`,
  `importance_bubble_map`. Tables: `*_summary.csv`. Metric: `ppl_damage`.
- `masks: [{path, count}]` — raw masks, summarized by region. **Empty by default**
  now (masks are an opt-in artifact: `PARAMETIC_PUBLISH_MASKS=1`), so the UI must
  handle 0 masks gracefully.

**Key artifacts a succeeded run exposes:** `report.md`, `metrics/ppl_damage.json`
(list of `{model, ppl, loss, zeroed_params, mask_tensors}` for original / code /
bottom / random), `figures/approx_spot/*` (+ `spot_parameter_summary.csv`,
`spot_module_summary.csv` — aggregated per layer/module, NOT per-parameter),
`figures/seed_agreement/*`.
