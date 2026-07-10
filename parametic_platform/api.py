from __future__ import annotations

import json
import secrets
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

from .catalog import get_area, get_mode, get_model, public_areas, public_models, public_modes
from .config import load_settings
from .db import AnalysisRequest, Job, find_active_request, find_completed_request, init_db, make_session_factory
from .spec import build_analysis_spec, cache_key

settings = load_settings()
SessionFactory = make_session_factory(settings)

app = FastAPI(title="Parametic Report API", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"


class AnalysisCreate(BaseModel):
    model_id: str = Field(examples=["qwen3-8b"])
    area_id: str = Field(default="java-code")
    mode: str = Field(default="approx-1024")
    k: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    request_id: str
    job_id: str | None
    status: str
    cache_hit: bool
    cache_key: str
    artifact_root: str | None
    manifest_path: str | None
    error: str | None


class AnalysisListItem(BaseModel):
    request_id: str
    model_id: str
    area_id: str
    mode: str
    status: str
    cache_key: str
    error: str | None
    created_at: str


def get_session():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


def auth_enabled() -> bool:
    return bool(settings.basic_auth_user and settings.basic_auth_password)


def auth_challenge() -> Response:
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Parametic Report"'})


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if not auth_enabled():
        raise RuntimeError("Basic auth is required: set PARAMETIC_BASIC_AUTH_USER and PARAMETIC_BASIC_AUTH_PASSWORD")
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return auth_challenge()
    try:
        import base64

        decoded = base64.b64decode(header.removeprefix("Basic "), validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return auth_challenge()
    if not (
        secrets.compare_digest(username, settings.basic_auth_user or "")
        and secrets.compare_digest(password, settings.basic_auth_password or "")
    ):
        return auth_challenge()
    return await call_next(request)


@app.on_event("startup")
def startup() -> None:
    if not auth_enabled():
        raise RuntimeError("Basic auth is required: set PARAMETIC_BASIC_AUTH_USER and PARAMETIC_BASIC_AUTH_PASSWORD")
    init_db(settings)
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    settings.scratch_root.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    # Landing page at the root; the analysis app lives at /app/.
    return FileResponse(WEB_DIR / "landing.html")


@app.get("/models")
def models() -> list[dict[str, Any]]:
    return public_models()


@app.get("/areas")
def areas() -> list[dict[str, Any]]:
    return public_areas()


@app.get("/modes")
def modes() -> list[dict[str, Any]]:
    return public_modes()


def public_analysis_response(
    request: AnalysisRequest,
    *,
    cache_hit: bool,
    job: Job | None = None,
    key: str | None = None,
) -> AnalysisResponse:
    if job is None:
        job = request.jobs[-1] if request.jobs else None
    return AnalysisResponse(
        request_id=request.id,
        job_id=job.id if job else None,
        status=request.status,
        cache_hit=cache_hit,
        cache_key=key or request.cache_key,
        artifact_root=None,
        manifest_path=None,
        error=request.error,
    )


@app.post("/analyses", response_model=AnalysisResponse)
def create_analysis(payload: AnalysisCreate, session: Session = Depends(get_session)) -> AnalysisResponse:
    try:
        model = get_model(payload.model_id)
        area = get_area(payload.area_id)
        mode = get_mode(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not mode.public and not settings.allow_internal_modes:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode}")

    spec = build_analysis_spec(
        model=model,
        area=area,
        mode=mode,
        k=payload.k,
        pipeline_version=settings.pipeline_version,
    )
    key = cache_key(spec)

    completed = find_completed_request(session, key)
    if completed is not None:
        return public_analysis_response(completed, cache_hit=True, key=key)

    active = find_active_request(session, key)
    if active is not None:
        return public_analysis_response(active, cache_hit=False, key=key)

    artifact_root = settings.artifact_root / key
    request = AnalysisRequest(
        cache_key=key,
        model_id=model.id,
        area_id=area.id,
        mode=mode.id,
        k=spec["k"],
        sample_size=mode.sample_size,
        status="queued",
        spec=spec,
        artifact_root=str(artifact_root),
    )
    session.add(request)
    session.flush()
    job = Job(request_id=request.id, cache_key=key, status="queued")
    session.add(job)
    session.commit()

    return public_analysis_response(request, cache_hit=False, job=job, key=key)


@app.get("/analyses", response_model=list[AnalysisListItem])
def list_analyses(limit: int = 50, session: Session = Depends(get_session)) -> list[AnalysisListItem]:
    limit = max(1, min(limit, 200))
    stmt = select(AnalysisRequest).order_by(AnalysisRequest.created_at.desc()).limit(limit)
    rows = session.execute(stmt).scalars().all()
    return [
        AnalysisListItem(
            request_id=row.id,
            model_id=row.model_id,
            area_id=row.area_id,
            mode=row.mode,
            status=row.status,
            cache_key=row.cache_key,
            error=row.error,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@app.get("/analyses/{request_id}", response_model=AnalysisResponse)
def get_analysis(request_id: str, session: Session = Depends(get_session)) -> AnalysisResponse:
    request = session.get(AnalysisRequest, request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="analysis request not found")
    return public_analysis_response(request, cache_hit=request.status == "succeeded")


def safe_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value or value.endswith("/"):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    return path.as_posix()


def is_listable_artifact(relative_path: str) -> bool:
    # Researcher platform: full transparency. Any path that passes the traversal
    # guard is listable/servable; safe_artifact_path raises on unsafe input.
    safe_artifact_path(relative_path)
    return True


def artifact_kind(relative_path: str) -> str:
    if relative_path == "report.md":
        return "report"
    if relative_path == "manifest.json":
        return "spec"
    if relative_path.startswith("metrics/"):
        return "metric"
    if relative_path.startswith("masks/"):
        return "mask"
    if relative_path.startswith("logs/"):
        return "log"
    if relative_path.endswith(".csv"):
        return "table"
    if relative_path.startswith("figures/"):
        return "figure"
    return "artifact"


def request_artifact_root(request_id: str, session: Session) -> Path:
    request = session.get(AnalysisRequest, request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="analysis request not found")
    if not request.artifact_root:
        raise HTTPException(status_code=404, detail="artifact root not available")
    return Path(request.artifact_root)


@app.get("/analyses/{request_id}/artifacts")
def get_artifacts(request_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    root = request_artifact_root(request_id, session)
    if not root.exists():
        return {"artifacts": [], "items": [], "masks": []}
    files = []
    mask_counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if not is_listable_artifact(relative_path):
            continue
        # Masks are 700+ per-tensor files; summarize by region instead of listing each.
        if relative_path.startswith("masks/"):
            region = path.parent.relative_to(root).as_posix()
            mask_counts[region] = mask_counts.get(region, 0) + 1
            continue
        files.append(relative_path)
    files = sorted(files)
    return {
        "artifacts": files,
        "items": [
            {
                "path": path,
                "kind": artifact_kind(path),
                "url": f"/analyses/{request_id}/artifacts/{quote(path)}",
            }
            for path in files
        ],
        "masks": [
            {"path": region, "kind": "mask", "count": count}
            for region, count in sorted(mask_counts.items())
        ],
    }


@app.get("/analyses/{request_id}/spec")
def get_spec(request_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Structured reproducibility spec parsed from the run manifest (full transparency)."""
    root = request_artifact_root(request_id, session)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="spec not available")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="spec not available") from exc
    return {
        "analysis": manifest.get("analysis", {}),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "finished_at": manifest.get("finished_at"),
        "stages": manifest.get("stages", []),
    }


@app.get("/analyses/{request_id}/artifacts/{artifact_path:path}")
def get_artifact_file(request_id: str, artifact_path: str, session: Session = Depends(get_session)) -> FileResponse:
    relative_path = safe_artifact_path(artifact_path)
    if not is_listable_artifact(relative_path):
        raise HTTPException(status_code=404, detail="artifact not found")
    root = request_artifact_root(request_id, session)
    full_path = (root / relative_path).resolve()
    try:
        full_path.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid artifact path") from exc
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type = None
    if full_path.suffix == ".md":
        media_type = "text/markdown"
    elif full_path.suffix == ".json":
        media_type = "application/json"
    elif full_path.suffix == ".png":
        media_type = "image/png"
    elif full_path.suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    return FileResponse(full_path, media_type=media_type)


if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="app")
