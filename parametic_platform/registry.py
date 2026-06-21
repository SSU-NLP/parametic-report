"""Operator-registered models, layered on top of the static catalog.

`MODEL_CATALOG` (catalog.py) stays the curated source of truth; registered models
accumulate beside it in the `registered_models` table. `resolve_model_spec()` is the
single lookup the API/spec builder uses — catalog first, then the registry — so a
resolved registration feeds the existing spec/cache_key flow unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .catalog import MODEL_CATALOG, ModelSpec, get_model
from .db import RegisteredModel
from .resolve import resolve_model

_SPEC_FIELDS = (
    "id", "display_name", "hf_model_id", "config_path",
    "tokenizer_path", "model_output_name", "expected_tensors", "revision",
)


def _row_to_spec(row: RegisteredModel) -> ModelSpec:
    return ModelSpec(
        id=row.id,
        display_name=row.display_name,
        hf_model_id=row.hf_model_id,
        config_path=row.config_path,
        tokenizer_path=row.tokenizer_path,
        model_output_name=row.model_output_name,
        expected_tensors=row.expected_tensors,
        revision=row.revision,
    )


def get_registered_model(session: Session, model_id: str) -> ModelSpec | None:
    row = session.get(RegisteredModel, model_id)
    return _row_to_spec(row) if row is not None else None


def list_registered_models(session: Session) -> list[ModelSpec]:
    rows = session.execute(select(RegisteredModel).order_by(RegisteredModel.created_at.asc())).scalars().all()
    return [_row_to_spec(row) for row in rows]


def resolve_model_spec(session: Session, model_id: str) -> ModelSpec:
    """Catalog first, then the registry. Raises ValueError if neither has it."""
    try:
        return get_model(model_id)
    except ValueError:
        pass
    spec = get_registered_model(session, model_id)
    if spec is None:
        raise ValueError(f"Unsupported model_id: {model_id}")
    return spec


def register_model(
    session: Session,
    hf_model_id: str,
    revision: str = "main",
    *,
    config: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Resolve and persist an HF model. Returns the resolution result with the
    persisted spec. Raises ValueError on an unsupported model or an id collision."""
    result = resolve_model(hf_model_id, revision, config=config, token=token)
    if result["model_spec"] is None:
        reasons = "; ".join(result["compatibility"].get("notes", [])) or "unsupported"
        raise ValueError(f"Cannot register {hf_model_id}: {reasons}")

    spec = result["model_spec"]
    model_id = spec["id"]
    if model_id in MODEL_CATALOG:
        raise ValueError(f"id '{model_id}' collides with a built-in catalog model")
    if session.get(RegisteredModel, model_id) is not None:
        raise ValueError(f"Model '{model_id}' is already registered")

    compat = result["compatibility"]
    session.add(
        RegisteredModel(
            id=model_id,
            display_name=spec["display_name"],
            hf_model_id=spec["hf_model_id"],
            config_path=spec["config_path"],
            tokenizer_path=spec["tokenizer_path"],
            model_output_name=spec["model_output_name"],
            expected_tensors=spec["expected_tensors"],
            revision=spec["revision"],
            model_type=compat.get("model_type"),
            compatibility=compat,
        )
    )
    session.commit()
    return result


def registered_model_dicts(session: Session) -> list[dict[str, Any]]:
    """Registered models as catalog-shaped dicts, tagged with provenance the UI
    badges with (source/model_type/compatibility status)."""
    rows = session.execute(select(RegisteredModel).order_by(RegisteredModel.created_at.asc())).scalars().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = asdict(_row_to_spec(row))
        item["source"] = "registered"
        item["model_type"] = row.model_type
        item["status"] = (row.compatibility or {}).get("status")
        out.append(item)
    return out
