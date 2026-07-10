from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .catalog import AreaSpec, ModeSpec, ModelSpec


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_analysis_spec(
    *,
    model: ModelSpec,
    area: AreaSpec,
    mode: ModeSpec,
    k: float | None,
    pipeline_version: str,
) -> dict[str, Any]:
    selected_k = mode.k if k is None else k
    return {
        "pipeline_version": pipeline_version,
        "model": asdict(model),
        "area": asdict(area),
        "mode": asdict(mode),
        "k": selected_k,
    }


def cache_key(spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return digest[:32]
