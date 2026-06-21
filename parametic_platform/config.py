from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_vesslrc(repo_root: Path) -> dict[str, str]:
    """Parse the repo-root `.vesslrc` (KEY="value" lines) so config.py and the
    vendored `scripts/vessl/*.sh` stay single-sourced. The shell scripts source
    `.vesslrc` directly; here we read the same keys for path construction."""
    path = repo_root / ".vesslrc"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        values[key.strip()] = raw.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class PlatformSettings:
    database_url: str
    repo_root: Path
    # Host-local roots. artifact_root is where finished VESSL results are downloaded
    # to and where the API serves them from; scratch_root stages job_spec.json.
    artifact_root: Path
    scratch_root: Path
    pipeline_version: str
    keep_scratch_on_success: bool
    keep_scratch_on_failure: bool
    basic_auth_user: str | None
    basic_auth_password: str | None
    allow_internal_modes: bool
    # Operator gate for the HF-model registration endpoints (/models/resolve, /register).
    allow_model_registration: bool
    # --- VESSL Cloud dispatch (see scripts/vessl/config.sh for the team standard) ---
    vessl_ns: str
    vessl_object_vol: str
    vessl_cluster_vol: str
    vessl_object_mnt: str
    vessl_cluster_mnt: str
    vessl_image: str
    vessl_gpu_count: int
    vessl_poll_seconds: float
    vessl_poll_max: int


def load_settings() -> PlatformSettings:
    repo_root = Path(os.getenv("PARAMETIC_REPO_ROOT", Path.cwd())).resolve()
    artifact_root = Path(os.getenv("PARAMETIC_ARTIFACT_ROOT", repo_root / "platform_artifacts")).resolve()
    scratch_root = Path(os.getenv("PARAMETIC_SCRATCH_ROOT", repo_root / "platform_scratch")).resolve()

    vrc = _read_vesslrc(repo_root)

    def vget(name: str, default: str) -> str:
        # precedence: explicit env > .vesslrc > team-standard default
        return os.getenv(name, vrc.get(name, default))

    return PlatformSettings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://parametic:parametic@localhost:5432/parametic",
        ),
        repo_root=repo_root,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        pipeline_version=os.getenv("PARAMETIC_PIPELINE_VERSION", "approx-mri-v1"),
        keep_scratch_on_success=_bool_env("PARAMETIC_KEEP_SCRATCH_ON_SUCCESS", False),
        keep_scratch_on_failure=_bool_env("PARAMETIC_KEEP_SCRATCH_ON_FAILURE", True),
        basic_auth_user=os.getenv("PARAMETIC_BASIC_AUTH_USER"),
        basic_auth_password=os.getenv("PARAMETIC_BASIC_AUTH_PASSWORD"),
        allow_internal_modes=_bool_env("PARAMETIC_ALLOW_INTERNAL_MODES", False),
        allow_model_registration=_bool_env("PARAMETIC_ALLOW_MODEL_REGISTRATION", False),
        vessl_ns=vget("VESSL_NS", "seonghyeon/parametic"),
        vessl_object_vol=vget("VESSL_OBJECT_VOL", "objvol-gsvyr0eu87wt"),
        vessl_cluster_vol=vget("VESSL_CLUSTER_VOL", "clustervol-r922i766wr02"),
        vessl_object_mnt=vget("VESSL_OBJECT_MNT", "/shared"),
        vessl_cluster_mnt=vget("VESSL_CLUSTER_MNT", "/work"),
        vessl_image=vget("VESSL_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"),
        vessl_gpu_count=int(os.getenv("PARAMETIC_VESSL_GPUS", "1")),
        vessl_poll_seconds=float(os.getenv("PARAMETIC_VESSL_POLL_SECONDS", "15")),
        vessl_poll_max=int(os.getenv("PARAMETIC_VESSL_POLL_MAX", "240")),
    )
