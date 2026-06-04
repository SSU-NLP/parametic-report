from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PlatformSettings:
    database_url: str
    repo_root: Path
    artifact_root: Path
    scratch_root: Path
    hf_cache_dir: Path
    runner_image: str
    runner_repo_root: str
    runner_artifact_root: str
    runner_scratch_root: str
    runner_hf_cache_dir: str
    gpu_device: str
    pipeline_version: str
    docker_shm_size: str
    keep_scratch_on_success: bool
    keep_scratch_on_failure: bool
    basic_auth_user: str | None
    basic_auth_password: str | None
    allow_internal_modes: bool


def load_settings() -> PlatformSettings:
    repo_root = Path(os.getenv("PARAMETIC_REPO_ROOT", Path.cwd())).resolve()
    artifact_root = Path(os.getenv("PARAMETIC_ARTIFACT_ROOT", repo_root / "platform_artifacts")).resolve()
    scratch_root = Path(os.getenv("PARAMETIC_SCRATCH_ROOT", repo_root / "platform_scratch")).resolve()
    hf_cache_dir = Path(os.getenv("HF_HOME", os.getenv("PARAMETIC_HF_CACHE", Path.home() / ".cache" / "huggingface"))).resolve()
    return PlatformSettings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://parametic:parametic@localhost:5432/parametic",
        ),
        repo_root=repo_root,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        hf_cache_dir=hf_cache_dir,
        runner_image=os.getenv("PARAMETIC_RUNNER_IMAGE", "parametic-runner:latest"),
        runner_repo_root=os.getenv("PARAMETIC_RUNNER_REPO_ROOT", "/workspace"),
        runner_artifact_root=os.getenv("PARAMETIC_RUNNER_ARTIFACT_ROOT", "/artifacts"),
        runner_scratch_root=os.getenv("PARAMETIC_RUNNER_SCRATCH_ROOT", "/scratch"),
        runner_hf_cache_dir=os.getenv("PARAMETIC_RUNNER_HF_CACHE", "/hf-cache"),
        gpu_device=os.getenv("PARAMETIC_GPU_DEVICE", "0"),
        pipeline_version=os.getenv("PARAMETIC_PIPELINE_VERSION", "approx-mri-v1"),
        docker_shm_size=os.getenv("PARAMETIC_DOCKER_SHM_SIZE", "64g"),
        keep_scratch_on_success=_bool_env("PARAMETIC_KEEP_SCRATCH_ON_SUCCESS", False),
        keep_scratch_on_failure=_bool_env("PARAMETIC_KEEP_SCRATCH_ON_FAILURE", True),
        basic_auth_user=os.getenv("PARAMETIC_BASIC_AUTH_USER"),
        basic_auth_password=os.getenv("PARAMETIC_BASIC_AUTH_PASSWORD"),
        allow_internal_modes=_bool_env("PARAMETIC_ALLOW_INTERNAL_MODES", False),
    )
