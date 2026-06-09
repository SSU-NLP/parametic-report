"""Shared pytest fixtures for the parametic platform host (API + worker).

Env is configured at MODULE IMPORT TIME (before any test module imports
parametic_platform.api, which snapshots settings at import). Tests therefore run
against a throwaway SQLite DB and tmp artifact/scratch roots — no Postgres, no
Docker, no GPU.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

import pytest

# --- configure env before importing the app (api.py reads settings at import) ---
_ROOT = Path(tempfile.mkdtemp(prefix="parametic-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_ROOT / 'test.db'}"
os.environ["PARAMETIC_ARTIFACT_ROOT"] = str(_ROOT / "artifacts")
os.environ["PARAMETIC_SCRATCH_ROOT"] = str(_ROOT / "scratch")
os.environ["PARAMETIC_BASIC_AUTH_USER"] = "demo"
os.environ["PARAMETIC_BASIC_AUTH_PASSWORD"] = "demo"
os.environ["PARAMETIC_ALLOW_INTERNAL_MODES"] = "1"
os.environ["PARAMETIC_RUNNER_IMAGE"] = "parametic-runner:test"
# Keep scratch so failed-run assertions can inspect it deterministically.
os.environ["PARAMETIC_KEEP_SCRATCH_ON_SUCCESS"] = "1"
os.environ["PARAMETIC_KEEP_SCRATCH_ON_FAILURE"] = "1"

_AUTH_HEADER = "Basic " + base64.b64encode(b"demo:demo").decode("ascii")

# 1x1 transparent PNG so figure files are real, servable bytes.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

REPORT_MARKDOWN = """# Spot Discovery Report

- model: `meta-llama/Llama-3.2-3B-Instruct`
- k: `0.01`

## Coding Spot Figures

### spot_mask_atlas_k0.01

![Top-0.01 mask density per tensor.](figures/approx_spot/spot_mask_atlas_k0.01.png)

_Top-0.01 mask density per tensor._

## PPL Damage Summary

- original: ppl=3.93, loss=1.37
- code top0.01: ppl=115096.54, loss=11.65
"""


def build_success_tree(artifact_root: Path) -> None:
    """Write the customer-visible + internal artifact tree a successful runner produces."""
    artifact_root = Path(artifact_root)
    (artifact_root / "figures" / "approx_spot").mkdir(parents=True, exist_ok=True)
    (artifact_root / "metrics").mkdir(parents=True, exist_ok=True)
    (artifact_root / "masks" / "code-region").mkdir(parents=True, exist_ok=True)
    (artifact_root / "logs").mkdir(parents=True, exist_ok=True)

    (artifact_root / "report.md").write_text(REPORT_MARKDOWN, encoding="utf-8")
    (artifact_root / "figures" / "approx_spot" / "spot_mask_atlas_k0.01.png").write_bytes(_PNG_BYTES)
    (artifact_root / "metrics" / "ppl_damage.json").write_text(
        json.dumps([{"model": "original", "ppl": 3.93, "loss": 1.37}]), encoding="utf-8"
    )
    # Internal-only artifacts that the customer endpoint must hide.
    (artifact_root / "manifest.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    (artifact_root / "masks" / "code-region" / "layer0.pt").write_bytes(b"\x00")


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    """Create tables once against the shared SQLite DB before any test runs."""
    from parametic_platform.config import load_settings
    from parametic_platform.db import init_db

    init_db(load_settings())


@pytest.fixture(autouse=True)
def _clean_db():
    """Isolate tests: wipe the shared SQLite tables before each test."""
    from parametic_platform.config import load_settings
    from parametic_platform.db import AnalysisRequest, Job, make_session_factory

    session_factory = make_session_factory(load_settings())
    with session_factory() as session:
        session.query(Job).delete()
        session.query(AnalysisRequest).delete()
        session.commit()
    yield


@pytest.fixture
def write_success_tree():
    return build_success_tree


@pytest.fixture
def api_module():
    import parametic_platform.api as api

    return api


@pytest.fixture
def client(api_module):
    from starlette.testclient import TestClient

    with TestClient(api_module.app) as test_client:
        test_client.headers.update({"Authorization": _AUTH_HEADER})
        yield test_client


@pytest.fixture
def db_session(api_module):
    """A session bound to the same SQLite DB the API uses."""
    session = api_module.SessionFactory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def settings(api_module):
    return api_module.settings


@pytest.fixture
def fake_runner(monkeypatch):
    """Install a fake docker runner: replaces worker.subprocess.Popen.

    On wait() it writes the artifact tree into `artifact_root` and returns rc.
    `succeed=False` writes an error.json breadcrumb and returns rc=1.
    """
    import parametic_platform.worker as worker

    def install(artifact_root, *, succeed=True, stage="create_masks", message="boom\nOOM at layer 3"):
        target = Path(artifact_root)

        class FakeProcess:
            def __init__(self, cmd, stdout=None, stderr=None, **kwargs):
                self._stdout = stdout

            def wait(self):
                target.mkdir(parents=True, exist_ok=True)
                if succeed:
                    build_success_tree(target)
                    return 0
                (target / "error.json").write_text(
                    json.dumps({"status": "failed", "error": {"stage": stage, "message": message}}),
                    encoding="utf-8",
                )
                return 1

        monkeypatch.setattr(worker.subprocess, "Popen", FakeProcess)

    return install


@pytest.fixture
def make_request(db_session):
    """Create a queued AnalysisRequest + Job, returning (request_id, job_id, artifact_root)."""
    from parametic_platform.db import AnalysisRequest, Job

    created: list[str] = []

    def _make(*, artifact_root, model_id="llama-3.2-3b", area_id="java-code-smoke", mode="approx-smoke", cache_key=None):
        key = cache_key or f"test{len(created)}"
        request = AnalysisRequest(
            cache_key=key,
            model_id=model_id,
            area_id=area_id,
            mode=mode,
            k=0.01,
            sample_size=8,
            status="queued",
            spec={"k": 0.01},
            artifact_root=str(artifact_root),
        )
        db_session.add(request)
        db_session.flush()
        job = Job(request_id=request.id, cache_key=key, status="queued")
        db_session.add(job)
        db_session.commit()
        created.append(request.id)
        return request.id, job.id, Path(artifact_root)

    return _make
