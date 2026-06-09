"""Worker job lifecycle (queued -> running -> succeeded/failed) with a faked runner.

No Docker/GPU: worker.subprocess.Popen is monkeypatched to write a runner-like
artifact tree (or an error.json breadcrumb on failure).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from parametic_platform.db import AnalysisRequest, Job
from parametic_platform.worker import claim_job, run_job


def test_run_job_success_marks_succeeded(settings, db_session, make_request, fake_runner, tmp_path):
    art = tmp_path / "art"
    rid, jid, _ = make_request(artifact_root=art)
    fake_runner(art, succeed=True)

    job = db_session.get(Job, jid)
    run_job(settings, db_session, job)

    db_session.refresh(job)
    request = db_session.get(AnalysisRequest, rid)
    assert job.status == "succeeded"
    assert job.stage == "complete"
    assert request.status == "succeeded"
    assert request.manifest_path and request.manifest_path.endswith("manifest.json")
    assert request.error is None


def test_run_job_failure_surfaces_stage(settings, db_session, make_request, fake_runner, tmp_path):
    art = tmp_path / "art"
    rid, jid, _ = make_request(artifact_root=art)
    fake_runner(art, succeed=False, stage="create_masks", message="boom\nOOM at layer 3")

    job = db_session.get(Job, jid)
    run_job(settings, db_session, job)

    db_session.refresh(job)
    request = db_session.get(AnalysisRequest, rid)
    assert job.status == "failed"
    assert request.status == "failed"
    assert "create_masks" in (request.error or "")
    assert "OOM at layer 3" in (request.error or "")


def test_claim_job_is_fifo(db_session, tmp_path):
    # Explicit, distinct created_at so the FIFO ordering is deterministic in a fast test.
    base = datetime(2026, 1, 1, 0, 0, 0)
    job_ids = []
    for index, offset in enumerate((0, 10)):
        request = AnalysisRequest(
            cache_key=f"k{index}",
            model_id="llama-3.2-3b",
            area_id="java-code-smoke",
            mode="approx-smoke",
            k=0.01,
            sample_size=8,
            status="queued",
            spec={},
            artifact_root=str(tmp_path / f"a{index}"),
        )
        db_session.add(request)
        db_session.flush()
        job = Job(
            request_id=request.id,
            cache_key=request.cache_key,
            status="queued",
            created_at=base + timedelta(seconds=offset),
        )
        db_session.add(job)
        db_session.flush()
        job_ids.append(job.id)
    db_session.commit()

    claimed = claim_job(db_session)
    assert claimed is not None
    assert claimed.id == job_ids[0]  # oldest queued job first
    assert claimed.status == "running"
    assert claimed.request.status == "running"
