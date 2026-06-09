"""End-to-end web path: submit from the UI -> daemon runs -> results are visible.

This is the contract the browser UI depends on. The runner is faked (no GPU), but
every API hop the front-end makes is exercised against the real app + DB.
"""
from __future__ import annotations

from parametic_platform.db import AnalysisRequest
from parametic_platform.worker import claim_job, run_job


def test_web_request_to_result(client, db_session, api_module, fake_runner):
    # 1. user submits an analysis from the web form
    resp = client.post(
        "/analyses",
        json={"model_id": "llama-3.2-3b", "area_id": "java-code-smoke", "mode": "approx-smoke"},
    )
    assert resp.status_code == 200
    request_id = resp.json()["request_id"]
    assert resp.json()["status"] == "queued"

    # 2. boot() lists it immediately (queued)
    listed = client.get("/analyses").json()
    assert any(r["request_id"] == request_id and r["status"] == "queued" for r in listed)

    # 3. the worker daemon claims and runs it (runner faked at its artifact root)
    request = db_session.get(AnalysisRequest, request_id)
    fake_runner(request.artifact_root, succeed=True)
    job = claim_job(db_session)
    assert job is not None and job.request_id == request_id
    run_job(api_module.settings, db_session, job)

    # 4. status transitions to succeeded (the poller would pick this up)
    listed = client.get("/analyses").json()
    assert any(r["request_id"] == request_id and r["status"] == "succeeded" for r in listed)
    assert client.get(f"/analyses/{request_id}").json()["status"] == "succeeded"

    # 5. customer-visible results are available
    artifacts = client.get(f"/analyses/{request_id}/artifacts").json()
    kinds = {item["kind"] for item in artifacts["items"]}
    assert {"report", "figure", "metric"} <= kinds

    # 6. the report renders with an inline figure (the demo payload)
    report = client.get(f"/analyses/{request_id}/artifacts/report.md")
    assert report.status_code == 200
    assert "![" in report.text
    assert "figures/approx_spot/spot_mask_atlas_k0.01.png" in report.text

    # 7. the inline figure URL actually serves the image
    figure = client.get(f"/analyses/{request_id}/artifacts/figures/approx_spot/spot_mask_atlas_k0.01.png")
    assert figure.status_code == 200
    assert figure.headers["content-type"] == "image/png"


def test_duplicate_submit_reuses_active_request(client):
    payload = {"model_id": "llama-3.2-3b", "area_id": "java-code-smoke", "mode": "approx-smoke"}
    first = client.post("/analyses", json=payload).json()
    second = client.post("/analyses", json=payload).json()
    # same deterministic spec -> same cache key, reuse the active request (no duplicate queue)
    assert first["cache_key"] == second["cache_key"]
    assert first["request_id"] == second["request_id"]
    assert len(client.get("/analyses").json()) == 1
