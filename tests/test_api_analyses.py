"""API contract: auth, catalog, and the analyses list/create/get endpoints."""
from __future__ import annotations


def _submit(client, area_id="java-code-smoke"):
    return client.post(
        "/analyses",
        json={"model_id": "llama-3.2-3b", "area_id": area_id, "mode": "approx-smoke"},
    )


# --- harness / auth smoke -------------------------------------------------

def test_health_open_without_auth(client):
    resp = client.get("/health", headers={"Authorization": ""})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_catalog_requires_auth(client):
    assert client.get("/models", headers={"Authorization": ""}).status_code == 401
    assert client.get("/models").status_code == 200


def test_smoke_area_hidden_from_public_listing(client):
    area_ids = [a["id"] for a in client.get("/areas").json()]
    assert "java-code" in area_ids
    assert "java-code-smoke" not in area_ids  # internal (public=False)


# --- GET /analyses list endpoint (the gap being filled) -------------------

def test_list_analyses_requires_auth(client):
    assert client.get("/analyses", headers={"Authorization": ""}).status_code == 401


def test_list_analyses_returns_newest_first(client):
    first = _submit(client).json()["request_id"]
    second = _submit(client, area_id="java-code").json()["request_id"]

    resp = client.get("/analyses")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list)

    ids = [r["request_id"] for r in rows]
    assert first in ids and second in ids
    # newest first: the second submission precedes the first in the list
    assert ids.index(second) < ids.index(first)

    row = next(r for r in rows if r["request_id"] == second)
    for field in ("request_id", "model_id", "area_id", "mode", "status", "cache_key", "created_at"):
        assert field in row
    assert row["model_id"] == "llama-3.2-3b"
    assert row["status"] in {"queued", "running", "succeeded", "failed"}


def test_list_analyses_respects_limit(client):
    # distinct k -> distinct spec/cache key -> three separate requests (no dedup)
    for k in (0.005, 0.01, 0.03):
        client.post(
            "/analyses",
            json={"model_id": "llama-3.2-3b", "area_id": "java-code-smoke", "mode": "approx-smoke", "k": k},
        )
    assert len(client.get("/analyses").json()) == 3
    assert len(client.get("/analyses", params={"limit": 2}).json()) == 2
