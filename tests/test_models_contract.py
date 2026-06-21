"""Contract the operator 'Add model' UI consumes: /models source badges + /capabilities.

Written test-first (TDD) for Step 3 of the new-model feature.
"""
from __future__ import annotations

import dataclasses

import pytest

from parametic_platform import resolve
from tests.test_resolve import QWEN3_8B

NEW_HF_ID = "Qwen/Qwen3-8B-Custom"
NEW_SLUG = "qwen3-8b-custom"


@pytest.fixture
def fake_hf_config(monkeypatch):
    def install(config):
        monkeypatch.setattr(resolve, "fetch_hf_config", lambda *a, **k: config)
    return install


def test_models_tags_catalog_source(client):
    listed = client.get("/models").json()
    by_id = {m["id"]: m for m in listed}
    assert by_id["llama-3.2-3b"]["source"] == "catalog"
    assert by_id["qwen3-8b"]["source"] == "catalog"


def test_models_tags_registered_source_with_provenance(client, fake_hf_config):
    fake_hf_config(QWEN3_8B)
    assert client.post("/models/register", json={"hf_model_id": NEW_HF_ID}).status_code == 200

    by_id = {m["id"]: m for m in client.get("/models").json()}
    reg = by_id[NEW_SLUG]
    assert reg["source"] == "registered"
    # provenance the UI badges with
    assert reg["model_type"] == "qwen3"
    assert reg["status"] == "supported"


def test_capabilities_reports_registration_enabled(client):
    body = client.get("/capabilities").json()
    assert body["model_registration"] is True


def test_capabilities_reflects_disabled_gate(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "settings", dataclasses.replace(api_module.settings, allow_model_registration=False))
    body = client.get("/capabilities").json()
    assert body["model_registration"] is False
