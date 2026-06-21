"""Registry CRUD + /models endpoints + analysis over a registered model."""
from __future__ import annotations

import pytest

from parametic_platform import registry, resolve
from tests.test_resolve import LLAMA_3_2_3B, QWEN3_8B

# A distinct HF id so the derived slug never collides with the static catalog.
NEW_HF_ID = "Qwen/Qwen3-8B-Custom"
NEW_CFG = dict(QWEN3_8B)
NEW_SLUG = "qwen3-8b-custom"


# --- registry layer -----------------------------------------------------------------

def test_register_and_lookup(db_session):
    result = registry.register_model(db_session, NEW_HF_ID, config=NEW_CFG)
    assert result["model_spec"]["id"] == NEW_SLUG

    spec = registry.get_registered_model(db_session, NEW_SLUG)
    assert spec is not None
    assert spec.hf_model_id == NEW_HF_ID
    assert spec.expected_tensors == 399


def test_resolve_model_spec_prefers_catalog(db_session):
    # catalog entry resolves without touching the registry
    spec = registry.resolve_model_spec(db_session, "llama-3.2-3b")
    assert spec.hf_model_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_resolve_model_spec_falls_back_to_registry(db_session):
    registry.register_model(db_session, NEW_HF_ID, config=NEW_CFG)
    spec = registry.resolve_model_spec(db_session, NEW_SLUG)
    assert spec.hf_model_id == NEW_HF_ID


def test_resolve_model_spec_unknown_raises(db_session):
    with pytest.raises(ValueError):
        registry.resolve_model_spec(db_session, "does-not-exist")


def test_register_duplicate_rejected(db_session):
    registry.register_model(db_session, NEW_HF_ID, config=NEW_CFG)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(db_session, NEW_HF_ID, config=NEW_CFG)


def test_register_catalog_collision_rejected(db_session):
    # An HF id whose slug equals a built-in catalog id must be rejected.
    with pytest.raises(ValueError, match="collides"):
        registry.register_model(db_session, "whoever/llama-3.2-3b", config=LLAMA_3_2_3B)


def test_register_unsupported_rejected(db_session):
    bert = {"model_type": "bert", "architectures": ["BertForMaskedLM"],
            "hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 30522}
    with pytest.raises(ValueError, match="Cannot register"):
        registry.register_model(db_session, "google-bert/bert-base-uncased", config=bert)


# --- API endpoints ------------------------------------------------------------------

@pytest.fixture
def fake_hf_config(monkeypatch):
    """Patch the network config fetch so /resolve and /register stay offline."""
    def install(config):
        monkeypatch.setattr(resolve, "fetch_hf_config", lambda *a, **k: config)
    return install


def test_resolve_endpoint_preview_no_persist(client, db_session, fake_hf_config):
    fake_hf_config(NEW_CFG)
    resp = client.post("/models/resolve", json={"hf_model_id": NEW_HF_ID})
    assert resp.status_code == 200
    assert resp.json()["compatibility"]["status"] == "supported"
    # preview does not persist
    assert registry.get_registered_model(db_session, NEW_SLUG) is None


def test_register_endpoint_persists_and_appears_in_models(client, fake_hf_config):
    fake_hf_config(NEW_CFG)
    resp = client.post("/models/register", json={"hf_model_id": NEW_HF_ID})
    assert resp.status_code == 200

    listed = client.get("/models").json()
    ids = {m["id"] for m in listed}
    assert "llama-3.2-3b" in ids  # static catalog still present
    assert NEW_SLUG in ids        # plus the registered model


def test_register_endpoint_rejects_unsupported(client, fake_hf_config):
    fake_hf_config({"model_type": "bert", "architectures": ["BertForMaskedLM"],
                    "hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 30522})
    resp = client.post("/models/register", json={"hf_model_id": "google-bert/bert-base-uncased"})
    assert resp.status_code == 400


def test_registration_gate_blocks_when_disabled(client, fake_hf_config, monkeypatch, api_module):
    import dataclasses

    fake_hf_config(NEW_CFG)
    # settings is a frozen dataclass — swap the whole object the module reads.
    monkeypatch.setattr(api_module, "settings", dataclasses.replace(api_module.settings, allow_model_registration=False))
    assert client.post("/models/resolve", json={"hf_model_id": NEW_HF_ID}).status_code == 403
    assert client.post("/models/register", json={"hf_model_id": NEW_HF_ID}).status_code == 403


def test_analysis_over_registered_model(client, fake_hf_config):
    # Register, then submit an analysis using the registered model id end-to-end
    # through the spec/cache_key/queue path (no GPU; just confirms it queues).
    fake_hf_config(NEW_CFG)
    assert client.post("/models/register", json={"hf_model_id": NEW_HF_ID}).status_code == 200

    resp = client.post(
        "/analyses",
        json={"model_id": NEW_SLUG, "area_id": "java-code-smoke", "mode": "approx-smoke"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["cache_key"]


def test_analysis_unknown_model_still_400(client):
    resp = client.post("/analyses", json={"model_id": "ghost-model", "area_id": "java-code-smoke", "mode": "approx-smoke"})
    assert resp.status_code == 400
