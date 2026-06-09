"""GET /analyses/{id}/spec: structured reproducibility spec from the manifest."""
from __future__ import annotations


def test_spec_requires_auth(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)
    assert client.get(f"/analyses/{rid}/spec", headers={"Authorization": ""}).status_code == 401


def test_spec_returns_reproducibility_fields(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)

    spec = client.get(f"/analyses/{rid}/spec").json()
    assert spec["analysis"]["model"]["hf_model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert spec["analysis"]["area"]["language"] == "java"
    assert spec["analysis"]["mode"]["seeds"] == [1234, 5678]
    assert spec["analysis"]["k"] == 0.01
    assert spec["analysis"]["pipeline_version"] == "approx-mri-v1"
    assert [s["name"] for s in spec["stages"]] == ["create_masks", "evaluate_ppl_damage"]
    assert spec["created_at"] and spec["finished_at"]


def test_spec_404_when_manifest_missing(client, make_request, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)  # no tree written -> no manifest
    assert client.get(f"/analyses/{rid}/spec").status_code == 404
