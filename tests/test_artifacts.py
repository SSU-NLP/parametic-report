"""Researcher artifact endpoint: full transparency (spec/masks/csv exposed),
masks summarized as groups (not 700+ individual entries), traversal still blocked.
"""
from __future__ import annotations


def test_artifacts_listing_exposes_research_files(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)

    body = client.get(f"/analyses/{rid}/artifacts").json()
    paths = set(body["artifacts"])

    # customer-visible + research files are all listed
    assert "report.md" in paths
    assert "metrics/ppl_damage.json" in paths
    assert "figures/approx_spot/spot_module_summary.csv" in paths
    assert "manifest.json" in paths
    assert "logs/worker.log" in paths

    kinds = {item["path"]: item["kind"] for item in body["items"]}
    assert kinds["report.md"] == "report"
    assert kinds["metrics/ppl_damage.json"] == "metric"
    assert kinds["figures/approx_spot/spot_module_summary.csv"] == "table"
    assert kinds["manifest.json"] == "spec"
    assert kinds["logs/worker.log"] == "log"

    # individual mask tensors are NOT flooded into the listing
    assert not any(p.startswith("masks/") for p in paths)
    # instead masks are summarized as region groups with counts
    masks = body["masks"]
    assert any(m["path"].startswith("masks/code-region/") and m["count"] >= 1 for m in masks)


def test_artifact_file_serving_full_transparency(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)

    assert client.get(f"/analyses/{rid}/artifacts/report.md").status_code == 200
    assert client.get(f"/analyses/{rid}/artifacts/manifest.json").status_code == 200
    assert client.get(f"/analyses/{rid}/artifacts/figures/approx_spot/spot_module_summary.csv").status_code == 200
    # a single mask tensor is downloadable by its full path (researcher transparency)
    assert client.get(f"/analyses/{rid}/artifacts/masks/code-region/llama-3.2-3b/top0.01/layer0.pt").status_code == 200

    # path traversal still rejected
    assert client.get(f"/analyses/{rid}/artifacts/../../etc/passwd").status_code in {400, 404}
    # auth still required
    assert client.get(f"/analyses/{rid}/artifacts/report.md", headers={"Authorization": ""}).status_code == 401
