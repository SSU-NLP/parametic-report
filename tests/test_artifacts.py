"""Customer artifact endpoint exposes only report/metrics/figures; hides internals."""
from __future__ import annotations


def test_artifacts_listing_exposes_only_customer_files(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)

    body = client.get(f"/analyses/{rid}/artifacts").json()
    paths = set(body["artifacts"])

    assert "report.md" in paths
    assert "metrics/ppl_damage.json" in paths
    assert any(p.startswith("figures/") for p in paths)

    # internal artifacts must be hidden
    assert "manifest.json" not in paths
    assert not any(p.startswith("masks/") for p in paths)
    assert not any(p.startswith("logs/") for p in paths)
    assert "job_spec.json" not in paths

    kinds = {item["path"]: item["kind"] for item in body["items"]}
    assert kinds["report.md"] == "report"
    assert kinds["metrics/ppl_damage.json"] == "metric"
    assert kinds["figures/approx_spot/spot_mask_atlas_k0.01.png"] == "figure"


def test_artifact_file_serving_and_blocking(client, make_request, write_success_tree, tmp_path):
    art = tmp_path / "art"
    rid, _, _ = make_request(artifact_root=art)
    write_success_tree(art)

    assert client.get(f"/analyses/{rid}/artifacts/report.md").status_code == 200
    assert (
        client.get(f"/analyses/{rid}/artifacts/figures/approx_spot/spot_mask_atlas_k0.01.png").status_code == 200
    )
    # blocked internal files
    assert client.get(f"/analyses/{rid}/artifacts/manifest.json").status_code == 404
    assert client.get(f"/analyses/{rid}/artifacts/masks/code-region/layer0.pt").status_code == 404
    # path traversal rejected
    assert client.get(f"/analyses/{rid}/artifacts/../../etc/passwd").status_code in {400, 404}
