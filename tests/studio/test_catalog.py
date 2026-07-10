from parametic_studio.catalog import available_models


def test_catalog_lists_models():
    ms = available_models()
    ids = [m["id"] for m in ms]
    assert any("0.5B" in m["label"] for m in ms)
    assert any("1.5B" in m["label"] for m in ms)
    assert all("id" in m and "label" in m for m in ms)
    assert len(ids) == len(set(ids))  # unique
