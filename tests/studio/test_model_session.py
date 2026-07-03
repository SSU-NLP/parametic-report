import math

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from parametic_studio.kernel.model_session import ModelSession

CELL = "mlp.gate_proj.weight"  # a real per-layer param in tiny Llama

PROMPT = torch.tensor([[1, 2, 3]])


def _tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"  # required for output_attentions
    return LlamaForCausalLM(cfg).eval()


class _Tok:
    def __init__(self, eos):
        self.eos_token_id = eos

    def decode(self, ids):
        return f"<{ids[0]}>"


def test_yields_max_tokens_when_no_eos():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=5))
    assert [e["step"] for e in evs] == [0, 1, 2, 3, 4]
    assert all("token_id" in e and "text" in e for e in evs)


def test_stop_halts_early():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    g = s.generate(PROMPT, max_tokens=100)
    next(g)
    next(g)
    s.stop()
    assert list(g) == []


def test_eos_stops_before_max():
    # greedy is deterministic (seeded weights) → first token is reproducible.
    first = next(ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu")).generate(PROMPT, max_tokens=5))["token_id"]
    s = ModelSession(_tiny(), _Tok(eos=first), torch.device("cpu"))
    assert list(s.generate(PROMPT, max_tokens=5)) == []


def test_generate_with_temperature_samples_tokens():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    torch.manual_seed(1)
    evs = list(s.generate(PROMPT, max_tokens=5, probes=(), temperature=1.0))
    assert [e["step"] for e in evs] == [0, 1, 2, 3, 4]     # sampling still yields a full run
    assert all(0 <= e["token_id"] < 32 for e in evs)       # valid vocab ids


def test_generate_temperature_zero_is_greedy():
    a = list(ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu")).generate(PROMPT, max_tokens=4, probes=()))
    b = list(ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu")).generate(PROMPT, max_tokens=4, probes=(), temperature=0.0))
    assert [e["token_id"] for e in a] == [e["token_id"] for e in b]   # temp 0 == argmax


def test_generate_omits_attn_when_not_subscribed():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=2, probes=()))
    assert [e["step"] for e in evs] == [0, 1]
    assert all("attn" not in e for e in evs)


def test_generate_yields_logitlens_frame():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=2, probes=("logitlens",)))
    L = 2
    for e in evs:
        assert "attn" not in e and "act" not in e
        assert len(e["logit"]) == L  # one row per layer
        assert all(isinstance(r["token"], str) and 0 <= r["prob"] <= 1 for r in e["logit"])


def test_generate_yields_activation_frame():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=3, probes=("activation",)))
    L = 2  # tiny model layers
    for e in evs:
        assert "attn" not in e            # not subscribed
        assert e["act"].shape == (L, 2)   # [L, modules: self_attn, mlp]
        assert (e["act"] >= 0).all()      # norms are non-negative


class _EncTok(_Tok):
    def encode(self, text):
        return [1, 2, 3]


def test_compute_spot_grid_shape_and_nonneg():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    spot = s.compute_spot(["alpha", "beta"])
    assert spot["layers"] == 2
    assert len(spot["modules"]) > 0
    assert len(spot["grid"]) == 2                              # one row per layer
    assert all(len(row) == len(spot["modules"]) for row in spot["grid"])
    assert all(v >= 0 for row in spot["grid"] for v in row)    # |grad*param| ≥ 0


def test_spot_step_accumulates_monotonically():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    acc = {}
    g1 = s.spot_step("alpha", acc, 1)              # running sum / 1
    sum1 = {k: v for k, v in acc.items()}
    g2 = s.spot_step("beta", acc, 2)              # running sum / 2
    assert g2["layers"] == 2 and len(g2["grid"]) == 2
    assert all(len(row) == len(g2["modules"]) for row in g2["grid"])
    assert all(acc[k] >= sum1[k] for k in sum1)    # accumulator only grows
    # compute_spot over the same examples == last spot_step's normalized grid
    g_all = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu")).compute_spot(["alpha", "beta"])
    assert g1["grid"] and g_all["grid"] == g2["grid"]


def test_generate_text_encodes_prompt():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    evs = list(s.generate_text("hi", max_tokens=3))
    assert [e["step"] for e in evs] == [0, 1, 2]


# ---- knob (B1): Locate × Edit × Evaluate ----

def test_locate_cell_targets_one_param_fully():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    region = s.locate_cell(0, CELL)
    assert list(region) == [f"model.layers.0.{CELL}"]
    assert region[f"model.layers.0.{CELL}"].all()  # whole weight selected


def test_locate_spot_selects_topk_fraction():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    region = s.locate_spot(["alpha", "beta"], topk=0.1)
    total = sum(m.numel() for m in region.values())
    sel = sum(int(m.sum()) for m in region.values())
    assert 0 < sel <= total * 0.2          # ~10% (slack for ties/discreteness)
    assert all(m.dtype == torch.bool for m in region.values())


def test_locate_spot_empty_examples_is_empty_region():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    assert s.locate_spot([], topk=0.1) == {}   # no torch.cat([]) crash
    s.intervene(s.locate_spot([]), "zero")     # empty region is a safe no-op
    s.clear()


def test_locate_spot_is_per_param_topk():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    region = s.locate_spot(["alpha", "beta"], topk=0.5)
    for name, m in region.items():
        p = dict(s.model.named_parameters())[name]
        assert int(m.sum()) == int(0.5 * p.numel())  # exactly 50% within each param


def test_intervene_clear_restores_weights_exactly():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    before = {n: p.data.clone() for n, p in s.model.named_parameters()}
    s.intervene(s.locate_cell(0, CELL), "zero")
    name = f"model.layers.0.{CELL}"
    now = dict(s.model.named_parameters())[name].data
    assert not torch.allclose(now, before[name])      # actually changed
    s.clear()
    for n, p in s.model.named_parameters():
        assert torch.allclose(p.data, before[n])       # fully reversible


def test_tensor_list_names_shapes_dtypes():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    ts = s.tensor_list()
    byname = {t["name"]: t for t in ts}
    assert f"model.layers.0.{CELL}" in byname
    t = byname[f"model.layers.0.{CELL}"]
    assert t["shape"] == [32, 16] and t["dtype"] == "float32"   # tiny llama gate_proj
    assert len(ts) == len(list(s.model.named_parameters()))     # every param listed


def test_save_region_persists_to_disk_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="test/tiny")
    region = s.locate_cell(0, CELL)
    s.save_region("r1", region)
    assert (tmp_path / "regions" / "test__tiny" / "r1.pt").exists()   # on disk
    s2 = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="test/tiny")
    name = f"model.layers.0.{CELL}"
    assert "r1" in s2._region_files and not s2.regions          # lazy: file known, mask not yet loaded
    assert torch.equal(s2.get_region("r1")[name], region[name])  # survives restart, loads on demand


def test_no_model_id_means_no_disk_io(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))  # tests/anonymous sessions
    s.save_region("r1", s.locate_cell(0, CELL))
    assert "r1" in s.regions and not (tmp_path / "regions").exists()  # memory only


def test_locate_spot_can_return_importance_grid():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    region, grid = s.locate_spot(["alpha"], topk=0.5, return_grid=True)
    assert len(region) > 0
    assert len(grid) == 2 and all(v >= 0 for row in grid for v in row)   # [L][M] |grad×param| cell grid


def test_saved_spot_region_keeps_importance_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")
    region, grid = s.locate_spot(["alpha"], topk=0.5, return_grid=True)
    s.save_region("r", region, grid=grid)
    assert s.region_grid("r")["importance"] == grid                       # viewer shows what the spot view showed
    s2 = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")
    assert s2.region_grid("r")["importance"] == grid                      # survives restart
    s.save_region("legacy", s.locate_cell(0, CELL))                       # no grid → fraction fallback
    assert s.region_grid("legacy")["importance"] is None


def test_delete_region_removes_memory_and_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")
    s.save_region("r", s.locate_cell(0, CELL))
    assert (tmp_path / "regions" / "t__t" / "r.pt").exists()
    s.delete_region("r")
    assert "r" not in s.regions and "r" not in s.region_grids
    assert not (tmp_path / "regions" / "t__t" / "r.pt").exists()
    s.delete_region("nope")  # unknown name is a no-op


def test_region_lazy_load_with_lru_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")
    for n in ("a", "b", "c"):
        s.save_region(n, s.locate_cell(0, CELL))
    s2 = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")  # fresh boot
    assert set(s2._region_files) == {"a", "b", "c"} and not s2.regions   # files scanned, masks not loaded
    s2.get_region("a"); s2.get_region("b")
    assert set(s2.regions) == {"a", "b"}                                 # loaded → cached
    s2.get_region("c")                                                   # third distinct → evict oldest ("a")
    assert set(s2.regions) == {"b", "c"}


def test_legacy_pt_migrates_to_meta_on_boot(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETIC_STUDIO_HOME", str(tmp_path))
    d = tmp_path / "regions" / "t__t"
    d.mkdir(parents=True)
    s0 = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))     # anonymous — just to build a mask
    region = s0.locate_cell(0, CELL)
    torch.save(region, d / "old.pt")                                     # v1 legacy: plain mask dict, no meta
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"), model_id="t/t")
    assert (d / "old.meta.json").exists()                                # migrated on boot
    count = sum(int(m.sum()) for m in region.values())
    assert s.region_meta() == [{"name": "old", "count": count}]


def test_region_compare_jaccard_and_intersection():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    s.save_region("a", s.locate_cell(0, CELL))
    s.save_region("b", s.locate_cell(0, CELL))          # identical → jaccard 1
    s.save_region("c", s.locate_cell(1, CELL))          # disjoint layer → jaccard 0
    cmp2 = s.region_compare(["a", "b"])
    assert cmp2["jaccard"]["a|b"] == 1.0
    gi = cmp2["modules"].index(CELL)
    assert cmp2["intersection"][0][gi] == 1.0            # whole cell in both
    cmp3 = s.region_compare(["a", "c"])
    assert cmp3["jaccard"]["a|c"] == 0.0
    assert all(v == 0.0 for row in cmp3["intersection"] for v in row)
    assert set(cmp3["grids"]) == {"a", "c"}              # per-region grids included


def test_region_grid_shows_selection_fraction():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    s.save_region("cell0", s.locate_cell(0, CELL))
    info = s.region_grid("cell0")
    gi = info["modules"].index(CELL.replace(".weight", "") + ".weight") if CELL in info["modules"] else info["modules"].index(CELL)
    assert info["grid"][0][gi] == 1.0                       # whole cell selected in layer 0
    assert all(v == 0.0 for v in info["grid"][1])           # nothing in layer 1
    assert info["count"] == sum(int(m.sum()) for m in s.get_region("cell0").values())


def test_save_region_and_intervene_by_it():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    s.save_region("mycell", s.locate_cell(0, CELL))
    assert "mycell" in s.regions
    name = f"model.layers.0.{CELL}"
    before = dict(s.model.named_parameters())[name].data.clone()
    s.intervene(s.get_region("mycell"), "zero", key="k")
    assert not torch.allclose(dict(s.model.named_parameters())[name].data, before)
    s.clear()


def test_multi_intervene_independent_keys():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    params = dict(s.model.named_parameters())
    nA, nB = "model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.gate_proj.weight"
    beforeA, beforeB = params[nA].data.clone(), params[nB].data.clone()
    s.intervene(s.locate_cell(0, CELL), "zero", key="a")
    s.intervene(s.locate_cell(1, CELL), "zero", key="b")     # both active at once
    assert not torch.allclose(params[nA].data, beforeA) and not torch.allclose(params[nB].data, beforeB)
    s.clear(key="a")                                          # remove one knob only
    assert torch.allclose(params[nA].data, beforeA)          # A restored
    assert not torch.allclose(params[nB].data, beforeB)      # B still intervened
    s.clear()                                                 # remove all
    assert torch.allclose(params[nB].data, beforeB)


def test_reintervene_same_key_reapplies_from_original():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    name = f"model.layers.0.{CELL}"
    before = dict(s.model.named_parameters())[name].data.clone()
    region = s.locate_cell(0, CELL)
    s.intervene(region, "zero", key="k")
    s.intervene(region, "scale", alpha=0.5, key="k")          # re-adjust same knob
    assert torch.allclose(dict(s.model.named_parameters())[name].data, before * 0.5)  # from original, not from zero
    s.clear()
    assert torch.allclose(dict(s.model.named_parameters())[name].data, before)


def test_suspend_restores_weights_but_keeps_knobs():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    name = f"model.layers.0.{CELL}"
    before = dict(s.model.named_parameters())[name].data.clone()
    s.intervene(s.locate_cell(0, CELL), "scale", alpha=0.5, key="k")
    s.suspend()
    assert torch.allclose(dict(s.model.named_parameters())[name].data, before)  # weights back to original
    assert "k" in s._active                                                     # knob entry kept
    s.resume()
    assert torch.allclose(dict(s.model.named_parameters())[name].data, before * 0.5)  # re-applied
    s.clear()
    assert torch.allclose(dict(s.model.named_parameters())[name].data, before)


def test_suspend_resume_multiple_knobs():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    params = dict(s.model.named_parameters())
    nA, nB = f"model.layers.0.{CELL}", f"model.layers.1.{CELL}"
    a0, b0 = params[nA].data.clone(), params[nB].data.clone()
    s.intervene(s.locate_cell(0, CELL), "zero", key="a")
    s.intervene(s.locate_cell(1, CELL), "scale", alpha=2.0, key="b")
    s.suspend()
    assert torch.allclose(params[nA].data, a0) and torch.allclose(params[nB].data, b0)
    s.resume()
    assert torch.allclose(params[nA].data, torch.zeros_like(a0))
    assert torch.allclose(params[nB].data, b0 * 2.0)
    s.clear()


def test_scale_alpha0_equals_zero():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    name = f"model.layers.0.{CELL}"
    region = s.locate_cell(0, CELL)
    s.intervene(region, "scale", alpha=0.0)
    scaled = dict(s.model.named_parameters())[name].data.clone()
    s.clear()
    s.intervene(region, "zero")
    zeroed = dict(s.model.named_parameters())[name].data
    assert torch.allclose(scaled, zeroed)
    s.clear()


def test_ppl_changes_under_intervention_then_restores():
    s = ModelSession(_tiny(), _EncTok(eos=-1), torch.device("cpu"))
    ex = ["alpha", "beta"]
    base = s.ppl(ex)
    s.intervene(s.locate_spot(ex, topk=0.5), "zero")
    assert not math.isclose(s.ppl(ex), base, rel_tol=1e-6)  # intervention has an effect
    s.clear()
    assert math.isclose(s.ppl(ex), base, rel_tol=1e-6)      # clear restored exactly


def test_generate_yields_square_causal_rows():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    rows = []
    for e in s.generate(PROMPT, max_tokens=3):  # PROMPT len 3
        rows += e.get("attn_rows", [])
    L = 2
    # prefill emits P=3 query rows (kv 1,2,3), then 2 decode rows (kv 4,5) → full causal square: row i kv=i+1
    assert [tuple(r.shape) for r in rows] == [(L, 1), (L, 2), (L, 3), (L, 4), (L, 5)]
    assert all(torch.allclose(r.sum(1), torch.ones(L), atol=1e-3) for r in rows)  # each row is a distribution


def test_drilldown_returns_per_head_of_last_step():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    rows = []
    for e in s.generate(PROMPT, max_tokens=3):
        rows += e.get("attn_rows", [])
    ph = s.drilldown(0)
    assert ph.shape == (2, 5)  # last query's per-head [heads, kv]
    assert torch.allclose(ph.mean(0), rows[-1][0], atol=1e-4)  # head-mean == last row, layer 0
