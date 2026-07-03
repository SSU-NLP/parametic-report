import copy

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from parametic_studio.kernel.lora import LinearLayer_LoRA
from parametic_studio.kernel.model_session import ModelSession

CELL = "mlp.gate_proj.weight"
EX = ["alpha beta", "gamma delta"]


class _Tok:
    eos_token_id = -1

    def encode(self, text):
        return [1, 2, 3, 4]

    def decode(self, ids):
        return f"<{ids[0]}>"


def _sess():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return ModelSession(LlamaForCausalLM(cfg).eval(), _Tok(), torch.device("cpu"))


def _run(s, **kw):
    return list(s.train_steps(EX, **kw))


def test_full_train_runs_and_loss_decreases():
    s = _sess()
    evs = _run(s, mode="full", steps=10, lr=1e-2)
    losses = [e["loss"] for e in evs]
    assert len(losses) == 10 and all(l == l for l in losses)  # finite (no NaN)
    assert losses[-1] < losses[0]                             # it actually learns


def test_spot_freeze_keeps_masked_weights():
    s = _sess()
    region = s.locate_cell(0, CELL)
    name = f"model.layers.0.{CELL}"
    before = dict(s.model.named_parameters())[name].data.clone()
    other = dict(s.model.named_parameters())[f"model.layers.1.{CELL}"].data.clone()
    _run(s, mode="spot-freeze", steps=5, lr=1e-2, region=region)
    params = dict(s.model.named_parameters())
    assert torch.allclose(params[name].data, before)                       # frozen region untouched
    assert not torch.allclose(params[f"model.layers.1.{CELL}"].data, other)  # rest trained


def test_spot_only_changes_only_masked():
    s = _sess()
    region = s.locate_spot(EX, topk=0.5)  # half of each layer param
    snap = {n: p.data.clone() for n, p in s.model.named_parameters()}
    _run(s, mode="spot-only", steps=5, lr=1e-2, region=region)
    for n, p in s.model.named_parameters():
        if n in region:
            m = region[n]
            assert torch.allclose(p.data[~m], snap[n][~m])          # outside mask untouched
            assert not torch.allclose(p.data[m], snap[n][m])        # inside mask trained
        else:
            assert torch.allclose(p.data, snap[n])                  # non-region params untouched


@pytest.mark.parametrize("mode", ["full", "spot-freeze", "spot-only"])
def test_reset_restores_exactly(mode):
    s = _sess()
    region = s.locate_cell(0, CELL) if mode != "full" else None
    snap = copy.deepcopy({n: p.data.clone() for n, p in s.model.named_parameters()})
    base_ppl = s.ppl(EX)
    _run(s, mode=mode, steps=5, lr=1e-2, region=region)
    s.reset_training()
    for n, p in s.model.named_parameters():
        assert torch.allclose(p.data, snap[n]), n
    assert abs(s.ppl(EX) - base_ppl) < 1e-4


def test_lora_base_frozen_output_changes_reset_exact():
    s = _sess()
    snap = {n: p.data.clone() for n, p in s.model.named_parameters()}
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        out0 = s.model(ids).logits.clone()
    _run(s, mode="lora", steps=10, lr=1e-2)
    for n, p in s.model.named_parameters():
        if "lora_" not in n:
            assert torch.allclose(p.data, snap[n]), n               # base frozen
    with torch.no_grad():
        assert not torch.allclose(s.model(ids).logits, out0)        # adapters took effect
    s.reset_training()
    assert not any(isinstance(m, LinearLayer_LoRA) for m in s.model.modules())  # back to nn.Linear
    with torch.no_grad():
        assert torch.allclose(s.model(ids).logits, out0)            # bit-exact undo


def test_train_clears_knobs_first():
    s = _sess()
    s.intervene(s.locate_cell(0, CELL), "zero", key="k")
    g = s.train_steps(EX, mode="full", steps=1, lr=1e-3)
    next(g)
    assert s._active == {}          # knobs auto-cleared before training
    list(g)
    s.reset_training()


def test_stop_removes_hooks_and_restores_grads():
    s = _sess()
    region = s.locate_cell(0, CELL)
    g = s.train_steps(EX, mode="spot-only", steps=100, lr=1e-3, region=region)
    next(g)
    s.stop()
    assert list(g) == []
    assert all(p.requires_grad for p in s.model.parameters())       # restored for locate_spot
    spot = s.locate_spot(EX, topk=0.5)                              # grads flow everywhere again
    assert len(spot) > 1
    s.reset_training()


def test_lora_params_never_enter_spot():
    s = _sess()
    _run(s, mode="lora", steps=1, lr=1e-3)
    assert all("lora_" not in m for m in s._modules())
    assert all("lora_" not in n for n in s.locate_spot(EX, topk=0.5))
    s.reset_training()


def test_retrain_without_reset_raises():
    s = _sess()
    _run(s, mode="full", steps=1, lr=1e-3)
    with pytest.raises(RuntimeError):
        next(s.train_steps(EX, mode="full", steps=1))
    s.reset_training()
    list(s.train_steps(EX, mode="full", steps=1))  # fine after reset
