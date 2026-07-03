import torch

from parametic_studio.device import resolve_device, dtype_for, pick_device


def test_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_indexed_cuda_falls_back_to_cpu_when_unavailable():
    if not torch.cuda.is_available():
        assert resolve_device("cuda:1") == torch.device("cpu")
    else:
        assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_pick_device_honors_explicit():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_matches_resolve_auto():
    # None/"auto" with no cuda (or a single card) → same as resolve_device("auto").
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 1):
        assert pick_device(None) == resolve_device("auto")
        assert pick_device("auto") == resolve_device("auto")


def test_cuda_unavailable_falls_back_to_cpu_not_exit():
    # script's resolve_device SystemExits here; studio must fall back gracefully.
    if not torch.cuda.is_available():
        assert resolve_device("cuda") == torch.device("cpu")


def test_auto_prefers_available_accelerator():
    dev = resolve_device("auto")
    if torch.backends.mps.is_available():
        assert dev == torch.device("mps")
    elif torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev == torch.device("cpu")


def test_dtype_per_device():
    assert dtype_for(torch.device("cuda")) == torch.bfloat16
    assert dtype_for(torch.device("mps")) == torch.bfloat16  # fp16 overflows → NaN/garbage on Qwen
    assert dtype_for(torch.device("cpu")) == torch.float32
