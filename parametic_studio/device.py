import torch

_DTYPE = {"cuda": torch.bfloat16, "mps": torch.bfloat16, "cpu": torch.float32}


def resolve_device(name="auto"):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")  # ponytail: graceful fallback; script's resolve_device SystemExits here
    return torch.device(name)


def dtype_for(device):
    return _DTYPE.get(torch.device(device).type, torch.float32)
