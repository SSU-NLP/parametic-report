import os

import fire
import torch
from transformers import AutoModelForCausalLM


def save(model_path: str, output_dir: str, dtype: str = "float16", local_files_only: bool = False):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )

    for name, param in model.named_parameters():
        if "layers." not in name:
            continue
        torch.save(param.detach().cpu(), os.path.join(output_dir, f"{name}.pt"))


if __name__ == "__main__":
    fire.Fire(save)
