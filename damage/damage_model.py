import os

import fire
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def process_file(weight_path, model, filename):
    try:
        weights = torch.load(weight_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading weight file {weight_path}: {e}")
        return

    true_positions = torch.nonzero(weights.detach().float().cpu() > 0, as_tuple=True)
    param_name = filename.replace(".pt", "")

    for name, param in model.named_parameters():
        if name != param_name:
            continue
        if param.size() != weights.size():
            print(f"Size mismatch for {name}: expected {param.size()}, got {weights.size()}")
            return
        param.data[true_positions] = 0
        return

    print(f"No matching parameter found for {param_name}")


def process_weight_files(weights_folder: str, original_model: str, output_dir: str = "./damaged_models/"):
    if not os.path.exists(weights_folder):
        print(f"Error: Weights folder '{weights_folder}' not found.")
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(original_model, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(original_model, token=hf_token)
    except Exception as e:
        print(f"Error loading model or tokenizer '{original_model}': {e}")
        return

    for filename in os.listdir(weights_folder):
        if filename.endswith(".pt"):
            process_file(os.path.join(weights_folder, filename), model, filename)

    print("Saving modified model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modified model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(process_weight_files)
