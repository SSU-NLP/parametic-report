import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Create a new directory to store the layer weights
new_directory = 'CodeLlama-34b-Instruct-hf'
os.makedirs(new_directory, exist_ok=True)

# Load the model
model_id = "meta-llama/CodeLlama-34b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=hf_token)

# Iterate over each layer and save the weights
for i, layer in enumerate(model.model.layers):
    # Save the input_layernorm weightp
    torch.save(layer.input_layernorm.weight, os.path.join(new_directory, f'model.layers.{i}.input_layernorm.weight.pt'))

    # Save the mlp weights
    torch.save(layer.mlp.down_proj.weight, os.path.join(new_directory, f'model.layers.{i}.mlp.down_proj.weight.pt'))
    torch.save(layer.mlp.gate_proj.weight, os.path.join(new_directory, f'model.layers.{i}.mlp.gate_proj.weight.pt'))
    torch.save(layer.mlp.up_proj.weight, os.path.join(new_directory, f'model.layers.{i}.mlp.up_proj.weight.pt'))

    # Save the post_attention_layernorm weight
    torch.save(layer.post_attention_layernorm.weight, os.path.join(new_directory, f'model.layers.{i}.post_attention_layernorm.weight.pt'))

    # Save the self_attn weights
    torch.save(layer.self_attn.k_proj.weight, os.path.join(new_directory, f'model.layers.{i}.self_attn.k_proj.weight.pt'))
    torch.save(layer.self_attn.o_proj.weight, os.path.join(new_directory, f'model.layers.{i}.self_attn.o_proj.weight.pt'))
    torch.save(layer.self_attn.q_proj.weight, os.path.join(new_directory, f'model.layers.{i}.self_attn.q_proj.weight.pt'))
    torch.save(layer.self_attn.v_proj.weight, os.path.join(new_directory, f'model.layers.{i}.self_attn.v_proj.weight.pt'))