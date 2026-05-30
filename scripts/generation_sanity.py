#!/usr/bin/env python3
import argparse
import gc
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    {
        "id": "java_factorial",
        "prompt": "Write a Java method `public static long factorial(int n)` that returns n factorial. Include only the method body and signature.",
    },
    {
        "id": "java_palindrome",
        "prompt": "Write a Java method `public static boolean isPalindrome(String s)` that returns true if s is a palindrome, ignoring case and non-alphanumeric characters.",
    },
    {
        "id": "java_two_sum",
        "prompt": "Write a Java method `public static int[] twoSum(int[] nums, int target)` that returns indices of two numbers adding to target.",
    },
    {
        "id": "java_parentheses",
        "prompt": "Write a Java method `public static boolean isValidParentheses(String s)` that validates parentheses, brackets, and braces using a stack.",
    },
    {
        "id": "java_binary_search",
        "prompt": "Write a Java method `public static int binarySearch(int[] arr, int target)` that returns the index of target or -1.",
    },
]


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model entries must use label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Model entries must use label=path")
    return label, path


def build_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": "You are a concise Java coding assistant. Return only Java code."},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {messages[0]['content']}\nUser: {user_prompt}\nAssistant:\n"


def generate_for_model(label, model_path, prompts, args, device, dtype, token):
    print(f"\n== generating {label} ==", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, torch_dtype=dtype)
    model.to(device)
    model.eval()

    rows = []
    for item in prompts:
        prompt_text = build_prompt(tokenizer, item["prompt"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append({
            "model": label,
            "prompt_id": item["id"],
            "prompt": item["prompt"],
            "generated": generated,
        })
        print(f"{label}/{item['id']} generated {len(generated_ids)} tokens", flush=True)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_markdown(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    lines = ["# Generation Sanity", ""]
    for prompt_id, prompt_rows in by_prompt.items():
        lines.extend([f"## {prompt_id}", "", prompt_rows[0]["prompt"], ""])
        for row in prompt_rows:
            lines.extend([f"### {row['model']}", "", "```java", row["generated"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side coding outputs for sanity checks.")
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    rows = []
    for label, model_path in args.model:
        rows.extend(generate_for_model(label, model_path, DEFAULT_PROMPTS, args, device, dtype, token))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
