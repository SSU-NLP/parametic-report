#!/usr/bin/env python3
import argparse
import ast
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tqdm import tqdm


JAVATUPLES_IMPORT = "import org.javatuples.*;\n"


def load_tasks(data_path: str, limit: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    print(f"[1/4] Loading benchmark: {data_path}", flush=True)
    table = pq.read_table(data_path)
    rows = table.to_pylist()
    tasks = []
    for row in rows:
        stop_tokens = row.get("stop_tokens") or []
        if isinstance(stop_tokens, str):
            try:
                stop_tokens = ast.literal_eval(stop_tokens)
            except Exception:
                stop_tokens = [stop_tokens]
        tasks.append(
            {
                "name": row["name"],
                "prompt": row["prompt"],
                "tests": row["tests"],
                "stop_tokens": stop_tokens,
            }
        )
        if limit is not None and len(tasks) >= limit:
            break
    print(f"[1/4] Loaded {len(tasks)} tasks", flush=True)
    return tasks


def strip_unavailable_imports(source: str) -> str:
    # The downloaded benchmark imports org.javatuples, but this workspace does not
    # ship the external jar. Most HumanEval-Java tasks do not use it.
    return source.replace(JAVATUPLES_IMPORT, "")


def truncate_at_stop(text: str, stop_tokens: list[str]) -> str:
    end = len(text)
    for token in stop_tokens:
        if not token:
            continue
        idx = text.find(token)
        if idx >= 0:
            end = min(end, idx)
    return text[:end]


def build_source(prompt: str, completion: str, tests: str) -> str:
    return strip_unavailable_imports(prompt + completion + tests)


def generate_completions(args: argparse.Namespace, tasks: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[2/4] Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    print(f"[2/4] Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    print(f"[2/4] Model ready on {device}", flush=True)

    results = []
    print(f"[3/4] Generating completions", flush=True)
    for task in tqdm(tasks, desc="generating", dynamic_ncols=True):
        encoded = tokenizer(task["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[-1] :]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion = truncate_at_stop(completion, task["stop_tokens"])
        source = build_source(task["prompt"], completion, task["tests"])
        results.append(
            {
                "name": task["name"],
                "prompt": task["prompt"],
                "completion": completion,
                "tests": task["tests"],
                "source": source,
            }
        )
    return results


def run_one_java(source: str, timeout: int) -> dict:
    if shutil.which("javac") is None or shutil.which("java") is None:
        return {
            "passed": False,
            "status": "missing_jdk",
            "compile_stdout": "",
            "compile_stderr": "javac/java not found in PATH",
            "run_stdout": "",
            "run_stderr": "",
            "elapsed_sec": 0.0,
        }

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "Problem.java"
        src.write_text(source, encoding="utf-8")

        try:
            compile_proc = subprocess.run(
                ["javac", "Problem.java"],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "status": "compile_timeout",
                "compile_stdout": exc.stdout or "",
                "compile_stderr": exc.stderr or "",
                "run_stdout": "",
                "run_stderr": "",
                "elapsed_sec": time.time() - start,
            }
        if compile_proc.returncode != 0:
            return {
                "passed": False,
                "status": "compile_error",
                "compile_stdout": compile_proc.stdout,
                "compile_stderr": compile_proc.stderr,
                "run_stdout": "",
                "run_stderr": "",
                "elapsed_sec": time.time() - start,
            }

        try:
            run_proc = subprocess.run(
                ["java", "-ea", "Problem"],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "status": "runtime_timeout",
                "compile_stdout": compile_proc.stdout,
                "compile_stderr": compile_proc.stderr,
                "run_stdout": exc.stdout or "",
                "run_stderr": exc.stderr or "",
                "elapsed_sec": time.time() - start,
            }
        return {
            "passed": run_proc.returncode == 0,
            "status": "passed" if run_proc.returncode == 0 else "runtime_error",
            "compile_stdout": compile_proc.stdout,
            "compile_stderr": compile_proc.stderr,
            "run_stdout": run_proc.stdout,
            "run_stderr": run_proc.stderr,
            "elapsed_sec": time.time() - start,
        }


def score_results(args: argparse.Namespace, rows: list[dict]) -> dict:
    scored = []
    print(f"[4/4] Scoring generated Java programs", flush=True)
    if shutil.which("javac") is None or shutil.which("java") is None:
        print("[4/4] WARNING: java/javac not found. Results will be marked missing_jdk.", flush=True)
    for row in tqdm(rows, desc="scoring", dynamic_ncols=True):
        outcome = run_one_java(row["source"], timeout=args.timeout)
        scored.append({**row, **outcome})

    passed = sum(1 for row in scored if row["passed"])
    total = len(scored)
    summary = {
        "total": total,
        "passed": passed,
        "pass_at_1": passed / total if total else 0.0,
        "missing_jdk": sum(1 for row in scored if row["status"] == "missing_jdk"),
        "compile_error": sum(1 for row in scored if row["status"] == "compile_error"),
        "compile_timeout": sum(1 for row in scored if row["status"] == "compile_timeout"),
        "runtime_error": sum(1 for row in scored if row["status"] == "runtime_error"),
        "runtime_timeout": sum(1 for row in scored if row["status"] == "runtime_timeout"),
    }

    output_dir = Path(args.result)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[4/4] Wrote results to {output_dir}", flush=True)
    return summary


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[3/4] Wrote generations to {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local HF models on HumanEval-Java.")
    parser.add_argument("--model", default="../models/Qwen2.5-1.5B")
    parser.add_argument("--data", default="../data/humaneval-java/test-00000-of-00001.parquet")
    parser.add_argument("--result", default="results/qwen2.5-1.5b-humaneval-java")
    parser.add_argument("--generations", default=None, help="Existing generations.jsonl to score.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.result)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Result directory: {output_dir}", flush=True)

    if args.score_only:
        if not args.generations:
            args.generations = str(output_dir / "generations.jsonl")
        print(f"[1/4] Loading existing generations: {args.generations}", flush=True)
        rows = read_jsonl(args.generations)
        print(f"[1/4] Loaded {len(rows)} generations", flush=True)
    else:
        tasks = load_tasks(args.data, limit=args.limit)
        rows = generate_completions(args, tasks)
        write_jsonl(str(output_dir / "generations.jsonl"), rows)

    if args.generate_only:
        print(f"Wrote generations to {output_dir / 'generations.jsonl'}")
        return

    summary = score_results(args, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



