# Handoff: Java Coding Spot Experiment

## Current Claim

As of 2026-05-30, the Java-derived top-1% parameter region can be described as a Java coding spot for `meta-llama/Llama-3.2-3B-Instruct`, with scope limits. The claim is supported by matched damage controls on Java PPL and a limited BigCode HumanEvalPack Java synthesize benchmark.

Safe wording: the Java full-data top-1% region is causally important for Java next-token prediction and Java code synthesis on the tested subset. Do not claim full paper reproduction, cross-language generalization, or non-code preservation yet.

## Repository State

- Branch: `benchmark/bigcode-eval`.
- BigCode Evaluation Harness clone: `/tmp/bigcode-evaluation-harness`.
- JDK inside container: `javac 21.0.11`.
- Hugging Face access is expected through `HF_TOKEN` in `.env`; do not commit `.env`.
- Main report: `reports/java_full_spot_report.md`.

## Completed Work

1. Created Java full dataset and tokenizer preprocessing for `tiny-codes-java-full`.
2. Accumulated `gradient * parameter` tensors for Java full training data at 10,000 examples.
3. Generated code-region masks for `top0.005`, `top0.01`, `top0.03`, and `top0.05`.
4. Generated matched control masks: `bottom`, `random_seed1`, `random_seed2`, `random_seed3`.
5. Damaged original model weights by zeroing selected mask positions for code and control regions.
6. Evaluated Java test-subset PPL/loss across all k values and controls.
7. Built visualization/report assets for module importance, model MRI-style maps, overview, and damage curves.
8. Installed and smoke-tested BigCode Evaluation Harness with HumanEvalPack Java synthesize.
9. Ran limited benchmark comparison on `humanevalsynthesize-java --limit 20` for original, code top0.01, random seed1 top0.01, and bottom top0.01.

## Key Results

Java PPL on the held-out subset:

| Model | PPL |
|---|---:|
| Original | 3.93 |
| Bottom top0.01 | 3.98 |
| Random top0.01 seed 1 | 4.07 |
| Random top0.01 seed 2 | 4.10 |
| Random top0.01 seed 3 | 4.11 |
| Code top0.01 | 115,096.54 |

BigCode HumanEvalPack Java synthesize, `--limit 20`, deterministic decoding:

| Model | pass@1 |
|---|---:|
| Original | 0.80 |
| Bottom top0.01 | 0.80 |
| Random top0.01 seed 1 | 0.75 |
| Code top0.01 | 0.00 |

Primary files:

- `reports/java_full_all_k_control_ppl.json`
- `reports/java_full_all_k_control_comparison/all_k_control_summary.csv`
- `reports/java_full_all_k_control_comparison/all_k_control_curves.png`
- `reports/bigcode_humanevalsynth_java_limit20_summary.csv`
- `reports/bigcode_humanevalsynth_java_limit20_pass1.png`
- `reports/java_full_model_mri_top0.01/`
- `reports/java_full_spot_report.md`

## Next Experiments

1. Run full `humanevalsynthesize-java` without `--limit` for the four top0.01 models: original, code, random_seed1, bottom.
2. Repeat benchmark evaluation for `top0.005`, `top0.03`, and `top0.05`.
3. Add more random seeds to benchmark evaluation, at least seeds 2 and 3 for `top0.01`.
4. Evaluate a non-code or general-language PPL set to check whether damage is code-specific.
5. Evaluate at least one other language from the paper setup to avoid overclaiming Java-only behavior.
6. Inspect generated benchmark outputs for failure mode: repeated tokens, syntax failures, runtime failures, or wrong answers.
7. If reporting externally, include confidence limits: one model, Java-only, HumanEvalPack subset unless the full run is complete.

## Full Benchmark Command Template

Run from repository root. Replace the model path and output label per condition.

```bash
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2-) \
.venv/bin/python /tmp/bigcode-evaluation-harness/main.py \
  --model damage/damaged_models/llama-3.2-3b/java-full/code/top0.01 \
  --tasks humanevalsynthesize-java \
  --prompt codellama \
  --max_length_generation 1536 \
  --temperature 0.0 \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --precision bf16 \
  --use_auth_token \
  --allow_code_execution \
  --save_generations \
  --save_generations_path reports/bigcode_humanevalsynth_java_code_top0.01_full_generations.json \
  --metric_output_path reports/bigcode_humanevalsynth_java_code_top0.01_full_metrics.json
```

Note: BigCode appends the task name to the generation output filename.

## Verification Checklist

- Confirm the starting model is the clean original model before each damage run.
- Confirm each damaged model has `config.json`, tokenizer files, and `model.safetensors`.
- Confirm masks have the same tensor count and selected-position count as expected for each k.
- Confirm `HF_TOKEN` is loaded for every dataset/model download path.
- Confirm benchmark commands use `--allow_code_execution` only in the intended environment.
- Confirm report figures use global normalization for claims and reserve per-tensor normalization for inspection only.

## Known Limits and Risks

- Full HumanEvalPack Java has not been completed yet.
- Current benchmark evidence is `--limit 20`, so report it as a subset result.
- `multiple-java` raw completion was unsuitable for this instruct model; `humanevalsynthesize-java --prompt codellama` produced meaningful smoke-test results.
- Sandbox `bwrap` errors may occur for local tooling; rerun necessary commands with approved escalation when they are blocked by namespace creation.
- Generated datasets, checkpoints, masks, and damaged models are large and should remain uncommitted.
