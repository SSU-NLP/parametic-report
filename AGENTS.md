# Repository Guidelines

## Project Structure & Module Organization

This repository is a research pipeline for identifying coding-related regions in LLM parameters.

- `data_preprocess/`: dataset download, language splitting, and tokenizer-based preprocessing. Tokenizer assets live under `data_preprocess/tokenizers/`.
- `training/further_training/`: DeepSpeed training scripts that accumulate `gradient * parameter` tensors.
- `training/utils/`: shared model, data, DeepSpeed, LoRA, and utility code.
- `region_selection/`: scripts that convert accumulated tensors into top-k boolean parameter masks.
- `damage/`: scripts that zero selected model weights and save damaged Hugging Face models.

There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

Run commands from the directory noted in each script comment.

```bash
python data_preprocess/create_code_dataset.py
```
Downloads and splits the configured code dataset into language-specific JSONL files.

```bash
cd data_preprocess
bash run_preprocess.sh tiny-codes go tokenizers/llama-3.2
```
Tokenizes one language and writes `.bin`, `.idx`, and `.dis` files.

```bash
cd training/further_training
bash code_train_core-10000.sh tiny-codes llama-3.2 meta-llama/Llama-3.2-3B-Instruct go
```
Runs DeepSpeed accumulation and writes checkpoint tensors.

```bash
cd region_selection
python extract_accumulated_core_linguistic_region.py
python extract_spot.py
```
Generates top-k region masks.

```bash
cd damage
python damage_model.py --weights_folder <mask-dir> --original_model <hf-model> --output_dir <out-dir>
```
Zeros masked parameters and saves the modified model.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and snake_case for functions, variables, and script filenames. Keep CLI arguments explicit and compatible with `fire` or `argparse` patterns already used here. Prefer `os.path.join` for paths. Avoid adding hard-coded absolute paths; expose paths as arguments where practical.

## Testing Guidelines

No automated test framework is configured. For changes, run the smallest relevant pipeline step with a tiny sample or local fixture. At minimum, validate imports with:

```bash
python -m py_compile <changed-file.py>
```

For tensor-producing changes, verify expected output files exist and can be loaded with `torch.load`.

## Commit & Pull Request Guidelines

Recent history is minimal and mixed (`chore: update readme`, `Add files via upload`). Prefer concise, imperative commit messages; Conventional Commit prefixes such as `fix:`, `feat:`, `docs:`, and `chore:` are encouraged.

Pull requests should include the affected pipeline stage, commands run, expected output paths, required GPU/model/tokenizer assumptions, and any changes to dataset or Hugging Face model IDs.

## Security & Configuration Tips

Do not commit `.env`, Hugging Face tokens, downloaded datasets, model checkpoints, or generated `.pt` masks. Keep `HF_TOKEN` in `.env` or the environment, and keep runtime paths and hyperparameters in `config.json`.

## Current Experiment Snapshot

As of 2026-05-30, this workspace has a Java-only Coding Spot experiment on branch `benchmark/bigcode-eval`.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`.
- Dataset/preprocess target: `tiny-codes-java-full` with Llama 3.2 tokenizer assets.
- Java full `gradient * parameter` checkpoint: `training/further_training_java_full/Llama-3.2-3B-Instruct/java/grad-mul-param_checkpoint_10000`.
- Region masks: `region_selection_java_full/code-region/llama-3.2-3b/top0.005`, `top0.01`, `top0.03`, `top0.05`.
- Control masks: `region_selection_java_full/control-region/llama-3.2-3b/{bottom,random_seed1,random_seed2,random_seed3}/top<k>`.
- Damaged models: `damage/damaged_models/llama-3.2-3b/java-full/`.
- Main report: `reports/java_full_spot_report.md`.
- Detailed handoff and next experiments: `handoff.md`.

Key result: top-1% Java code-region damage collapses Java PPL and the limited BigCode `humanevalsynthesize-java --limit 20` benchmark, while matched random and bottom controls remain close to the original model.
