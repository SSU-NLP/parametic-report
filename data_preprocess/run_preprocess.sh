#!/bin/bash
set -euo pipefail

# Must be run from any directory. Defaults come from ../../config.json.
# Example Usage: bash run_preprocess.sh "tiny-codes" "go,java" "data_preprocess/tokenizers/llama-3.2"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.json}"
CONFIG_GET="$REPO_ROOT/scripts/config_get.py"
if [[ -n "${PARAMETIC_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PARAMETIC_PYTHON_BIN"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
elif [[ "${PARAMETIC_IGNORE_REPO_VENV:-0}" != "1" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python"
fi

config_get() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1"; }
config_join() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1" --join "$2"; }
config_path() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1" --path-root "$REPO_ROOT"; }

DATASET_NAME="${1:-$(config_get data.dataset_name)}"
LANGUAGE_ARG="${2:-$(config_join data.languages " ")}"
TOKENIZER_PATH="${3:-$(config_get data.tokenizer_path)}"

# Local tokenizer directories are resolved relative to the repo. Hugging Face IDs
# such as Qwen/Qwen3-8B are passed through unchanged.
if [[ "$TOKENIZER_PATH" != /* && -e "$REPO_ROOT/$TOKENIZER_PATH" ]]; then
    TOKENIZER_PATH="$REPO_ROOT/$TOKENIZER_PATH"
fi

TOKENIZER_NAME="$(basename "$TOKENIZER_PATH")"
SEQ_LENGTH="$(config_get data.preprocess_seq_length)"
NUM_WORKERS="$(config_get data.preprocess_num_workers)"
DO_KEEP_NEWLINES="$(config_get data.do_keep_newlines)"
DO_SPLIT_FUNCTIONS="$(config_get data.do_split_functions)"

LANGUAGE_ARG="${LANGUAGE_ARG//,/ }"
read -r -a LANGUAGES <<< "$LANGUAGE_ARG"

OPTIONAL_ARGS=()
if [[ "$DO_KEEP_NEWLINES" == "true" ]]; then
    OPTIONAL_ARGS+=(--do_keep_newlines)
fi
if [[ "$DO_SPLIT_FUNCTIONS" == "true" ]]; then
    OPTIONAL_ARGS+=(--do_split_functions)
fi

DATASET_TYPES=(test train)

for type in "${DATASET_TYPES[@]}"; do
    for lang in "${LANGUAGES[@]}"; do
        INPUT_FILE_PATH="$SCRIPT_DIR/dataset/${DATASET_NAME}/${type}/${lang}.jsonl"
        OUTPUT_DIR="$SCRIPT_DIR/dataset/${DATASET_NAME}/preprocessed/${TOKENIZER_NAME}/${type}/$lang/"
        mkdir -p "$OUTPUT_DIR"

        "$PYTHON_BIN" "$SCRIPT_DIR/preprocess-llama.py"             --mode "write"             --file_path "$INPUT_FILE_PATH"             --save_prefix "$type"             --save_path "$OUTPUT_DIR"             --language "$lang"             --seq_length "$SEQ_LENGTH"             --tokenizer_path "$TOKENIZER_PATH"             --num_workers "$NUM_WORKERS"             "${OPTIONAL_ARGS[@]}"
    done
done
