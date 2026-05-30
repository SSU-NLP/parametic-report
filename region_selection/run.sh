#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.json}"
CONFIG_GET="$REPO_ROOT/scripts/config_get.py"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python"
fi

config_get() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1"; }
config_join() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1" --join "$2"; }
config_json() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1" --json; }
config_path() { "$PYTHON_BIN" "$CONFIG_GET" "$CONFIG_PATH" "$1" --path-root "$REPO_ROOT"; }

MODEL_NAME="$(config_get region_selection.model_name)"
MODEL_PATH="$(config_get region_selection.original_model_path)"
MODEL_OUTPUT_NAME="${MODEL_PATH##*/}"
INPUT_ROOT="$(config_path region_selection.input_root)"
REGION_OUTPUT_ROOT="$(config_path region_selection.output_root)"
LANGUAGE_LIST_JSON="$(config_json data.languages)"
K_VALUES_ARG="$(config_join region_selection.k_values " ")"
SAMPLE_LIST_JSON="$(config_json region_selection.sample_list)"

read -r -a K_VALUES <<< "$K_VALUES_ARG"

INPUT_DIR="$INPUT_ROOT/$MODEL_OUTPUT_NAME"
mkdir -p "$REGION_OUTPUT_ROOT"
cd "$REGION_OUTPUT_ROOT"

for k in "${K_VALUES[@]}"; do
    "$PYTHON_BIN" "$SCRIPT_DIR/extract_accumulated_core_linguistic_region.py"         --model_name="$MODEL_NAME"         --original_model_path="$MODEL_PATH"         --language_list="$LANGUAGE_LIST_JSON"         --sample_list="$SAMPLE_LIST_JSON"         --k="$k"         --input_dir="$INPUT_DIR"
    echo "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"
done
