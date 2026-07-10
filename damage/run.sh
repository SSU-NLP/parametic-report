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
render_template() {
    local template="$1"
    local language="$2"
    template="${template//\{model\}/$MODEL_NAME}"
    template="${template//\{language\}/$language}"
    template="${template//\{k\}/$k}"
    if [[ "$template" != /* ]]; then
        template="$REPO_ROOT/$template"
    fi
    printf '%s
' "$template"
}

MODEL_NAME="$(config_get region_selection.model_name)"
MODEL_PATH="$(config_get region_selection.original_model_path)"
WEIGHTS_TEMPLATE="$(config_get damage.weights_path_template)"
OUTPUT_TEMPLATE="$(config_get damage.output_path_template)"
INCLUDE_LANGUAGE="$(config_get damage.include_language)"
LANGUAGE_ARG="$(config_join data.languages " ")"
K_VALUES_ARG="$(config_join region_selection.k_values " ")"

read -r -a LANGUAGES <<< "$LANGUAGE_ARG"
read -r -a K_VALUES <<< "$K_VALUES_ARG"

for k in "${K_VALUES[@]}"; do
    if [[ "$INCLUDE_LANGUAGE" == "true" ]]; then
        for lang in "${LANGUAGES[@]}"; do
            WEIGHTS_FOLDER="$(render_template "$WEIGHTS_TEMPLATE" "$lang")"
            OUTPUT_DIR="$(render_template "$OUTPUT_TEMPLATE" "$lang")"
            "$PYTHON_BIN" "$SCRIPT_DIR/damage_model.py"                 --weights_folder "$WEIGHTS_FOLDER"                 --original_model "$MODEL_PATH"                 --output_dir "$OUTPUT_DIR"
        done
    else
        lang=""
        WEIGHTS_FOLDER="$(render_template "$WEIGHTS_TEMPLATE" "$lang")"
        OUTPUT_DIR="$(render_template "$OUTPUT_TEMPLATE" "$lang")"
        "$PYTHON_BIN" "$SCRIPT_DIR/damage_model.py"             --weights_folder "$WEIGHTS_FOLDER"             --original_model "$MODEL_PATH"             --output_dir "$OUTPUT_DIR"
    fi
done
