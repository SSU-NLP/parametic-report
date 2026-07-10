#!/bin/bash
set -euo pipefail

# Must be run from any directory. Defaults come from ../../config.json.
# Example Usage: bash code_train_core-10000.sh "tiny-codes" "llama-3.2" "meta-llama/Llama-3.2-3B-Instruct" "go,java"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.json}"
CONFIG_GET="$REPO_ROOT/scripts/config_get.py"
if [[ -x "$REPO_ROOT/.venv/bin/deepspeed" ]]; then
    DEEPSPEED_BIN="$REPO_ROOT/.venv/bin/deepspeed"
else
    DEEPSPEED_BIN="deepspeed"
fi

config_get() { python "$CONFIG_GET" "$CONFIG_PATH" "$1"; }
config_join() { python "$CONFIG_GET" "$CONFIG_PATH" "$1" --join "$2"; }
config_path() { python "$CONFIG_GET" "$CONFIG_PATH" "$1" --path-root "$REPO_ROOT"; }

DATASET_NAME="${1:-$(config_get data.dataset_name)}"
TOKENIZER="${2:-$(basename "$(config_get data.tokenizer_path)")}"
MODEL="${3:-$(config_get training.model_name_or_path)}"
LANGUAGE_ARG="${4:-$(config_join data.languages " ")}"

GPU_IDS="$(config_join training.gpu_ids ",")"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
total_cards="${#GPU_ARRAY[@]}"

export PYTORCH_CUDA_ALLOC_CONF="$(config_get training.pytorch_cuda_alloc_conf)"

MAX_SEQ_LEN="$(config_get training.max_seq_len)"
LEARNING_RATE="$(config_get training.learning_rate)"
WEIGHT_DECAY="$(config_get training.weight_decay)"
NUM_TRAIN_EPOCHS="$(config_get training.num_train_epochs)"
PER_DEVICE_TRAIN_BATCH_SIZE="$(config_get training.per_device_train_batch_size)"
PER_DEVICE_EVAL_BATCH_SIZE="$(config_get training.per_device_eval_batch_size)"
GRADIENT_ACCUMULATION_STEPS="$(config_get training.gradient_accumulation_steps)"
LR_SCHEDULER_TYPE="$(config_get training.lr_scheduler_type)"
NUM_WARMUP_STEPS="$(config_get training.num_warmup_steps)"
ZERO_STAGE="$(config_get training.zero_stage)"
SEED="$(config_get training.seed)"
DATA_OUTPUT_PATH="$(config_path training.data_output_path)"
OUTPUT_ROOT="$(config_path training.output_root)"
LORA_DIM="$(config_get training.lora_dim)"
LORA_MODULE_NAME="$(config_get training.lora_module_name)"
SAVE_SAMPLES_ARG="$(config_join training.save_samples " ")"
read -r -a SAVE_SAMPLES <<< "$SAVE_SAMPLES_ARG"

OPTIONAL_ARGS=()
if [[ "$(config_get training.offload)" == "true" ]]; then
    OPTIONAL_ARGS+=(--offload)
fi
if [[ "$(config_get training.gradient_checkpointing)" == "true" ]]; then
    OPTIONAL_ARGS+=(--gradient_checkpointing)
fi
if [[ "$(config_get training.disable_dropout)" == "true" ]]; then
    OPTIONAL_ARGS+=(--disable_dropout)
fi
if [[ "$(config_get training.only_optimize_lora)" == "true" ]]; then
    OPTIONAL_ARGS+=(--only_optimize_lora)
fi
if [[ "$(config_get training.skip_eval)" == "true" ]]; then
    OPTIONAL_ARGS+=(--skip_eval)
fi
if [[ "$(config_get training.skip_final_model_save)" == "true" ]]; then
    OPTIONAL_ARGS+=(--skip_final_model_save)
fi

LANGUAGE_ARG="${LANGUAGE_ARG//,/ }"
read -r -a LANGUAGES <<< "$LANGUAGE_ARG"
MODEL_OUTPUT_NAME="${MODEL##*/}"
DATASET_ROOT="$REPO_ROOT/data_preprocess/dataset"

mkdir -p "$OUTPUT_ROOT"
cd "$SCRIPT_DIR"

for lang in "${LANGUAGES[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_OUTPUT_NAME/$lang"
    mkdir -p "$OUTPUT_DIR"
    echo "$OUTPUT_DIR"

    "$DEEPSPEED_BIN" "$SCRIPT_DIR/accumulate_grad_mul_params-10000.py"         --model_name_or_path "$MODEL"         --pretrain_train_data_path "$DATASET_ROOT/${DATASET_NAME}/preprocessed/${TOKENIZER}/train/$lang/train"         --pretrain_test_data_path "$DATASET_ROOT/${DATASET_NAME}/preprocessed/${TOKENIZER}/test/$lang/test"         --data_output_path "$DATA_OUTPUT_PATH"         --max_seq_len "$MAX_SEQ_LEN"         --learning_rate "$LEARNING_RATE"         --weight_decay "$WEIGHT_DECAY"         --num_train_epochs "$NUM_TRAIN_EPOCHS"         --total_cards "$total_cards"         --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"         --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"         --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"         --lr_scheduler_type "$LR_SCHEDULER_TYPE"         --num_warmup_steps "$NUM_WARMUP_STEPS"         --zero_stage "$ZERO_STAGE"         --seed "$SEED"         --lora_dim "$LORA_DIM"         --lora_module_name "$LORA_MODULE_NAME"         --save_samples "${SAVE_SAMPLES[@]}"         "${OPTIONAL_ARGS[@]}"         --deepspeed         --output_dir "$OUTPUT_DIR"         > "$OUTPUT_DIR/training_log.log" 2>&1
done
wait
