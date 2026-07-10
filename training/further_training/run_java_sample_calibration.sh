#!/bin/bash
set -euo pipefail

# Run Java small-sample calibration accumulation without mutating config.json.
# Example:
#   bash training/further_training/run_java_sample_calibration.sh
# Optional env overrides:
#   CALIBRATION_SEEDS="1234 5678"
#   CALIBRATION_SAVE_SAMPLES="512 1024 2048"
#   CALIBRATION_OUTPUT_ROOT="training/sample_calibration_java"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.json}"
CONFIG_GET="$REPO_ROOT/scripts/config_get.py"
if [[ -n "${PARAMETIC_DEEPSPEED_BIN:-}" ]]; then
    DEEPSPEED_BIN="$PARAMETIC_DEEPSPEED_BIN"
elif [[ -n "${DEEPSPEED_BIN:-}" ]]; then
    DEEPSPEED_BIN="$DEEPSPEED_BIN"
elif [[ "${PARAMETIC_IGNORE_REPO_VENV:-0}" != "1" && -x "$REPO_ROOT/.venv/bin/deepspeed" ]]; then
    DEEPSPEED_BIN="$REPO_ROOT/.venv/bin/deepspeed"
else
    DEEPSPEED_BIN="deepspeed"
fi
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
TOKENIZER="${2:-$(basename "$(config_get data.tokenizer_path)")}"
MODEL="${3:-$(config_get training.model_name_or_path)}"
LANGUAGE="${4:-java}"

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
CONFIG_GRADIENT_ACCUMULATION_STEPS="$(config_get training.gradient_accumulation_steps)"
LR_SCHEDULER_TYPE="$(config_get training.lr_scheduler_type)"
NUM_WARMUP_STEPS="$(config_get training.num_warmup_steps)"
ZERO_STAGE="$(config_get training.zero_stage)"
DATA_OUTPUT_PATH="$(config_path training.data_output_path)"
LORA_DIM="$(config_get training.lora_dim)"
LORA_MODULE_NAME="$(config_get training.lora_module_name)"

CALIBRATION_SEEDS="${CALIBRATION_SEEDS:-1234 5678}"
CALIBRATION_SAVE_SAMPLES="${CALIBRATION_SAVE_SAMPLES:-512 1024 2048}"
CALIBRATION_OUTPUT_ROOT="${CALIBRATION_OUTPUT_ROOT:-$REPO_ROOT/training/sample_calibration_java}"
if [[ "$CALIBRATION_OUTPUT_ROOT" != /* ]]; then
    CALIBRATION_OUTPUT_ROOT="$REPO_ROOT/$CALIBRATION_OUTPUT_ROOT"
fi
CALIBRATION_EXPECTED_TENSORS="${CALIBRATION_EXPECTED_TENSORS:-254}"
read -r -a SEEDS <<< "$CALIBRATION_SEEDS"
read -r -a SAVE_SAMPLES <<< "$CALIBRATION_SAVE_SAMPLES"

GLOBAL_BATCH_SIZE=$(( total_cards * PER_DEVICE_TRAIN_BATCH_SIZE ))

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
OPTIONAL_ARGS+=(--skip_eval --skip_final_model_save)

MODEL_OUTPUT_NAME="${MODEL##*/}"
DATASET_ROOT="$REPO_ROOT/data_preprocess/dataset"
mkdir -p "$CALIBRATION_OUTPUT_ROOT"
cd "$SCRIPT_DIR"

for seed in "${SEEDS[@]}"; do
    OUTPUT_DIR="$CALIBRATION_OUTPUT_ROOT/$MODEL_OUTPUT_NAME/seed_$seed/$LANGUAGE"
    mkdir -p "$OUTPUT_DIR"
    echo "writing calibration checkpoints to $OUTPUT_DIR"

    for sample in "${SAVE_SAMPLES[@]}"; do
        CHECKPOINT_DIR="$OUTPUT_DIR/grad-mul-param_checkpoint_$sample"
        COMPLETED_COUNT=0
        if [[ -d "$CHECKPOINT_DIR" ]]; then
            COMPLETED_COUNT="$(find "$CHECKPOINT_DIR" -maxdepth 1 -name '*.pt' | wc -l)"
        fi
        if (( COMPLETED_COUNT >= CALIBRATION_EXPECTED_TENSORS )); then
            echo "skipping completed seed=$seed sample=$sample tensors=$COMPLETED_COUNT"
            continue
        fi

        SAVE_STEP=$(( (sample + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
        echo "running seed=$seed sample=$sample gradient_accumulation_steps=$SAVE_STEP"

        "$DEEPSPEED_BIN" "$SCRIPT_DIR/accumulate_grad_mul_params-10000.py" \
            --model_name_or_path "$MODEL" \
            --pretrain_train_data_path "$DATASET_ROOT/${DATASET_NAME}/preprocessed/${TOKENIZER}/train/$LANGUAGE/train" \
            --pretrain_test_data_path "$DATASET_ROOT/${DATASET_NAME}/preprocessed/${TOKENIZER}/test/$LANGUAGE/test" \
            --data_output_path "$DATA_OUTPUT_PATH" \
            --max_seq_len "$MAX_SEQ_LEN" \
            --learning_rate "$LEARNING_RATE" \
            --weight_decay "$WEIGHT_DECAY" \
            --num_train_epochs "$NUM_TRAIN_EPOCHS" \
            --total_cards "$total_cards" \
            --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
            --gradient_accumulation_steps "$SAVE_STEP" \
            --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
            --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
            --num_warmup_steps "$NUM_WARMUP_STEPS" \
            --zero_stage "$ZERO_STAGE" \
            --seed "$seed" \
            --lora_dim "$LORA_DIM" \
            --lora_module_name "$LORA_MODULE_NAME" \
            --save_samples "$sample" \
            "${OPTIONAL_ARGS[@]}" \
            --deepspeed \
            --output_dir "$OUTPUT_DIR" \
            > "$OUTPUT_DIR/training_log_sample_${sample}.log" 2>&1
    done
done

cat <<MSG
Calibration accumulation complete.

Next comparison command:
$PYTHON_BIN $REPO_ROOT/scripts/calibrate_sample_size.py \\
  --full-checkpoint $REPO_ROOT/training/further_training_java_full/$MODEL_OUTPUT_NAME/$LANGUAGE/grad-mul-param_checkpoint_10000 \\
  --approx-root $CALIBRATION_OUTPUT_ROOT/$MODEL_OUTPUT_NAME \\
  --language $LANGUAGE \\
  --seeds ${SEEDS[*]} \\
  --sample-sizes ${SAVE_SAMPLES[*]} \\
  --k-values $(config_join region_selection.k_values " ") \\
  --output-dir $REPO_ROOT/reports/java_sample_calibration \\
  --device auto \\
  --tie-mode stable
MSG
