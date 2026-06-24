#!/usr/bin/env bash
# Phase B — paper reproduction (Kim et al. 2024 "Coding Spot") VESSL job driver.
#
# Reproduces the paper's MAIN experiment (distinct from the transplant/bridge family in
# scripts/transplant/): a multi-language summed-importance top-k% "code spot" is zeroed
# (damage) and the damaged checkpoint is scored on the ORIGINAL Python HumanEval + general
# tasks (paper Table 1). Single model per run (vs transplant's base+coder pair).
#
# Pipeline (each a self-contained VESSL job; run from repo root after `bash scripts/vessl/push.sh`):
#   calibrate  : full-10000 grad·param accumulation (reuses the VALIDATED platform runner;
#                MODE=full-10000 just raises the sample count of the same accumulation).
#                Scores -> PR_OBJ/scores/<model-id>/...
#   spot       : extract code-region (summed top-k% bool mask) + matched random/bottom controls   [TODO]
#   damage     : zero each mask -> saved damaged checkpoints                                       [TODO]
#   eval-he    : original Python HumanEval pass@1 on a damaged checkpoint                          [TODO]
#   eval-gen   : lm-eval-harness general tasks (gsm8k/hellaswag/mmlu/truthfulqa/winogrande)        [TODO]
#
# Reuses scripts/transplant/make_calib_spec.py (model-agnostic spec emitter) and the proven
# per-job isolation submit() pattern (copy pushed /shared code into a unique /work dir, run there).
#
# Env overrides:
#   MODEL_HF (default meta-llama/Llama-3.2-3B-Instruct), MODEL_ID, MODEL_OUT, MODEL_TOK,
#   EXPECTED (skip-safe tensor-count overestimate), AREA (java-code), MODE (full-10000),
#   K (code-region fraction, paper: 0.000025/0.0001/0.0009/0.0025), WATCH=1 (block on each job).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/vessl/config.sh"

# --- experiment config (single model; defaults = smallest paper model w/ a Table 1 row) ----
MODEL_HF="${MODEL_HF:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_ID="${MODEL_ID:-llama-3.2-3b-it}"
MODEL_OUT="${MODEL_OUT:-Llama-3.2-3B-Instruct}"   # = ${MODEL_HF##*/}; calibration scores land under this
MODEL_TOK="${MODEL_TOK:-$MODEL_HF}"
EXPECTED="${EXPECTED:-254}"                        # llama-3.2-3b = 28 layers; skip-safe overestimate
AREA="${AREA:-java-code}"                          # tiny-codes-java-full (1차 = java single)
MODE="${MODE:-full-10000}"                         # paper full 10000-example accumulation
K="${K:-0.0025}"                                   # code-region fraction (largest paper k = 0.25%)
SEED="${SEED:-1234}"                               # paper-style single seed for spot extraction
LANGS="${LANGS:-java}"                             # spot summed over these langs (1차=java; 2차=10 langs)
CAL_SEED="${CAL_SEED:-1234}"                        # single-seed calibration (mode 기본 2→1, 비용 절반)
# 10 paper languages (Python excluded); C#/C++ via filesystem-safe aliases (catalog AREAs)
PAPER_AREAS="java-code bash-code csharp-code cpp-code go-code javascript-code julia-code ruby-code rust-code typescript-code"

# DS_NAME/LANG/SAMPLE come from the catalog (single source of truth) — supports ANY area
# (java-code, bash-code, csharp-code, ...) with no per-area case block here.
resolve_area() {  # area -> sets globals DS_NAME, LANG, SAMPLE
  local area="$1" out
  out="$(cd "$ROOT" && python3 -c "
from parametic_platform import catalog
a=catalog.AREA_CATALOG['$area']; m=catalog.MODE_CATALOG['$MODE']
print(a.dataset_name, a.language, m.sample_size)" 2>/dev/null)"
  [ -z "$out" ] && { echo "unknown AREA=$area or MODE=$MODE" >&2; exit 1; }
  read -r DS_NAME LANG SAMPLE <<< "$out"
}
resolve_area "$AREA"

NS="$VESSL_NS"
PR_OBJ="$VESSL_OBJECT_MNT/$NS/paper-repro"     # /shared/<ns>/paper-repro (durable)
PR_WORK="$VESSL_CLUSTER_MNT/$NS/paper-repro"   # /work/<ns>/paper-repro (fast SSD)
SCORES="$PR_OBJ/scores"; DATA="$PR_OBJ/data"
HFCACHE="$VESSL_CLUSTER_MNT/$NS/hf-cache"
HF_TOKEN_VAL="$(grep '^HF_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"

PREAMBLE="export HF_HOME=$HFCACHE; export PARAMETIC_RUNNER_PYTHON=python; export PARAMETIC_IGNORE_REPO_VENV=1; export PYTHONUNBUFFERED=1; mkdir -p $HFCACHE"
SUBMIT="$ROOT/scripts/vessl/submit.sh"
WATCH_FLAG=(); [[ "${WATCH:-0}" == "1" ]] && WATCH_FLAG=(--watch)

active_job_exists() {  # name -> exit 0 if a non-terminal job with that name exists
  vesslctl job list 2>/dev/null | awk -v n="$1" '
    $2==n && ($3=="created"||$3=="running"||$3=="pending"||$3=="idle"||$3=="initializing"){f=1}
    END{exit !f}'
}

submit() {  # name, jobcode, cmd, cpu(0/1)   (per-job isolation: own /work code copy, --no-sync)
  local name="$1" jobcode="$2" cmd="$3" cpu="${4:-0}"
  if active_job_exists "$name"; then
    echo "[guard] active job '$name' already exists — refusing duplicate submit" >&2
    return 0
  fi
  local isolate="rm -rf '$jobcode'; mkdir -p '$jobcode'; cp -r '$VESSL_CODE_SHARED/.' '$jobcode/'; cd '$jobcode'"
  local env_args=()
  [[ -n "$HF_TOKEN_VAL" ]] && env_args=(--env "HF_TOKEN=$HF_TOKEN_VAL")
  # spot/mask jobs are top-k + S3-IO bound (GPU idle) → run on the cheap CPU spec.
  # NGPU>1: big models (8B, vocab 128k) OOM on 1 card → ZeRO-2 shards optimizer+grads across cards.
  local spec_flag=(--gpus "${NGPU:-1}"); [[ "$cpu" == "1" ]] && spec_flag=(--cpu)
  bash "$SUBMIT" --name "$name" "${spec_flag[@]}" --image "$VESSL_IMAGE" --no-sync \
    --pip "-r requirements-runner.txt" --tag parametic "${env_args[@]}" \
    "${WATCH_FLAG[@]}" --cmd "$isolate; $cmd"
}

uniq_jobcode() { echo "$PR_WORK/jobs/$1-$(date +%s%N)"; }

calibrate() {  # area(optional, default $AREA) -> single-seed grad·param accumulation via platform runner
  local area="${1:-$AREA}"
  resolve_area "$area"                              # sets DS_NAME, LANG, SAMPLE for this area's language
  # idempotent: skip if this model+lang+seed already has scores on the durable volume
  local ckdir="$NS/paper-repro/scores/$MODEL_ID/seed_$CAL_SEED/$LANG/grad-mul-param_checkpoint_$SAMPLE"
  if vesslctl volume ls "$VESSL_OBJECT_VOL" --prefix "$ckdir/" 2>/dev/null | grep -q '\.pt'; then
    echo "[cal] $MODEL_ID/$LANG scores already present (seed $CAL_SEED) — skip"; return 0
  fi
  local name="pr-cal-$MODEL_ID-$LANG"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  # per-language artifact/scratch dirs so concurrent language jobs never clobber each other
  local art="$PR_OBJ/cal/$MODEL_ID/$LANG/artifacts"
  local scr="$PR_WORK/cal/$MODEL_ID/$LANG/scratch"
  local spec_local; spec_local="$(mktemp /tmp/pr_spec_${MODEL_ID}_${LANG}_XXXX.json)"
  python3 "$ROOT/scripts/transplant/make_calib_spec.py" --model-id "$MODEL_ID" --hf-model "$MODEL_HF" \
    --tokenizer "$MODEL_TOK" --model-output-name "$MODEL_OUT" --expected-tensors "$EXPECTED" \
    --area "$area" --mode "$MODE" --k "$K" --seeds "$CAL_SEED" \
    --repo-root "$jobcode" --artifact-root "$art" --scratch-root "$scr" > "$spec_local"
  vesslctl volume upload "$VESSL_OBJECT_VOL" "$spec_local" \
    --remote-prefix "$NS/paper-repro/jobs/$MODEL_ID/$LANG" --overwrite
  local spec_remote="$PR_OBJ/jobs/$MODEL_ID/$LANG/$(basename "$spec_local")"

  local cmd="$PREAMBLE"
  # multi-GPU / memory knobs for big models: rewrite the job-local config.json before the runner.
  # NGPU>1 → gpu_ids=[0..N-1] so run_java_sample_calibration launches deepspeed across N cards
  # (ZeRO-2 shards optimizer+grads). GRAD_CKPT=1 → gradient_checkpointing (grads identical, less mem).
  local gck="False"; [[ "${GRAD_CKPT:-0}" == "1" ]] && gck="True"
  cmd="$cmd; python3 -c \"import json; p='config.json'; d=json.load(open(p)); t=d['training']; t['gpu_ids']=list(range(${NGPU:-1})); t['gradient_checkpointing']=bool(t.get('gradient_checkpointing')) or $gck; json.dump(d,open(p,'w'))\""
  # reuse pre-generated jsonl from the durable volume to skip per-job HF streaming (dataset_load),
  # which otherwise rate-limits (429) when many calibration jobs run concurrently. runner skips
  # dataset_load when train/test jsonl already exist in the job-local dataset dir.
  cmd="$cmd; D=$PR_OBJ/datasets/$DS_NAME; if [ -d \"\$D\" ]; then mkdir -p data_preprocess/dataset/$DS_NAME; cp -r \"\$D/.\" data_preprocess/dataset/$DS_NAME/ && echo '[reuse] durable jsonl -> skip HF dataset_load'; fi"
  cmd="$cmd; python -m parametic_platform.runner --job-spec $spec_remote"
  # scores land at scratch/calibration/<MODEL_OUT>/seed_<CAL_SEED>/<lang>/... ; copy the whole
  # subtree to the durable per-model scores dir (per-language subdirs never collide)
  cmd="$cmd; mkdir -p $SCORES/$MODEL_ID; cp -r $scr/calibration/$MODEL_OUT/. $SCORES/$MODEL_ID/"
  echo "[cal] $MODEL_ID/$LANG ($area/$MODE seed=$CAL_SEED k=$K) scores -> $SCORES/$MODEL_ID"
  submit "$name" "$jobcode" "$cmd"
}

spot() {  # from SAVED scores: code-region (per-tensor top-k%, abs-summed over LANGS) + matched bottom/random controls
  # IO: stage scores off the slow S3 object volume (/shared) onto fast job-local /work SSD ONCE,
  # then build masks for every k in KS reusing that local copy (S3 read once, not per-k/per-tensor).
  local kslist="${KS:-$K}"
  local name="pr-spot-$MODEL_ID"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local langs="${LANGS//,/ }"
  local W="$jobcode/spotwork"
  # read scores straight off /shared per-tensor (NO /work staging — 10-lang full scores are ~60GB
  # and blow the /work quota). create_approx_spot_masks reads each tensor ONCE and reuses it for
  # all k (multi-k --ks), so a single S3 pass covers every k. only the small bool masks hit disk.
  local ckpts=""
  for l in $langs; do ckpts="$ckpts $SCORES/$MODEL_ID/seed_$SEED/$l/grad-mul-param_checkpoint_$SAMPLE"; done
  local cmd="$PREAMBLE; pip install -q --break-system-packages fire >/dev/null 2>&1 || true; mkdir -p $W; cd $W"
  cmd="$cmd; python $jobcode/scripts/create_approx_spot_masks.py --checkpoints$ckpts"
  cmd="$cmd --code-output-root code-region/$MODEL_ID --control-output-root control/$MODEL_ID --ks $kslist --random-seeds 1 --device auto"
  for kk in $kslist; do
    local klabel="top${kk}"
    local D="$PR_OBJ/masks/$MODEL_ID/$klabel"
    cmd="$cmd; mkdir -p $D; cp -r code-region/$MODEL_ID/$klabel $D/code; cp -r control/$MODEL_ID/bottom/$klabel $D/bottom; cp -r control/$MODEL_ID/random_seed1/$klabel $D/random_seed1"
  done
  cmd="$cmd; echo SPOT_DONE"
  echo "[spot] $MODEL_ID k=($kslist) seed=$SEED langs=($langs) -> $PR_OBJ/masks/$MODEL_ID/top<k>/{code,bottom,random_seed1}"
  submit "$name" "$jobcode" "$cmd" 1   # CPU: top-k + S3-IO bound, GPU idle
}

eval_he() {  # label(code|bottom|random_seed1|original): (optional) damage from mask, then Python HumanEval pass@1
  local label="$1"
  local name="pr-evalhe-$MODEL_ID-k${K}-$label"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local klabel="top${K}"
  local maskarg="--mask-dir none"
  [ "$label" != "original" ] && maskarg="--mask-dir $PR_OBJ/masks/$MODEL_ID/$klabel/$label --k $K"
  # lm-eval `humaneval` (original openai_humaneval, 164) + chat template — instruct models actually
  # generate (bigcode raw completion left them empty). code execution via --confirm_run_unsafe_code.
  local setup="pip install -q --break-system-packages lm-eval accelerate fire python-dotenv >/dev/null 2>&1 || true; export HF_ALLOW_CODE_EVAL=1"
  # NO auto cap: humaneval_instruct has until-stops, and a collapsed `code` model emits an empty
  # ```` ``` ```` (~2 toks) and stops — so it isn't slow, and every condition shares the task's
  # default budget (1024) for fairness. (Verified: qwen-coder code@0.0025% = 0.0 even at 1024 toks,
  # generations were empty fences — genuine collapse, not truncation.) For milder damage on bigger
  # models (7B/8B) at small k this avoids artificially truncating real partial solutions.
  # GEN_MAXTOKS still lets you force a cap if ever needed.
  local genmax=""; [ -n "${GEN_MAXTOKS:-}" ] && genmax="--gen-max-toks ${GEN_MAXTOKS}"
  # LOG_SAMPLES=1 → persist per-problem generations to inspect generation patterns
  local logs=""; [ -n "${LOG_SAMPLES:-}" ] && logs="--log-samples"
  local cmd="$PREAMBLE; $setup; python scripts/eval_humaneval_python.py"
  cmd="$cmd --label $label --model-id $MODEL_HF $maskarg --apply-chat-template $genmax $logs"
  cmd="$cmd --limit ${HE_LIMIT:-0} --work-dir $jobcode/evalwork --output $PR_OBJ/results-he-chat/$MODEL_ID/$klabel/$label/metrics.json"
  echo "[eval-he] $MODEL_ID k=$K $label (lm-eval chat) -> $PR_OBJ/results-he-chat/$MODEL_ID/$klabel/$label/metrics.json"
  submit "$name" "$jobcode" "$cmd"
}

eval_gen() {  # label(code|bottom|random_seed1|original): (optional) damage, then lm-eval 5 general tasks
  local label="$1"
  local name="pr-evalgen-$MODEL_ID-k${K}-$label"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local klabel="top${K}"
  local maskarg="--mask-dir none"
  [ "$label" != "original" ] && maskarg="--mask-dir $PR_OBJ/masks/$MODEL_ID/$klabel/$label --k $K"
  # code-spot damage collapses the model → no EOS → gsm8k runs to max. Cap gen length for `code`
  # only (random/bottom/original emit EOS normally, keep default so their scores stay faithful).
  local genmax=""; [ "$label" = "code" ] && genmax="--gen-max-toks ${GEN_MAXTOKS:-64}"
  local setup="pip install -q --break-system-packages lm-eval fire python-dotenv >/dev/null 2>&1 || true"
  local cmd="$PREAMBLE; $setup; python scripts/eval_general_lmeval.py"
  cmd="$cmd --label $label --model-id $MODEL_HF $maskarg $genmax"
  cmd="$cmd --batch-size ${GEN_BATCH:-auto}${GEN_LIMIT:+ --limit $GEN_LIMIT}"
  cmd="$cmd --work-dir $jobcode/genwork --output $PR_OBJ/results-general/$MODEL_ID/$klabel/$label/metrics.json"
  echo "[eval-gen] $MODEL_ID k=$K $label -> $PR_OBJ/results-general/$MODEL_ID/$klabel/$label/metrics.json"
  submit "$name" "$jobcode" "$cmd"
}

dataset_prefetch() {  # CPU job: generate 10-language tiny-codes jsonl ONCE -> durable, so every
  # calibration job reuses it instead of streaming from HF (avoids 429 under concurrency).
  local name="pr-dataset-prefetch"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local langs="bash csharp cpp go java javascript julia ruby rust typescript"
  local lcsv; lcsv="$(echo $langs | tr ' ' ',')"
  local isolate="rm -rf '$jobcode'; mkdir -p '$jobcode'; cp -r '$VESSL_CODE_SHARED/.' '$jobcode/'; cd '$jobcode'"
  local cmd="$PREAMBLE; pip install -q --break-system-packages fire datasets python-dotenv >/dev/null 2>&1 || true"
  # single 1.63M scan extracts all 10 languages at once (one HF streaming pass)
  cmd="$cmd; python data_preprocess/create_code_dataset.py --hf_dataset_name=nampdn-ai/tiny-codes --output_dataset_name=tiny-codes-paper-tmp --total_examples=1630000 --train_ratio=0.8 --languages='$lcsv'"
  # redistribute each language jsonl to its per-language durable dataset dir (matches catalog dataset_name)
  cmd="$cmd; for l in $langs; do D=$PR_OBJ/datasets/tiny-codes-\$l-full; mkdir -p \$D/train \$D/test; cp data_preprocess/dataset/tiny-codes-paper-tmp/train/\$l.jsonl \$D/train/\$l.jsonl; cp data_preprocess/dataset/tiny-codes-paper-tmp/test/\$l.jsonl \$D/test/\$l.jsonl; echo \"\$l jsonl -> \$D\"; done; echo DATASET_PREFETCH_DONE"
  local env_args=(); [[ -n "$HF_TOKEN_VAL" ]] && env_args=(--env "HF_TOKEN=$HF_TOKEN_VAL")
  echo "[dataset-prefetch] CPU job -> $PR_OBJ/datasets/tiny-codes-<lang>-full (10 langs, 1 scan)"
  bash "$SUBMIT" --name "$name" --cpu --no-sync --tag parametic "${env_args[@]}" "${WATCH_FLAG[@]}" --cmd "$isolate; $cmd"
}

prefetch() {  # CPU job: warm /work hf-cache with benchmark datasets + verify eval parsing (tiny model)
  local name="pr-prefetch"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local tiny="sshleifer/tiny-gpt2"
  local isolate="rm -rf '$jobcode'; mkdir -p '$jobcode'; cp -r '$VESSL_CODE_SHARED/.' '$jobcode/'; cd '$jobcode'"
  local pip="pip install -q --break-system-packages transformers datasets accelerate lm-eval fire python-dotenv >/dev/null 2>&1 || true; export HF_ALLOW_CODE_EVAL=1"
  local cmd="$PREAMBLE; $pip"
  cmd="$cmd; python scripts/eval_humaneval_python.py --label prefetch --model-id $tiny --apply-chat-template --limit 2 --gen-max-toks 64 --work-dir $jobcode/he --output $PR_OBJ/prefetch/humaneval.json || true"
  cmd="$cmd; python scripts/eval_general_lmeval.py --label prefetch --model-id $tiny --limit 2 --batch-size 4 --work-dir $jobcode/gen --output $PR_OBJ/prefetch/general.json || true"
  cmd="$cmd; echo PREFETCH_DONE"
  local env_args=(); [[ -n "$HF_TOKEN_VAL" ]] && env_args=(--env "HF_TOKEN=$HF_TOKEN_VAL")
  echo "[prefetch] CPU job -> warms hf-cache + verifies eval parsing ($PR_OBJ/prefetch/)"
  bash "$SUBMIT" --name "$name" --cpu --image "$VESSL_IMAGE" --no-sync \
    --tag parametic "${env_args[@]}" "${WATCH_FLAG[@]}" --cmd "$isolate; $cmd"
}

case "${1:-}" in
  dataset-prefetch) dataset_prefetch;;
  prefetch)        prefetch;;
  calibrate)       calibrate;;
  calibrate-langs) for A in $PAPER_AREAS; do calibrate "$A"; done;;
  spot)        spot;;
  eval-he)     eval_he "${2:?label required (code|bottom|random_seed1|original)}";;
  eval-he-all) for l in original code random_seed1 bottom; do eval_he "$l"; done;;
  eval-gen)     eval_gen "${2:?label required (code|bottom|random_seed1|original)}";;
  eval-gen-all) for l in original code random_seed1 bottom; do eval_gen "$l"; done;;
  *) echo "usage: $0 {prefetch|calibrate|calibrate-langs|spot|eval-he <label>|eval-he-all|eval-gen <label>|eval-gen-all}" >&2; exit 1;;
esac
