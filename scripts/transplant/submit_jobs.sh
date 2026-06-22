#!/usr/bin/env bash
# Transplant experiment — VESSL job driver.
#
# Two stages (see plan / handoff):
#   1. calibration: run the validated platform runner for base & coder, then copy each
#      model's grad·param scores (and base's java test bin) to a durable /shared dir.
#   2. eval: one self-contained job per condition — reconstruct spots, transplant,
#      save, score (java PPL + HumanEvalPack java pass@1).
#
# Usage (run from repo root, after `bash scripts/vessl/push.sh`):
#   scripts/transplant/submit_jobs.sh cal-base        # calibrate base (+ copy test bin)
#   scripts/transplant/submit_jobs.sh cal-coder       # calibrate coder
#   scripts/transplant/submit_jobs.sh eval <strategy> # one eval condition
#   scripts/transplant/submit_jobs.sh eval-all        # all eval conditions
#
# Robustness: there is no shared /work/code and no separate `sync` step. Each job copies
# the pushed /shared code into its OWN unique /work dir and runs there, so concurrent
# jobs never clobber one another. A same-named non-terminal job blocks a duplicate submit
# (guards against flaky-reconnect double-sends). Just `push.sh` then submit.
#
# Env overrides: AREA (java-code-smoke|java-code), MODE (approx-smoke|approx-1024|approx-2048),
#   K, HE_LIMIT (HumanEval --limit, 0=full), PPL_SAMPLES, WATCH=1 (block on each job).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/vessl/config.sh"

# --- experiment config -------------------------------------------------------------
BASE_ID="qwen2.5-1.5b";        BASE_HF="Qwen/Qwen2.5-1.5B";        BASE_MOUT="Qwen2.5-1.5B";        BASE_TOK="Qwen/Qwen2.5-1.5B"
CODER_ID="qwen2.5-coder-1.5b"; CODER_HF="Qwen/Qwen2.5-Coder-1.5B"; CODER_MOUT="Qwen2.5-Coder-1.5B"; CODER_TOK="Qwen/Qwen2.5-Coder-1.5B"
# instruct (it) references — sanity check that HumanEval works at all on this family
BASE_IT_HF="${BASE_IT_HF:-Qwen/Qwen2.5-1.5B-Instruct}"
CODER_IT_HF="${CODER_IT_HF:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

# IT=1 swaps the whole experiment to the instruct models (separate score tags + result
# suffix, so it never collides with the base-model run). HumanEval is an instruct task,
# so the base (non-instruct) models score as noise; instruct gives a real floor->ceiling.
SUFFIX=""; BASE_TAG="base"; CODER_TAG="coder"
if [[ "${IT:-0}" == "1" ]]; then
  BASE_HF="$BASE_IT_HF";  BASE_MOUT="Qwen2.5-1.5B-Instruct";        BASE_TOK="$BASE_IT_HF";  BASE_ID="qwen2.5-1.5b-it"
  CODER_HF="$CODER_IT_HF"; CODER_MOUT="Qwen2.5-Coder-1.5B-Instruct"; CODER_TOK="$CODER_IT_HF"; CODER_ID="qwen2.5-coder-1.5b-it"
  SUFFIX="-it"; BASE_TAG="base-it"; CODER_TAG="coder-it"
fi
EXPECTED=366                       # skip-safe overestimate (qwen2 1.5b has 336 layer tensors)
AREA="${AREA:-java-code-smoke}"
MODE="${MODE:-approx-1024}"
K="${K:-0.01}"
LANG="java"
SEEDS=(1234 5678)
STRATEGIES=(base coder v1 v2 v3a v3b v3c v3d v3ctrl)

# catalog-derived (kept in sync with parametic_platform/catalog.py) — for score paths
case "$AREA" in
  java-code-smoke) DS_NAME="tiny-codes-java-smoke";;
  java-code-mid)   DS_NAME="tiny-codes-java-mid";;
  java-code-xl)    DS_NAME="tiny-codes-java-xl";;
  java-code)       DS_NAME="tiny-codes-java-full";;
  *) echo "unknown AREA=$AREA" >&2; exit 1;;
esac
case "$MODE" in
  approx-smoke) SAMPLE=8;;
  approx-1024)  SAMPLE=1024;;
  approx-2048)  SAMPLE=2048;;
  full-10000)   SAMPLE=10000;;
  *) echo "unknown MODE=$MODE" >&2; exit 1;;
esac

NS="$VESSL_NS"
TX_OBJ="$VESSL_OBJECT_MNT/$NS/transplant"      # /shared/<ns>/transplant
TX_WORK="$VESSL_CLUSTER_MNT/$NS/transplant"    # /work/<ns>/transplant
SCORES="$TX_OBJ/scores"; DATA="$TX_OBJ/data"; RESULTS="$TX_OBJ/results"
RESULTS_CW="$TX_OBJ/results-cowork"; DATA_MULTIPL="$TX_OBJ/multipl-java"   # colleague repro (completion eval)
HFCACHE="$VESSL_CLUSTER_MNT/$NS/hf-cache"
HF_TOKEN_VAL="$(grep '^HF_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"

PREAMBLE="export HF_HOME=$HFCACHE; export PARAMETIC_RUNNER_PYTHON=python; export PARAMETIC_IGNORE_REPO_VENV=1; export PYTHONUNBUFFERED=1; mkdir -p $HFCACHE"
SUBMIT="$ROOT/scripts/vessl/submit.sh"
WATCH_FLAG=(); [[ "${WATCH:-0}" == "1" ]] && WATCH_FLAG=(--watch)

active_job_exists() {  # name -> exit 0 if a non-terminal job with that name exists
  vesslctl job list 2>/dev/null | awk -v n="$1" '
    $2==n && ($3=="running"||$3=="pending"||$3=="idle"||$3=="initializing"){f=1}
    END{exit !f}'
}

submit() {  # name, jobcode, cmd
  local name="$1" jobcode="$2" cmd="$3"
  # Idempotency guard: a flaky reconnect can resend the same Bash call, double-submitting
  # a job. Refuse if a same-named job is already non-terminal.
  if active_job_exists "$name"; then
    echo "[guard] active job '$name' already exists — refusing duplicate submit" >&2
    return 0
  fi
  # Per-job isolation: copy the pushed code from /shared into a UNIQUE /work dir and run
  # there (--no-sync so submit.sh does not touch the shared tree). No two jobs share a
  # code dir, so concurrent runs can never clobber each other — this removes the old
  # shared-/work/code sync race entirely, and a duplicate submit just gets its own dir.
  local isolate="rm -rf '$jobcode'; mkdir -p '$jobcode'; cp -r '$VESSL_CODE_SHARED/.' '$jobcode/'; cd '$jobcode'"
  local env_args=()
  [[ -n "$HF_TOKEN_VAL" ]] && env_args=(--env "HF_TOKEN=$HF_TOKEN_VAL")
  bash "$SUBMIT" --name "$name" --gpus 1 --image "$VESSL_IMAGE" --no-sync \
    --pip "-r requirements-runner.txt" --tag parametic "${env_args[@]}" \
    "${WATCH_FLAG[@]}" --cmd "$isolate; $cmd"
}

uniq_jobcode() { echo "$TX_WORK/txjobs/$1-$(date +%s%N)"; }

calibrate() {  # tag, model-id, hf, mout, tok, copy_test_bin(0/1)
  local tag="$1" mid="$2" hf="$3" mout="$4" tok="$5" copydata="$6"
  local name="tx-cal-$tag"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local art="$TX_OBJ/cal/$tag/artifacts"
  local scr="$TX_WORK/cal/$tag/scratch"
  local spec_local; spec_local="$(mktemp /tmp/tx_spec_${tag}_XXXX.json)"
  # runner's repo_root is this job's isolated code dir (dataset/preprocess land under it)
  python3 "$HERE/make_calib_spec.py" --model-id "$mid" --hf-model "$hf" --tokenizer "$tok" \
    --model-output-name "$mout" --expected-tensors "$EXPECTED" --area "$AREA" --mode "$MODE" --k "$K" \
    --repo-root "$jobcode" --artifact-root "$art" --scratch-root "$scr" > "$spec_local"
  vesslctl volume upload "$VESSL_OBJECT_VOL" "$spec_local" \
    --remote-prefix "$NS/transplant/jobs/$tag" --overwrite
  local spec_remote="$TX_OBJ/jobs/$tag/$(basename "$spec_local")"

  local cmd="$PREAMBLE; python -m parametic_platform.runner --job-spec $spec_remote"
  cmd="$cmd; mkdir -p $SCORES/$tag; cp -r $scr/calibration/$mout/. $SCORES/$tag/"
  if [[ "$copydata" == "1" ]]; then
    local tokname; tokname="$(basename "$tok")"
    local pp="$jobcode/data_preprocess/dataset/$DS_NAME/preprocessed/$tokname/test/$LANG"
    cmd="$cmd; mkdir -p $DATA; cp $pp/test.bin $DATA/test.bin; cp $pp/test.idx $DATA/test.idx"
  fi
  echo "[cal] $tag scores->$SCORES/$tag"
  submit "$name" "$jobcode" "$cmd"
}

eval_one() {  # strategy
  local s="$1"
  local name="tx-eval-$s$SUFFIX"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local bs="" ds=""
  for seed in "${SEEDS[@]}"; do
    bs="$bs $SCORES/$BASE_TAG/seed_$seed/$LANG/grad-mul-param_checkpoint_$SAMPLE"
    ds="$ds $SCORES/$CODER_TAG/seed_$seed/$LANG/grad-mul-param_checkpoint_$SAMPLE"
  done
  # BigCode harness setup (JDK for java compile + harness clone/install)
  local setup="apt-get update -qq && apt-get install -y -qq default-jdk git >/dev/null 2>&1 || true"
  setup="$setup; if [ ! -d /tmp/bigcode ]; then git clone --depth 1 https://github.com/bigcode-project/bigcode-evaluation-harness /tmp/bigcode; fi"
  setup="$setup; pip install -q --break-system-packages -e /tmp/bigcode || true; export HF_ALLOW_CODE_EVAL=1"

  local cmd="$PREAMBLE; $setup; python scripts/eval_humanevalpack_java.py"
  cmd="$cmd --strategy $s --base-model $BASE_HF --donor-model $CODER_HF --k $K --seed 0"
  cmd="$cmd --base-scores$bs --donor-scores$ds"
  cmd="$cmd --data-prefix $DATA/test --max-samples ${PPL_SAMPLES:-128}"
  cmd="$cmd --bigcode-dir /tmp/bigcode --he-prompt ${HE_PROMPT:-codellama} --he-limit ${HE_LIMIT:-0}"
  cmd="$cmd --work-dir $jobcode/evalwork --output $RESULTS/$s$SUFFIX/metrics.json"
  echo "[eval] $s$SUFFIX -> $RESULTS/$s$SUFFIX/metrics.json"
  submit "$name" "$jobcode" "$cmd"
}

eval_original() {  # label, hf_model — evaluate an unmodified model (instruct reference)
  local label="$1" hf="$2"
  local name="tx-eval-$label"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local setup="apt-get update -qq && apt-get install -y -qq default-jdk git >/dev/null 2>&1 || true"
  setup="$setup; if [ ! -d /tmp/bigcode ]; then git clone --depth 1 https://github.com/bigcode-project/bigcode-evaluation-harness /tmp/bigcode; fi"
  setup="$setup; pip install -q --break-system-packages -e /tmp/bigcode || true; export HF_ALLOW_CODE_EVAL=1"
  local cmd="$PREAMBLE; $setup; python scripts/eval_humanevalpack_java.py"
  cmd="$cmd --strategy base --base-model $hf --donor-model $hf --k $K --seed 0"
  cmd="$cmd --data-prefix $DATA/test --max-samples ${PPL_SAMPLES:-128}"
  cmd="$cmd --bigcode-dir /tmp/bigcode --he-prompt ${HE_PROMPT:-codellama} --he-limit ${HE_LIMIT:-0}"
  cmd="$cmd --work-dir $jobcode/evalwork --output $RESULTS/$label/metrics.json"
  echo "[eval-it] $label ($hf) -> $RESULTS/$label/metrics.json"
  submit "$name" "$jobcode" "$cmd"
}

cowork_one() {  # strategy — colleague repro: raw base model + completion eval on MultiPL-E java
  local s="$1"
  local name="tx-cw-$s"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local bs="" ds=""
  for seed in "${SEEDS[@]}"; do
    bs="$bs $SCORES/base/seed_$seed/$LANG/grad-mul-param_checkpoint_$SAMPLE"
    ds="$ds $SCORES/coder/seed_$seed/$LANG/grad-mul-param_checkpoint_$SAMPLE"
  done
  local setup="apt-get update -qq && apt-get install -y -qq default-jdk >/dev/null 2>&1 || true"
  setup="$setup; pip install -q --break-system-packages pyarrow >/dev/null 2>&1 || true"
  local cmd="$PREAMBLE; $setup; python scripts/transplant/eval_transplant_cowork.py"
  cmd="$cmd --strategy $s --base-model Qwen/Qwen2.5-1.5B --donor-model Qwen/Qwen2.5-Coder-1.5B --k $K"
  cmd="$cmd --base-scores$bs --donor-scores$ds"
  cmd="$cmd --data $DATA_MULTIPL/test.parquet --result $RESULTS_CW/$s --work-dir $jobcode/cwwork"
  [ -n "${HE_LIMIT:-}" ] && [ "${HE_LIMIT}" != "0" ] && cmd="$cmd --limit $HE_LIMIT"
  echo "[cowork] $s -> $RESULTS_CW/$s"
  submit "$name" "$jobcode" "$cmd"
}

case "${1:-}" in
  cal-base)  calibrate "$BASE_TAG"  "$BASE_ID"  "$BASE_HF"  "$BASE_MOUT"  "$BASE_TOK"  1;;
  cal-coder) calibrate "$CODER_TAG" "$CODER_ID" "$CODER_HF" "$CODER_MOUT" "$CODER_TOK" 0;;
  eval)      eval_one "${2:?strategy required}";;
  eval-all)  for s in "${STRATEGIES[@]}"; do eval_one "$s"; done;;
  eval-it)   eval_original base-it "$BASE_IT_HF"; eval_original coder-it "$CODER_IT_HF";;
  cowork)    cowork_one "${2:?strategy required}";;
  cowork-all) for s in "${STRATEGIES[@]}"; do cowork_one "$s"; done;;
  *) echo "usage: $0 {cal-base|cal-coder|eval <strategy>|eval-all|eval-it|cowork <s>|cowork-all}" >&2; exit 1;;
esac
