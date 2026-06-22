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
BRIDGE="$TX_OBJ/bridge"   # paper-spot (A∖B) "bridge" masks per model/scale
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

bridge_one() {  # tag(base/coder), hf — build paper-spot "bridge" mask: A(|weight| top-k) ∖ B(java grad top-k)
  local tag="$1" hf="$2"
  local CK="${CORE_K:-$K}"                         # core(B) size to exclude; defaults to k
  local blabel="k${K}"; [ "$CK" != "$K" ] && blabel="k${K}-c${CK}"
  local name="tx-bridge-$tag-$SAMPLE-$blabel"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local sc="$SCORES/$tag/seed_1234"               # java grad input_dir (single seed, paper-style)
  local W="$jobcode/bridgework"; local PS="$jobcode/scripts/paper_spot"
  local cmd="$PREAMBLE; pip install -q --break-system-packages fire >/dev/null 2>&1 || true; mkdir -p $W; cd $W"
  cmd="$cmd; python $PS/save_model_layer_weights.py --model_path $hf --output_dir weights --dtype float32"
  cmd="$cmd; python $PS/extract_accumulated_core_linguistic_region.py --model_name $tag --original_model_path $hf --language_list '[\"$LANG\"]' --sample_list '[$SAMPLE]' --k $CK --input_dir $sc"
  cmd="$cmd; python $PS/extract_spot.py --model_name $tag --original_model_path $hf --core_path code-region/$tag/top$CK --instruct_path weights --sample_list '[$SAMPLE]' --k $K --core_k $CK --input_dir $sc --code_or_lang code"
  cmd="$cmd; mkdir -p $BRIDGE/${tag}-${SAMPLE}-$blabel; cp -r code-spot/$tag/top$K/. $BRIDGE/${tag}-${SAMPLE}-$blabel/ && echo BRIDGE_DONE"
  echo "[bridge] $tag (sample $SAMPLE, k=$K core_k=$CK) -> $BRIDGE/${tag}-${SAMPLE}-$blabel"
  submit "$name" "$jobcode" "$cmd"
}

bridge_eval_one() {  # strategy — bridge transplant (paper A∖B masks) + cowork completion eval
  local s="$1"
  local CK="${CORE_K:-$K}"
  local blabel="k${K}"; [ "$CK" != "$K" ] && blabel="k${K}-c${CK}"
  local name="tx-bre-$s-$SAMPLE-$blabel"
  local jobcode; jobcode="$(uniq_jobcode "$name")"
  local setup="apt-get update -qq && apt-get install -y -qq default-jdk >/dev/null 2>&1 || true"
  setup="$setup; pip install -q --break-system-packages pyarrow >/dev/null 2>&1 || true"
  local cmd="$PREAMBLE; $setup; python scripts/transplant/eval_bridge_cowork.py"
  cmd="$cmd --strategy $s --base-model Qwen/Qwen2.5-1.5B --donor-model Qwen/Qwen2.5-Coder-1.5B"
  cmd="$cmd --bridge-base $BRIDGE/base-$SAMPLE-$blabel --bridge-coder $BRIDGE/coder-$SAMPLE-$blabel"
  cmd="$cmd --data $DATA_MULTIPL/test.parquet --result $TX_OBJ/results-bridge-$SAMPLE-$blabel/$s --work-dir $jobcode/brwork"
  cmd="$cmd --batch-size ${HE_BATCH:-32}"
  [ -n "${HE_LIMIT:-}" ] && [ "${HE_LIMIT}" != "0" ] && cmd="$cmd --limit $HE_LIMIT"
  echo "[bridge-eval] $s (sample $SAMPLE, k=$K core_k=$CK, batch=${HE_BATCH:-32}) -> results-bridge-$SAMPLE-$blabel/$s"
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
  bridge)    case "${2:?tag(base|coder) required}" in
               base)  bridge_one base "$BASE_HF";;
               coder) bridge_one coder "$CODER_HF";;
               *) echo "bridge tag must be base|coder" >&2; exit 1;; esac;;
  bridge-all) bridge_one base "$BASE_HF"; bridge_one coder "$CODER_HF";;
  bridge-eval)     bridge_eval_one "${2:?strategy required}";;
  bridge-eval-all) for s in base coder v1 v2 v3; do bridge_eval_one "$s"; done;;
  analyze-gen)  # generations.jsonl completion 쌍별 비교 (왜 동일 pass@1인지)
    aname="tx-analyze-gen-k${K}"; ajob="$(uniq_jobcode "$aname")"
    aroot="$TX_OBJ/results-bridge-$SAMPLE-k${K}"
    acmd="$PREAMBLE; python scripts/transplant/compare_generations.py $aroot ${ANALYZE_STRATS:-v2 rand ndlo vhi vlo perm ndhi}"
    submit "$aname" "$ajob" "$acmd";;
  analyze-status)  # 각 strategy summary.json의 채점 status 세분화
    sname="tx-analyze-status-k${K}"; sjob="$(uniq_jobcode "$sname")"
    sroot="$TX_OBJ/results-bridge-$SAMPLE-k${K}"
    scmd="$PREAMBLE; python scripts/transplant/status_breakdown.py $sroot ${ANALYZE_STRATS:-v2 rand ndlo vhi vlo perm ndhi}"
    submit "$sname" "$sjob" "$scmd";;
  analyze-mcnemar)  # base 대비 per-problem flip McNemar 검정
    nname="tx-mcnemar-k${K}"; njob="$(uniq_jobcode "$nname")"
    nroot="$TX_OBJ/results-bridge-$SAMPLE-k${K}"
    ncmd="$PREAMBLE; python scripts/transplant/mcnemar.py $nroot ${MCNEMAR_REF:-base} ${ANALYZE_STRATS:-coder v2 rand ndlo vhi vlo perm ndhi reverse}"
    submit "$nname" "$njob" "$ncmd";;
  *) echo "usage: $0 {cal-base|cal-coder|eval <s>|eval-all|eval-it|cowork <s>|cowork-all|bridge <base|coder>|bridge-all|bridge-eval <s>|bridge-eval-all|analyze-gen}" >&2; exit 1;;
esac
