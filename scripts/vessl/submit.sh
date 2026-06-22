#!/usr/bin/env bash
# Submit a VESSL job with project conventions baked in: both volumes mounted,
# code synced from /shared -> /work (fast disk), then your command runs from the code dir.
#
#   scripts/vessl/submit.sh --name smoke --gpus 1 --cmd "python scripts/00_smoke_test.py"
#   scripts/vessl/submit.sh --name cpu-check --cpu --cmd "ls -la /work && nvidia-smi || true"
#   scripts/vessl/submit.sh --name train --gpus 1 --pip "peft transformers" --cmd "python scripts/20_train_consistency.py" --watch
#
# Flags:
#   --name NAME     job name (required)
#   --cmd  "CMD"    command to run inside the code dir (required)
#   --gpus N        A100 SXM count (1/2/4/8); default 1. Ignored if --cpu.
#   --cpu           use the CPU-only spec instead of GPU
#   --image IMG     container image (default: $VESSL_IMAGE)
#   --pip "PKGS"    pip install these before running (e.g. "peft transformers")
#   --no-sync       skip the /shared->/work code copy (use code already on /work)
#   --env KEY=VALUE  pass env var to the job (repeatable)
#   --tag TAG       job tag (default: project name from VESSL_NS)
#   --watch         poll until terminal and stream logs after submit
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

NAME=""; CMD=""; GPUS=1; USE_CPU=0; IMAGE="$VESSL_IMAGE"; PIP=""; SYNC=1; TAG="${VESSL_NS##*/}"; WATCH=0
ENV_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    NAME="$2"; shift 2;;
    --cmd)     CMD="$2"; shift 2;;
    --gpus)    GPUS="$2"; shift 2;;
    --cpu)     USE_CPU=1; shift;;
    --image)   IMAGE="$2"; shift 2;;
    --pip)     PIP="$2"; shift 2;;
    --no-sync) SYNC=0; shift;;
    --env)    ENV_ARGS+=( -e "$2" ); shift 2;;
    --tag)     TAG="$2"; shift 2;;
    --watch)   WATCH=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -z "$NAME" ]] && { echo "--name required" >&2; exit 1; }
[[ -z "$CMD"  ]] && { echo "--cmd required"  >&2; exit 1; }

if [[ "$USE_CPU" == "1" ]]; then SPEC="$VESSL_CPU_SPEC"; else SPEC="${VESSL_GPU_SPEC_PREFIX}${GPUS}"; fi

# In-container preamble: ensure namespace dirs, copy code to fast disk, cd, optional pip.
PRE="set -e; mkdir -p '${VESSL_CODE_WORK}' '${VESSL_DATA_SHARED}' '${VESSL_RESULTS_SHARED}'"
if [[ "$SYNC" == "1" ]]; then
  # Replace the code tree wholesale (it is tiny, ~12MB). An incremental `cp -ru` fails
  # with "File exists" when a prior job left files of a different type/owner on /work.
  PRE="${PRE}; echo '[harness] sync code ${VESSL_CODE_SHARED} -> ${VESSL_CODE_WORK}'; rm -rf '${VESSL_CODE_WORK:?}'; mkdir -p '${VESSL_CODE_WORK}'; cp -r '${VESSL_CODE_SHARED}/.' '${VESSL_CODE_WORK}/'"
fi
PRE="${PRE}; cd '${VESSL_CODE_WORK}'"
[[ -n "$PIP" ]] && PRE="${PRE}; echo '[harness] pip install ${PIP}'; pip install -q --break-system-packages ${PIP}"
FULL_CMD="${PRE}; echo '[harness] === user cmd ==='; ${CMD}"

echo "spec=$SPEC image=$IMAGE"
echo "cmd=$FULL_CMD"
OUT="$(vesslctl job create \
  -n "$NAME" \
  -r "$SPEC" \
  -i "$IMAGE" \
  --cluster-volume "${VESSL_CLUSTER_VOL}:${VESSL_CLUSTER_MNT}" \
  --object-volume "${VESSL_OBJECT_VOL}:${VESSL_OBJECT_MNT}" \
  "${ENV_ARGS[@]}" \
  --tag "$TAG" \
  --cmd "$FULL_CMD" 2>&1)"
echo "$OUT"

SLUG="$(echo "$OUT" | grep -oE 'job-[a-z0-9]+' | head -1)"
[[ -z "$SLUG" ]] && { echo "could not parse job slug" >&2; exit 1; }
echo "slug: $SLUG"
if [[ "$WATCH" == "1" ]]; then exec bash "$HERE/watch.sh" "$SLUG"; fi
