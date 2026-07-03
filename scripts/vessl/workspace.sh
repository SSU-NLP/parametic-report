#!/usr/bin/env bash
# EXPERIMENTAL: not yet live-verified. VESSL workspace subcommands and flags below
# were read from `vesslctl workspace create --help`, but no live workspace has been
# stood up from this script. Treat the command it prints as a starting point.
#
# Stand up a *persistent* Parametic Studio kernel on a VESSL workspace (a long-lived
# container, unlike the batch jobs submit.sh fires). The kernel's :8000 is exposed as
# an HTTP port, giving a public wss:// URL the frontend can point at.
#
#   scripts/vessl/workspace.sh --name studio [--gpus 1] [--token SECRET]
#
# Reuses account/volume/image constants from config.sh (single source of truth).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

NAME="studio"; GPUS=1; USE_CPU=0; TOKEN=""
# ⚠️ workspace create needs a --cluster slug (batch jobs infer it from the resource-spec,
#    but the workspace API asks explicitly). Find yours with `vesslctl cluster list` and
#    set VESSL_CLUSTER in .vesslrc / env. No safe default exists to hardcode here.
CLUSTER="${VESSL_CLUSTER:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    NAME="$2"; shift 2;;
    --gpus)    GPUS="$2"; shift 2;;
    --cpu)     USE_CPU=1; shift;;
    --cluster) CLUSTER="$2"; shift 2;;
    --token)   TOKEN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ "$USE_CPU" == "1" ]]; then SPEC="$VESSL_CPU_SPEC"; else SPEC="${VESSL_GPU_SPEC_PREFIX}${GPUS}"; fi
[[ -z "$CLUSTER" ]] && { echo "set VESSL_CLUSTER (see: vesslctl cluster list)" >&2; exit 1; }

# Startup script: sync code to fast disk, install runner deps, boot the kernel with the token.
# Same /shared -> /work code sync convention as submit.sh; push.sh must have run first.
TOKEN_LINE=""
[[ -n "$TOKEN" ]] && TOKEN_LINE="export PARAMETIC_STUDIO_TOKEN='${TOKEN}'"
INIT="$(cat <<EOF
set -e
mkdir -p '${VESSL_CODE_WORK}'
cp -ru '${VESSL_CODE_SHARED}/.' '${VESSL_CODE_WORK}/'
cd '${VESSL_CODE_WORK}'
pip install -q --break-system-packages -r requirements-studio.txt || true
${TOKEN_LINE}
nohup python -m parametic_studio.api > studio-kernel.log 2>&1 &
EOF
)"

echo "[workspace] name=$NAME spec=$SPEC image=$VESSL_IMAGE cluster=$CLUSTER"
[[ -n "$TOKEN" ]] && echo "[workspace] PARAMETIC_STUDIO_TOKEN will be injected (public URL → token required)"

# --port NAME:PORT:PROTOCOL exposes :8000 over HTTP → a public URL VESSL assigns.
vesslctl workspace create \
  --name "$NAME" \
  --cluster "$CLUSTER" \
  --resource-spec "$SPEC" \
  --image "$VESSL_IMAGE" \
  --cluster-volume "${VESSL_CLUSTER_VOL}:${VESSL_CLUSTER_MNT}" \
  --object-volume "${VESSL_OBJECT_VOL}:${VESSL_OBJECT_MNT}" \
  --port "studio:8000:http" \
  --init-script "$INIT"

echo
echo "[workspace] created. Find the exposed URL with:"
echo "    vesslctl workspace show $NAME"
echo "[workspace] point the frontend kernel URL at that host as wss://<host>/ws"
[[ -n "$TOKEN" ]] && echo "[workspace] and set the token field in Settings to: $TOKEN"
