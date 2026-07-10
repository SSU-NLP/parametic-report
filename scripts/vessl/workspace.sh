#!/usr/bin/env bash
# Stand up a *persistent* Parametic Studio kernel on a VESSL workspace (a long-lived
# container, unlike the batch jobs submit.sh fires). The kernel's :8000 is exposed as
# an HTTP port, giving a public wss:// URL the frontend can point at.
#
#   bash scripts/vessl/push.sh                          # upload code first
#   scripts/vessl/workspace.sh --cpu --token SECRET     # cheap connectivity test ($0.30/hr)
#   scripts/vessl/workspace.sh --gpus 1 --token SECRET  # A100 ($1.55/hr)
#
# Cluster defaults to where config.sh's volumes live (betelgeuse); override with
# --cluster / VESSL_CLUSTER. Reuses account/volume/image constants from config.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

NAME="studio"; GPUS=1; USE_CPU=0; TOKEN=""; MODEL="Qwen/Qwen2.5-0.5B-Instruct"
CLUSTER="${VESSL_CLUSTER:-cluster-betelgeuse}"   # where config.sh's volumes live
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    NAME="$2"; shift 2;;
    --gpus)    GPUS="$2"; shift 2;;
    --cpu)     USE_CPU=1; shift;;
    --cluster) CLUSTER="$2"; shift 2;;
    --model)   MODEL="$2"; shift 2;;
    --token)   TOKEN="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ "$USE_CPU" == "1" ]]; then SPEC="$VESSL_CPU_SPEC"; else SPEC="${VESSL_GPU_SPEC_PREFIX}${GPUS}"; fi
[[ -z "$CLUSTER" ]] && { echo "set VESSL_CLUSTER (see: vesslctl cluster list)" >&2; exit 1; }

# Startup script: sync code to fast disk, install deps (torch already in the base image so
# requirements-studio.txt's torch>=2.3 is a no-op), boot the kernel bound to 0.0.0.0 so the
# exposed port can reach it. Same /shared -> /work code-sync convention as submit.sh.
# --break-system-packages: the pytorch base image ships a PEP-668 externally-managed python.
TOKEN_LINE=""
[[ -n "$TOKEN" ]] && TOKEN_LINE="export PARAMETIC_STUDIO_TOKEN='${TOKEN}'"
INIT="$(cat <<EOF
set -e
mkdir -p '${VESSL_CODE_WORK}'
cp -ru '${VESSL_CODE_SHARED}/.' '${VESSL_CODE_WORK}/'
cd '${VESSL_CODE_WORK}'
pip install -q --break-system-packages -r requirements-studio.txt
export PARAMETIC_STUDIO_HOST=0.0.0.0 PARAMETIC_STUDIO_PORT=8000
export PARAMETIC_STUDIO_MODEL='${MODEL}'
${TOKEN_LINE}
nohup python -m parametic_studio.api > /work/studio-kernel.log 2>&1 &
EOF
)"

echo "[workspace] name=$NAME spec=$SPEC image=$VESSL_IMAGE cluster=$CLUSTER model=$MODEL"
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
