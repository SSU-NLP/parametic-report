#!/usr/bin/env bash
# VESSL harness — shared config (single source of truth).
#
# Reusable across projects: put THIS directory on PATH (e.g. add the repo's
# scripts/vessl to PATH, or symlink push.sh/submit.sh/watch.sh into ~/.local/bin),
# then drop a `.vesslrc` at each project root with at least `VESSL_NS`.
#
# Account-level defaults below are for SSU-NLPLab and are env-overridable.
# Per-project settings come from `<project-root>/.vesslrc`.

# --- Account-level defaults (override via env if ever needed) ---
: "${VESSL_ORG:=SSU-NLPLab}"
: "${VESSL_TEAM:=Default}"
: "${VESSL_GPU_SPEC_PREFIX:=resourcespec-a100x}"    # + N (1/2/4/8): A100 SXM 80GB, $1.55/hr each, betelgeuse
: "${VESSL_CPU_SPEC:=resourcespec-a100cpu}"         # CPU-only, $0.30/hr
: "${VESSL_IMAGE:=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel}"
: "${VESSL_OBJECT_VOL:=objvol-gsvyr0eu87wt}"        # ssu-volume — slow HDD, S3, cross-cluster — data + results
: "${VESSL_CLUSTER_VOL:=clustervol-r922i766wr02}"   # test       — fast SSD, betelgeuse-local   — code / fast-access
: "${VESSL_OBJECT_MNT:=/shared}"
: "${VESSL_CLUSTER_MNT:=/work}"

# --- Per-project: detect project root, load its .vesslrc ---
: "${VESSL_PROJECT_ROOT:=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [[ -f "$VESSL_PROJECT_ROOT/.vesslrc" ]]; then
  # shellcheck disable=SC1091
  source "$VESSL_PROJECT_ROOT/.vesslrc"
fi

# VESSL_NS is required (set in .vesslrc or env), e.g. "seonghyeon/omni-cons".
if [[ -z "${VESSL_NS:-}" ]]; then
  echo "[vessl] VESSL_NS not set. Add '$VESSL_PROJECT_ROOT/.vesslrc' with:" >&2
  echo "          VESSL_NS=\"<user>/<project>\"" >&2
  echo "        or export VESSL_NS." >&2
  return 1 2>/dev/null || exit 1
fi

# --- Derived (namespace under BOTH shared volumes; never write to volume roots) ---
export VESSL_ORG VESSL_TEAM VESSL_GPU_SPEC_PREFIX VESSL_CPU_SPEC VESSL_IMAGE
export VESSL_OBJECT_VOL VESSL_CLUSTER_VOL VESSL_OBJECT_MNT VESSL_CLUSTER_MNT VESSL_NS VESSL_PROJECT_ROOT
export VESSL_CODE_REMOTE_PREFIX="${VESSL_NS}/code"   # S3 key prefix on object vol for uploaded code
export VESSL_DATA_REMOTE_PREFIX="${VESSL_NS}/data"
export VESSL_CODE_SHARED="${VESSL_OBJECT_MNT}/${VESSL_NS}/code"   # uploaded code lands here
export VESSL_CODE_WORK="${VESSL_CLUSTER_MNT}/${VESSL_NS}/code"    # fast copy; jobs run from here
export VESSL_DATA_SHARED="${VESSL_OBJECT_MNT}/${VESSL_NS}/data"
export VESSL_RESULTS_SHARED="${VESSL_OBJECT_MNT}/${VESSL_NS}/results"
