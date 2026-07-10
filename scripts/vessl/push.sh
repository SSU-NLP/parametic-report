#!/usr/bin/env bash
# Upload the current project's code to the object volume under its namespace.
# Project root is auto-detected (git toplevel), so this works from any project
# that has a .vesslrc. Code lands at <object-vol>:<VESSL_CODE_REMOTE_PREFIX>
# -> mounts at $VESSL_CODE_SHARED. Data/outputs are excluded (push data separately).
#
#   push.sh            # upload code
#   push.sh --dry-run  # preview, don't upload
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

command -v tar >/dev/null || { echo "tar required on the dev box" >&2; exit 1; }

# Generic excludes + per-project extras from .vesslrc ($VESSL_PUSH_EXCLUDES, space-separated).
EXCLUDES=( .git .env .claude .venv __pycache__ '*.pyc' .logs results .DS_Store )
read -r -a EXTRA <<< "${VESSL_PUSH_EXCLUDES:-}"
[[ ${#EXTRA[@]} -gt 0 ]] && EXCLUDES+=( "${EXTRA[@]}" )
TAR_EX=()
for e in "${EXCLUDES[@]}"; do TAR_EX+=( --exclude="$e" ); done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Stage a clean copy from the detected project root.
tar -C "$VESSL_PROJECT_ROOT" "${TAR_EX[@]}" -cf - . | tar -C "$STAGE" -xf -

echo "Project root: $VESSL_PROJECT_ROOT"
echo "Excludes: ${EXCLUDES[*]}"
echo "Staged top-level:"
( cd "$STAGE" && find . -maxdepth 2 -type d | sort | head -40 )

if [[ "$DRY" == "1" ]]; then
  echo "[dry-run] would upload -> ${VESSL_OBJECT_VOL}:${VESSL_CODE_REMOTE_PREFIX}"
  vesslctl volume upload "$VESSL_OBJECT_VOL" "$STAGE" --remote-prefix "$VESSL_CODE_REMOTE_PREFIX" --dry-run
  exit 0
fi

echo "Uploading code -> ${VESSL_OBJECT_VOL}:${VESSL_CODE_REMOTE_PREFIX} ..."
vesslctl volume upload "$VESSL_OBJECT_VOL" "$STAGE" --remote-prefix "$VESSL_CODE_REMOTE_PREFIX" --overwrite
echo "Done. In-container path after mount: ${VESSL_CODE_SHARED}"
