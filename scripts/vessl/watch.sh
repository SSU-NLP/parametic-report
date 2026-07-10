#!/usr/bin/env bash
# Poll a VESSL job until it reaches a terminal state, then print the tail of its logs.
#
#   scripts/vessl/watch.sh <job-slug> [max_polls]
set -euo pipefail
SLUG="${1:?usage: watch.sh <job-slug> [max_polls]}"
MAX="${2:-180}"
st=""
for i in $(seq 1 "$MAX"); do
  st="$(vesslctl job show "$SLUG" 2>/dev/null | awk -F': *' '/^State:/{print $2}')"
  echo "[poll $i] state=$st"
  case "$st" in succeeded|failed|terminated) break;; esac
  sleep 10
done
echo "=== FINAL: $st ==="
echo "=== LOGS (tail) ==="
vesslctl job logs "$SLUG" 2>&1 | tail -120
