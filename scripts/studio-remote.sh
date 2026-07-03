#!/usr/bin/env bash
# One-command SSH path to a remote Parametic Studio kernel.
#
#   scripts/studio-remote.sh user@gpu-host [remote-repo-dir]
#
# Boots the kernel on the remote host (reuses one already listening on :8000),
# then holds an SSH tunnel so the local frontend talks to ws://localhost:8000/ws
# unchanged. Security is SSH — no PARAMETIC_STUDIO_TOKEN needed on this path.
set -euo pipefail

HOST="${1:?usage: studio-remote.sh user@gpu-host [remote-repo-dir]}"
REPO="${2:-~/parametic-report}"

echo "[studio-remote] ensuring kernel is up on $HOST ($REPO)"
# Start the kernel only if nothing is already listening on the remote :8000.
ssh "$HOST" bash -lc "'
  if curl -sf http://localhost:8000/openapi.json >/dev/null 2>&1 \
     || ss -ltn 2>/dev/null | grep -q :8000 \
     || netstat -ltn 2>/dev/null | grep -q :8000; then
    echo \"[remote] kernel already listening on :8000 — reusing\"
  else
    echo \"[remote] starting kernel\"
    cd $REPO
    nohup python -m parametic_studio.api > studio-kernel.log 2>&1 &
    sleep 2
    echo \"[remote] kernel launched (log: $REPO/studio-kernel.log)\"
  fi
'"

echo "[studio-remote] frontend URL stays ws://localhost:8000/ws"
echo "[studio-remote] tunnel up — Ctrl-C to close it (remote kernel keeps running)"

# Foreground -N tunnel; trap Ctrl-C so we exit cleanly without killing the remote kernel.
trap 'echo; echo "[studio-remote] tunnel closed — remote kernel left running on $HOST"; exit 0' INT
ssh -N -L 8000:localhost:8000 "$HOST"
