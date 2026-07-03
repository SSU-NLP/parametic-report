# Remote kernel

The frontend talks to the kernel over one WebSocket (`ws://localhost:8000/ws`). To run
the kernel on a GPU box instead of your laptop, pick one of three paths.

## (a) Point the frontend at a URL

If a kernel is already reachable, just change the kernel URL in **Settings** — the app
stores it in `localStorage` and reloads. Nothing else to run. Use this when someone else
already stood a kernel up (via path b or c).

## (b) SSH tunnel — `scripts/studio-remote.sh` (recommended)

```bash
scripts/studio-remote.sh user@gpu-host [remote-repo-dir]   # default dir: ~/parametic-report
```

One-time remote setup: clone the repo and `pip install -r requirements-studio.txt`.

Boots the kernel on the remote host (reuses one already listening on `:8000`), then holds
an `ssh -N -L 8000:localhost:8000` tunnel in the foreground. The frontend keeps using
`ws://localhost:8000/ws` unchanged. Ctrl-C closes the tunnel; the remote kernel is left
running.

**Security is SSH**, so no token is needed on this path. The tunnel is loopback-only.

## (c) VESSL workspace — public URL

For a persistent kernel with a public `wss://` URL (no SSH), use a VESSL **workspace**
(a long-lived container, unlike the batch jobs `submit.sh` fires):

```bash
bash scripts/vessl/push.sh                                  # push code first
scripts/vessl/workspace.sh --name studio --gpus 1 --token SECRET
vesslctl workspace show studio                              # find the exposed URL
```

The exposed URL is **public**, so a token is **required**:

1. Launch the workspace with `--token SECRET` (injects `PARAMETIC_STUDIO_TOKEN`).
2. In **Settings**, set the kernel URL to `wss://<exposed-host>/ws` and the **token**
   field to `SECRET`.

When the kernel has a token set, it rejects the socket unless the client's first frame is
`{"type":"auth","token":"SECRET"}` (close code `4401` on mismatch). The frontend sends
this automatically whenever the token field is non-empty. A token-less (localhost) kernel
has no gate and ignores stray auth frames, so the same client works against both.

> ⚠️ `scripts/vessl/workspace.sh` is **experimental / not yet live-verified**. Confirm the
> `vesslctl workspace` flags against your CLI version and set `VESSL_CLUSTER` first.

## Caveats (all remote paths)

- **`PARAMETIC_STUDIO_HOME`** (regions, datasets, runs) becomes the **remote** filesystem.
  Regions/datasets/runs you save live on the GPU box, not your laptop.
- **`link_path`** resolves against the **remote** filesystem — link paths that exist on the
  remote host, not local ones.
- **CUDA** is picked up automatically by `device.py` (auto). The **bf16 CUDA** path is
  **not yet verified** — validate generation/PPL on the target GPU before trusting numbers.
