# Packaging — Parametic Studio (macOS + Windows)

The desktop app is a **UI shell** (Tauri v2 + React). The **kernel** (`parametic_studio/`,
torch inference) is *not* bundled in v1 — it runs in the user's own Python environment. The app
spawns it on launch and kills it on exit.

## Prerequisites (end user)

The kernel needs a working Python. On first run the app spawns
`python -m parametic_studio.api` from the repo checkout.

1. Python **3.10+**.
2. Repo checkout of `parametic-report` (contains `parametic_studio/`).
3. `pip install -r requirements-studio.txt` (torch: use the wheel for your machine — mps works
   with the default wheel on Apple Silicon; cuda needs the matching index).
4. `~/.parametic_studio/config.json`:
   ```json
   {
     "python_path": "/path/to/python",   // optional — omit to auto-try python3 / python
     "kernel_dir": "/path/to/parametic-report",
     "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct"
   }
   ```
   - `python_path` optional. Absent → the app tries `python3` then `python` (Windows: `python`
     then `python3`) from `PATH`.
   - `kernel_dir` optional in dev only (the app falls back to two levels up from its cwd, which
     is `studio_web/src-tauri` under `tauri dev`). **For a packaged build, set it.**
   - `HOME` locates `~`; on Windows `USERPROFILE` is used as a fallback.

Alternatively, point the frontend at a **remote kernel** (GPU box) instead of a local Python —
see [REMOTE_KERNEL.md](./REMOTE_KERNEL.md).

## Local build

```bash
cd studio_web
npm ci
npm run tauri build
```

- **macOS** → `src-tauri/target/release/bundle/dmg/*.dmg` + `.app`. The build is **unsigned**,
  so Gatekeeper blocks a double-click: **right-click → Open** the first time (or
  `xattr -dr com.apple.quarantine "Parametic Studio.app"`).
- **Windows** → `.msi` / `.exe` (nsis) under `bundle/msi` / `bundle/nsis`. Build on a Windows
  machine with the same command, or use CI (below).

## CI

`.github/workflows/studio-build.yml` builds both platforms (macOS aarch64 + Windows) via
`tauri-apps/tauri-action`. Trigger: manual `workflow_dispatch`, or push a `studio-v*` tag.
Bundles are uploaded as workflow artifacts. **Unsigned** — no notarization / Authenticode.

## Windows v1 caveats

- **No parent-watchdog.** The kernel's orphan guard (`_watch_parent` in `api.py`) is **posix
  only** — Windows keeps a process's ppid unchanged after the parent dies, so ppid-polling can't
  detect an orphan. Clean exit is still covered by the app's `RunEvent::Exit` kill. But an
  **abnormal** app exit (crash / force-kill) can leave an orphan kernel holding `:8000` — end it
  in **Task Manager** (kill the `python` process). Not implemented via ctypes/psutil in v1.
- **No `mps`.** The kernel auto-selects cuda (if available) or cpu. No Apple GPU on Windows.
- **Unverified.** v1 is validated on macOS dev only; the Windows path is untested end-to-end.

## Troubleshooting

- **App opens but says `kernel offline · reconnecting…`** — the packaged app couldn't spawn a
  kernel: `kernel_dir`/`python_path` missing or wrong in `~/.parametic_studio/config.json`
  (a Finder-launched app has cwd `/` and a minimal `PATH`, so the dev fallbacks don't apply).
  Fix the config and relaunch, or point Settings → kernel connection at a remote kernel.
- **Force-quit semantics** — a kernel the app *spawned* self-exits within ~2s of the app dying
  (posix parent-watch), even on force-quit. A kernel the app *attached* to (something was
  already on `:8000` at launch — e.g. your own dev kernel) is intentionally left running: the
  app never kills a kernel it didn't start.

## Follow-up

- Bundle Python + torch so no user-side Python setup is needed (v2).
- Code signing + notarization (macOS) and Authenticode (Windows).
- Windows orphan-kernel guard (job objects, or a psutil-based watchdog).
