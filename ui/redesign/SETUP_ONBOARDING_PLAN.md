# SETUP_ONBOARDING_PLAN — first-run kernel setup (macOS check + one-click pip install)

> Execution plan for subagents. Scope is FIXED (approved design): if the local kernel doesn't
> connect in ~6s and we're not in remote/SSH mode, show a setup overlay — auto-detected Python
> list → pick/Browse → streamed `pip install -r requirements-studio.txt` → auto-respawn kernel →
> overlay closes on WS connect. Windows keeps the NSIS path (`installer-hooks.nsh`) untouched.
> Ponytail: reuse `spawn_kernel` / `load_config` / `studio_home` / dialog plugin / `tauriListen`
> patterns; no new abstractions beyond one Rust module and one overlay component.

Files touched (disjoint by task — safe to parallelize):
- T1: `studio_web/scripts/bundle-kernel.mjs`, `studio_web/src-tauri/tauri.conf.json`
- T2: `studio_web/src-tauri/src/setup.rs` (new), `studio_web/src-tauri/src/lib.rs`, `studio_web/src-tauri/Cargo.toml`
- T3: `studio_web/src/App.tsx`
- T4: integration verify (no new files)

---

## 0. INTERFACE CONTRACT (frozen — all tasks code against this, do not renegotiate)

### Tauri commands (Rust name → JS invoke; Tauri auto-converts JS camelCase args → Rust snake_case)

```rust
// setup.rs — all #[tauri::command]
async fn discover_pythons() -> Vec<PyInfo>                       // scan + probe all candidates
async fn probe_python(path: String) -> Option<PyInfo>            // one path (Browse…), source="custom"; None = not a usable python
async fn kernel_env_status() -> EnvStatus                        // current config python resolved + probed
fn install_kernel_deps(app: AppHandle, python_path: String) -> Result<(), String>
    // returns IMMEDIATELY after spawning the pip thread (Err = requirements file missing /
    // install already running / spawn failed). Progress + completion arrive as events below.
    // On pip exit 0: writes python_path into ~/.parametic_studio/config.json (preserving other keys).
fn respawn_kernel(app: AppHandle, python_path: Option<String>) -> Result<(), String>
    // if python_path is Some: write it to config.json first (preserve keys). Then kill the
    // app-spawned kernel child (if any), wait for :8000 to free (≤3s poll), spawn_kernel again,
    // store the new child in the managed Kernel state. Err string on failure.
```

```rust
#[derive(serde::Serialize, Clone)]
struct PyInfo {
    path: String,        // realpath (dedup key)
    version: String,     // "3.11.9"
    source: String,      // "config" | "conda (active)" | "conda" | "pyenv" | "pyenv shim" | "homebrew" | "system" | "custom"
    pip: bool,
    missing: Vec<String>,// subset of KERNEL_DEPS not importable
    ready: bool,         // pip && missing.is_empty()
}
#[derive(serde::Serialize)]
struct EnvStatus {
    python_path: Option<String>, // None = no usable python found at all
    deps_ok: bool,
    missing: Vec<String>,
}
const KERNEL_DEPS: [&str; 6] = ["torch", "transformers", "fastapi", "uvicorn", "websockets", "datasets"];
```

### Events (payload is a String, matching the existing `ssh-status` pattern + `tauriListen`'s `listen<string>`)

| event | payload | when |
|---|---|---|
| `deps-progress` | raw output line (plain string) | each stdout/stderr line from pip |
| `deps-done` | JSON string `{"ok":bool,"code":int}` (code = pip exit, -1 if unknown) | pip process exits |

### TS mirror types + JS call shapes (App.tsx)

```ts
type PyInfo = { path: string; version: string; source: string; pip: boolean; missing: string[]; ready: boolean }
type EnvStatus = { python_path: string | null; deps_ok: boolean; missing: string[] }

// invoke arg naming (camelCase → snake_case is automatic):
invoke('install_kernel_deps', { pythonPath: p })
invoke('respawn_kernel', { pythonPath: p })          // pythonPath may be omitted/null
invoke('probe_python', { path: p })
```

### Probe protocol (what T2 implements, what the unit test locks)

One `python -c` invocation per candidate (importlib.util.find_spec — fast, no real imports):

```python
import json,sys,importlib.util as u;m=["torch","transformers","fastapi","uvicorn","websockets","datasets"];print(json.dumps({"version":".".join(map(str,sys.version_info[:3])),"pip":u.find_spec("pip") is not None,"missing":[x for x in m if u.find_spec(x) is None]}))
```

Expected stdout (single line JSON): `{"version":"3.11.9","pip":true,"missing":["torch","datasets"]}`.
Non-zero exit / unparseable stdout / 10s timeout ⇒ candidate skipped (probe returns None).

### pip invocation (locked by unit test)

`<python> -m pip install --disable-pip-version-check -r <resources>/requirements-studio.txt`
(arg vector: `["-m","pip","install","--disable-pip-version-check","-r","<req>"]` — mirrors
installer-hooks.nsh deps; the requirements file IS the single source of truth, no inline dep list.)

---

## T1 — Bundle `requirements-studio.txt` as an app resource  [PARALLEL group A]

**Files:** `studio_web/scripts/bundle-kernel.mjs`, `studio_web/src-tauri/tauri.conf.json`

**Implement:**
1. `bundle-kernel.mjs`: after the kernel copy, also `cpSync(resolve(here,'..','..','requirements-studio.txt'), resolve(here,'..','src-tauri','resources','requirements-studio.txt'))`. Error-exit if the source file is missing (same style as the kernel-source check).
2. `tauri.conf.json` → `bundle.resources`: add `"resources/requirements-studio.txt": "requirements-studio.txt"` alongside the existing `parametic_studio` entry. (Resource then resolves at runtime as `app.path().resource_dir()/requirements-studio.txt` — same layout `bundled_kernel_dir()` relies on.)

**Depends on:** nothing.
**Verify:** `node studio_web/scripts/bundle-kernel.mjs` → prints both copies, `studio_web/src-tauri/resources/requirements-studio.txt` exists and equals repo root copy (`diff`). `tauri.conf.json` still valid JSON (`node -e "JSON.parse(require('fs').readFileSync('studio_web/src-tauri/tauri.conf.json'))"`).

---

## T2 — Rust setup module: discover / status / install / respawn  [PARALLEL group A]

**Files:** `studio_web/src-tauri/src/setup.rs` (new), `studio_web/src-tauri/src/lib.rs` (small edits), `studio_web/src-tauri/Cargo.toml`

**lib.rs edits (keep minimal):**
- `mod setup;` + register in `invoke_handler`: `setup::discover_pythons, setup::probe_python, setup::kernel_env_status, setup::install_kernel_deps, setup::respawn_kernel`.
- Make reusable helpers visible to setup.rs: `pub(crate) fn studio_home()`, `pub(crate) fn load_config()`, `pub(crate) fn spawn_kernel()`, and the `Kernel` struct fields (`pub(crate) struct Kernel(pub(crate) Mutex<Option<Child>>)`), plus `pub(crate) fn kernel_running()`. No logic changes to `spawn_kernel` itself.

**Cargo.toml:** add `"process"` to the tokio features (probe runs via `tokio::process::Command` + `tokio::time::timeout(10s)` inside the async commands; pip streaming uses std `Command` + threads — see below).

**setup.rs — implement:**

1. **Pure helpers (unit-tested, no process spawn):**
   - `fn parse_probe(path: &str, source: &str, stdout: &str) -> Option<PyInfo>` — parse the probe JSON line (tolerate trailing noise: scan for the last line that parses), compute `ready = pip && missing.is_empty()`.
   - `fn pip_install_args(req: &Path) -> Vec<String>` — exactly the locked vector from §0.
   - `fn candidate_paths(configured: Option<String>, home: &Path) -> Vec<(PathBuf, &'static str)>` — ordered, existence-unchecked list builder: config path (`"config"`), `$CONDA_PREFIX/bin/python` (`"conda (active)"` — read env inside caller, pass as param or read here), `<home>/miniconda3/bin/python` + `<home>/anaconda3/bin/python` (`"conda"`), glob `<home>/.pyenv/versions/*/bin/python` (`"pyenv"` — plain `read_dir`, no glob crate), `<home>/.pyenv/shims/python3` (`"pyenv shim"`), `/opt/homebrew/bin/python3` + `/usr/local/bin/python3` (`"homebrew"`), `/usr/bin/python3` (`"system"`).
2. **`async fn probe_one(path, source) -> Option<PyInfo>`** — skip if `!path.exists()`; run the §0 probe via `tokio::process::Command`, 10s timeout, then `parse_probe`. Set `PyInfo.path` to `fs::canonicalize` of the input (fallback: input as-is).
3. **`discover_pythons`** — build candidates (configured from `load_config().0`), probe each, dedup by canonical `path` keeping the FIRST occurrence (order above = priority, so "config" label wins). Return `Vec<PyInfo>` (ready ones need no sorting — frontend renders in order).
4. **`probe_python`** — `probe_one(path, "custom")`.
5. **`kernel_env_status`** — resolve: configured `python_path` if set, else try bare `python3` then `python` (PATH — `Command::new` resolves). First candidate that probes OK ⇒ `EnvStatus{python_path: Some(canonical), deps_ok: info.ready, missing: info.missing}`. None probe OK ⇒ `{python_path: None, deps_ok: false, missing: KERNEL_DEPS.to_vec()}`.
6. **`fn requirements_path(app: &AppHandle) -> Option<PathBuf>`** — `resource_dir()/requirements-studio.txt` if it exists, else dev fallback `current_dir().parent().parent()/requirements-studio.txt` (same two-levels-up dance as `spawn_kernel`'s dev fallback).
7. **`install_kernel_deps`** — guard with `static INSTALLING: AtomicBool` (`swap(true)` → if already true return `Err("install already running")`; clear on every exit path). Resolve `requirements_path` (Err if missing). Spawn `std::process::Command` with the locked args, `stdout(Stdio::piped()).stderr(Stdio::piped())`, `PYTHONUNBUFFERED=1`. Two reader threads (`BufReader::lines`) each `app.emit("deps-progress", line)` (clone `AppHandle`); a third (or the joining) thread `wait()`s, then: if exit ok → `write_config_python(&python_path)` → `app.emit("deps-done", format!(r#"{{"ok":true,"code":0}}"#))`; else `{"ok":false,"code":<code or -1>}`. `log::info!`/`log::warn!` at start/end. Command itself returns `Ok(())` right after thread spawn.
8. **`fn write_config_python(python_path: &str)`** — read `studio_home()/config.json` as `serde_json::Value` (default `{}` if absent/unparseable), set `["python_path"]`, `create_dir_all(studio_home())`, write pretty. Preserves `kernel_dir`/`model`/`hf_token`/`datasets_dir`/anything else. (Read raw JSON, NOT via `load_config()` — that tuple drops unknown keys.)
9. **`respawn_kernel`** — if `python_path` is Some non-empty: `write_config_python` first. Take the `Kernel` state: `kill()` + `wait()` the child if present. Then poll `!kernel_running()` up to 3s (100ms steps) so a dying spawned kernel can release :8000 (otherwise `spawn_kernel` would "attach" to a corpse). Then `*lock = spawn_kernel(&app)`; if the slot is still `None` AND `!kernel_running()` return `Err("kernel spawn failed — check python path")`, else `Ok(())` (None + port up = legit attach mode).

**Style:** match lib.rs — `log::` macros, `app.emit`, no `unwrap()` on I/O paths, comments explaining the one non-obvious thing (port-free poll, INSTALLING guard).

**Tests (in `setup.rs` `#[cfg(test)]`, ponytail — one per non-trivial unit):**
- `parse_probe` happy path: the §0 sample stdout → `PyInfo{version:"3.11.9", pip:true, missing:["torch","datasets"], ready:false}`; and garbage stdout → `None`; and `missing:[] & pip:true` → `ready:true`.
- `pip_install_args(Path::new("/x/requirements-studio.txt"))` == the exact locked vector.

**Depends on:** contract only (§0). T1 not required to compile (dev fallback covers `cargo test`).
**Verify:** `cd studio_web/src-tauri && cargo test && cargo check` — both clean.

---

## T3 — Frontend: setup overlay + 6s trigger + remote-stale banner  [PARALLEL group A]

**File:** `studio_web/src/App.tsx` only. Codes against §0 — do NOT wait for T2.

**Implement:**

1. **Wrapper:** add next to `tauriInvokeResult` (~line 18):
   ```ts
   async function tauriInvokeValue<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
     if (!inTauri()) throw new Error('not running in the desktop app')
     const { invoke } = await import('@tauri-apps/api/core')
     return await invoke<T>(cmd, args)
   }
   ```
   Add the two §0 TS types near the top-level type defs.

2. **State (near the kernel/ssh state block ~line 776):**
   `setupOpen: boolean`, `envStatus: EnvStatus | null`, `pythons: PyInfo[] | null` (null = scanning), `selectedPy: string` (path), `installing: boolean`, `installLog: string[]`, `installExit: {ok:boolean; code:number} | null`, `remoteStale: boolean`. Plus `kernelUpRef` mirroring `kernelUp` (same `ref.current =` idiom as `openRef`).

3. **6s trigger effect (mount-once):**
   - `if (!inTauri()) return` (browser: keep today's behavior untouched).
   - `setTimeout(6000)`: if `kernelUpRef.current === true` → nothing.
   - If `isRemoteConnected()` → `setRemoteStale(true)` (banner, NOT the overlay — don't fight the SSH flow).
   - Else → `const st = await tauriInvokeValue<EnvStatus>('kernel_env_status')`; if `!st.deps_ok` → `setEnvStatus(st); setSetupOpen(true); rescan()`. If `deps_ok` → do nothing (normal "reconnecting…" already covers a transient outage).
   - `rescan()` = `setPythons(null); tauriInvokeValue<PyInfo[]>('discover_pythons').then(setPythons)`; preselect the first `ready` entry, else first entry.

4. **Auto-close effect:** `useEffect(() => { if (kernelUp) { setSetupOpen(false); setRemoteStale(false); setInstalling(false) } }, [kernelUp])`.

5. **Event listeners (mount-once, same pattern as the `ssh-status` effect ~line 1093):**
   - `tauriListen('deps-progress', (line) => setInstallLog((l) => [...l.slice(-499), line]))` (cap 500 lines).
   - `tauriListen('deps-done', (payload) => { const r = JSON.parse(payload); setInstalling(false); setInstallExit(r); if (r.ok) { tauriInvokeResult('respawn_kernel').catch((e) => toast(\`[setup] ${e}\`)); resumeReconnect() } })`.
   - `resumeReconnect()` = `reconnectAttempts.current = 0; if (reconnectTimer.current) { clearTimeout(reconnectTimer.current); reconnectTimer.current = null } scheduleReconnect()` — kills any 30s backoff so the fresh kernel is picked up in ~1s.

6. **Actions:**
   - `startInstall()`: `setInstallLog([]); setInstallExit(null); setInstalling(true); tauriInvokeResult('install_kernel_deps', { pythonPath: selectedPy }).catch((e) => { setInstalling(false); toast(\`[setup] ${e}\`) })`.
   - `useThisPython()` (selected entry is `ready`): `tauriInvokeResult('respawn_kernel', { pythonPath: selectedPy })` then `resumeReconnect()`; errors → toast.
   - `browsePython()`: reuse the `plugin-dialog` dynamic-import pattern from `browseSshKeyPath` (no extension filter). On pick: `const info = await tauriInvokeValue<PyInfo | null>('probe_python', { path: picked })`; `info` → prepend to `pythons` (dedup by path) + select it; `null` → toast `[setup] not a usable python: <path>`.
   - `useLocalKernel()`: `localStorage.removeItem('ps_kernel_url'); localStorage.removeItem('ps_kernel_token'); sessionStorage.removeItem('ps_tunnel_live'); window.location.reload()`.

7. **Overlay UI** (render just before the toast stack, `position:fixed; inset:0; zIndex:60`, dimmed backdrop, centered panel ~520px styled like the settings panel — `var(--bg-1)`, `1px solid var(--line-strong)`, radius 14):
   - Header: **Kernel setup** + subtitle `local kernel isn't starting — Python dependencies are missing` (+ `missing: {envStatus.missing.join(', ')}` when non-empty).
   - Python list: `pythons === null` → `scanning…`; `[]` → `no Python found — install Python 3.10+ (e.g. brew install python) then rescan` + `[rescan]` Btn. Rows: radio-style click-to-select — `.mono` path, version, source label, then green `ready ✓` or `missing: torch, …` in `var(--text-2)`. `[Browse…]` Btn under the list (disabled while `installing`).
   - Primary action row: selected entry `ready` → `[Use this Python]`; else `[Install dependencies]` (`Btn color="var(--accent)"`, disabled when `!selectedPy || installing`). Hint line: `~2GB download (torch) — may take several minutes`.
   - Log box (rendered once `installLog.length || installing`): fixed height ~180px, `.mono` 11px, `overflow:auto`, auto-scroll to bottom (small effect on `installLog`).
   - `installExit && !installExit.ok` → error block: `pip failed (exit {code}) — see log above` + hint `externally-managed or offline? pick a different Python from the list` (the verbatim pip error is already in the log).
   - `installing === false && installExit?.ok` → status line `starting kernel…` (auto-close handles the rest).
   - Footer (small, `hint` style): `[retry]` (re-run `kernel_env_status` + `rescan`) · `[dismiss]` (`setSetupOpen(false)` — back to the status-bar "reconnecting…") · `[use local kernel]` only when `localStorage.getItem('ps_kernel_url')` is set.
   - Overlay must not unmount the rest of the app (render on top; reconnect loop keeps running behind it).

8. **Remote-stale banner:** when `remoteStale && kernelUp === false` render a small fixed banner (bottom-center, above status bar, zIndex 55): `remote kernel unreachable · [retry]` (→ `resumeReconnect(); setRemoteStale(false)`) `· [use local kernel]` (→ `useLocalKernel()`). Hidden once `kernelUp`.

**Don't touch:** `scheduleReconnect` internals, socket/auth logic, the SSH connect flow, browser (non-Tauri) behavior.

**Depends on:** contract §0 only.
**Verify:** `cd studio_web && npx tsc --noEmit -p .` → 0 errors. Grep-check: no new `window.prompt/confirm`, all new invokes behind `inTauri()`.

---

## T4 — Integration verify + manual scenario checklist  [SEQUENTIAL — after T1+T2+T3]

**Implement:** nothing new — fix whatever the checks below surface (interface drift between T2/T3 is the likely failure; §0 is the arbiter).

**Automated:**
1. `node studio_web/scripts/bundle-kernel.mjs` → both resources copied.
2. `cd studio_web/src-tauri && cargo test && cargo check`.
3. `cd studio_web && npx tsc --noEmit -p . && npm run build` (build runs bundle-kernel via beforeBuildCommand path only in `tauri build`; plain `npm run build` = vite, fine).
4. Optional smoke: `cd studio_web && npx tauri build --debug` (or `tauri dev`) compiles the full app.

**Manual (lead/user, macOS — record results in HANDOFF.md §2):**
- A. Point `~/.parametic_studio/config.json` `python_path` at a dep-less python (e.g. `/usr/bin/python3`) → launch app → within ~6-8s the setup overlay appears listing pyenv/conda/homebrew pythons with correct ready/missing labels.
- B. Select a `ready ✓` python → `[Use this Python]` → kernel comes up, overlay auto-closes, config.json now has that python_path and still has its other keys.
- C. Fresh venv python (no deps) → `[Install dependencies]` → pip lines stream into the log → `deps-done ok` → kernel respawns → overlay closes on WS connect.
- D. `/usr/bin/python3` (externally-managed) → install fails → verbatim pip error visible in log + "pick a different Python" hint; selecting another entry recovers.
- E. Set a bogus `ps_kernel_url` in localStorage → relaunch → after ~6s the remote-stale banner (not the overlay) appears → `[use local kernel]` clears keys + reloads to local.
- F. Regression: normal happy path (valid config) shows NO overlay; SSH connect flow unchanged; browser (vite, non-Tauri) unchanged.

---

## Dispatch order

```
group A (parallel):   T1 (bundling)   T2 (Rust setup.rs)   T3 (App.tsx overlay)
                         \                |                  /
sequential:                          T4 (integration verify)
```

T1/T2/T3 touch disjoint files and all code against the frozen contract in §0.
Suggested subagents: T2 → opus (largest, Rust + lifecycle care), T3 → opus/sonnet, T1 → sonnet (tiny), T4 → sonnet.
