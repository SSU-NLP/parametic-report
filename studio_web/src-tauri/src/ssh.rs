//! SSH tunnel to a remote GPU kernel (STUDIO_PRODUCTION_PLAN P7).
//!
//! `ssh_connect` authenticates to the user's remote host, starts (or reuses) the studio kernel
//! bound to remote 127.0.0.1:8000, then opens a local port-forward: 127.0.0.1:8422 → (over the SSH
//! session, `direct-tcpip`) → remote 127.0.0.1:8000. The frontend then connects to
//! `ws://localhost:8422/ws`. The tunnel is owned by the Rust backend, not the webview, so a webview
//! reload never drops it. Security is the SSH channel itself — the remote kernel is bound to
//! loopback and never exposed on the remote network, so it needs no token.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use russh::client::{self, Handle};
use russh::keys::ssh_key;
use russh::{ChannelMsg, Disconnect};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const LOCAL_FORWARD_PORT: u16 = 8422;
const REMOTE_KERNEL_PORT: u32 = 8000;
const KERNEL_READY_TIMEOUT_SECS: u64 = 60;

/// A live tunnel: the SSH session handle plus the accept-loop task. Dropping/aborting the accept
/// loop closes the local listener; disconnecting the handle closes the SSH session and every
/// forwarded channel with it.
struct Conn {
    session: Arc<Handle<Client>>,
    accept_loop: JoinHandle<()>,
}

/// Managed Tauri state. `Mutex<Option<..>>` because at most one tunnel exists at a time; a fresh
/// `ssh_connect` tears down any existing one first.
#[derive(Default)]
pub struct SshState(Mutex<Option<Conn>>);

impl SshState {
    /// Abort the accept loop and disconnect the SSH session. Idempotent. The remote kernel is left
    /// running on purpose — the user owns it and other clients may still be attached.
    pub fn shutdown(&self) {
        if let Some(conn) = self.0.lock().unwrap().take() {
            conn.accept_loop.abort();
            let session = conn.session.clone();
            // best-effort graceful disconnect; the drop of `session` closes the socket regardless.
            tokio::spawn(async move {
                let _ = session
                    .disconnect(Disconnect::ByApplication, "client disconnect", "")
                    .await;
            });
        }
    }
}

/// russh client handler. Accepts any server key: this is a user-initiated tunnel to a host the user
/// typed themselves (like `ssh -o StrictHostKeyChecking=no`); the SSH transport still encrypts.
struct Client;

impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &ssh_key::PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

fn emit_status(app: &AppHandle, payload: &str) {
    let _ = app.emit("ssh-status", payload);
}

/// Single-quote a value for safe embedding in a remote `sh -c` command line.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Build the remote command that reuses the kernel if :8000 is already listening, else launches it
/// with `nohup`. `python` may carry candidates (`python3 || python`) chosen by the caller.
fn kernel_launch_command(
    repo_dir: &str,
    python: &str,
    model: &Option<String>,
) -> String {
    let model_env = match model {
        Some(m) => format!("PARAMETIC_STUDIO_MODEL={} ", sh_quote(m)),
        None => String::new(),
    };
    let check = format!(
        "curl -sf http://127.0.0.1:{p}/ >/dev/null 2>&1 || (command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':{p} ')",
        p = REMOTE_KERNEL_PORT
    );
    let launch = format!(
        "cd {dir} && PARAMETIC_STUDIO_HOST=127.0.0.1 PARAMETIC_STUDIO_PORT={port} {model}nohup {py} -m parametic_studio.api > /tmp/studio-kernel.log 2>&1 &",
        dir = sh_quote(repo_dir),
        port = REMOTE_KERNEL_PORT,
        model = model_env,
        py = python,
    );
    // if already listening, do nothing; otherwise launch in the background.
    format!("if {check}; then echo already-running; else {launch} echo started; fi")
}

/// Run one remote command over a fresh session channel; return its combined stdout as a String.
async fn remote_exec(session: &Handle<Client>, command: &str) -> Result<String, String> {
    let mut channel = session
        .channel_open_session()
        .await
        .map_err(|e| format!("open session channel: {e}"))?;
    channel
        .exec(true, command.as_bytes())
        .await
        .map_err(|e| format!("exec: {e}"))?;
    let mut out = Vec::new();
    while let Some(msg) = channel.wait().await {
        match msg {
            ChannelMsg::Data { data } => out.extend_from_slice(&data),
            ChannelMsg::ExtendedData { data, .. } => out.extend_from_slice(&data),
            ChannelMsg::Eof | ChannelMsg::Close | ChannelMsg::ExitStatus { .. } => {}
            _ => {}
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Poll the remote :8000 over SSH exec until it answers, or time out.
async fn wait_kernel_ready(session: &Handle<Client>) -> Result<(), String> {
    let probe = format!(
        "curl -sf http://127.0.0.1:{}/ >/dev/null 2>&1 && echo ok",
        REMOTE_KERNEL_PORT
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(KERNEL_READY_TIMEOUT_SECS);
    loop {
        if let Ok(out) = remote_exec(session, &probe).await {
            if out.contains("ok") {
                return Ok(());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "remote kernel did not become ready on :{} within {}s (see /tmp/studio-kernel.log on the host)",
                REMOTE_KERNEL_PORT, KERNEL_READY_TIMEOUT_SECS
            ));
        }
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
}

/// Accept loop for the local forward. Each inbound TCP connection gets its own `direct-tcpip`
/// channel to remote 127.0.0.1:8000, and raw bytes are copied both ways — no HTTP parsing, so the
/// WebSocket upgrade passes through untouched.
async fn run_forward(listener: TcpListener, session: Arc<Handle<Client>>) {
    loop {
        let (mut inbound, peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                log::warn!("ssh forward accept failed: {e}");
                continue;
            }
        };
        let session = session.clone();
        tokio::spawn(async move {
            let channel = match session
                .channel_open_direct_tcpip(
                    "127.0.0.1",
                    REMOTE_KERNEL_PORT,
                    &peer.ip().to_string(),
                    peer.port() as u32,
                )
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("ssh direct-tcpip open failed: {e}");
                    let _ = inbound.shutdown().await;
                    return;
                }
            };
            let mut stream = channel.into_stream();
            // raw bidirectional pipe; carries the WebSocket upgrade + frames verbatim.
            if let Err(e) = tokio::io::copy_bidirectional(&mut inbound, &mut stream).await {
                log::debug!("ssh forward copy ended: {e}");
            }
        });
    }
}

#[tauri::command]
pub async fn ssh_connect(
    app: AppHandle,
    host: String,
    port: u16,
    username: String,
    password: String,
    repo_dir: String,
    python_path: Option<String>,
    model: Option<String>,
) -> Result<(), String> {
    // tear down any existing tunnel first so a reconnect is clean.
    if let Some(state) = app.try_state::<SshState>() {
        state.shutdown();
    }

    emit_status(&app, r#"{"state":"connecting","detail":"authenticating"}"#);

    let config = Arc::new(client::Config {
        keepalive_interval: Some(Duration::from_secs(20)),
        ..Default::default()
    });
    let mut session = client::connect(config, (host.as_str(), port), Client)
        .await
        .map_err(|e| {
            let msg = format!("connect failed: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?;

    let auth = session
        .authenticate_password(&username, password)
        .await
        .map_err(|e| {
            let msg = format!("auth failed: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?;
    // password is consumed by authenticate_password (moved above) — never logged, never stored.
    if !auth.success() {
        let msg = "auth failed: password rejected".to_string();
        emit_status(&app, &error_json(&msg));
        return Err(msg);
    }

    let session = Arc::new(session);

    // start (or reuse) the remote kernel.
    emit_status(&app, r#"{"state":"starting-kernel"}"#);
    // caller-provided python, else try python3 then python via shell `||`.
    let python = python_path
        .clone()
        .unwrap_or_else(|| "$(command -v python3 || command -v python)".to_string());
    let launch = kernel_launch_command(&repo_dir, &python, &model);
    remote_exec(&session, &launch).await.map_err(|e| {
        let msg = format!("start kernel: {e}");
        emit_status(&app, &error_json(&msg));
        msg
    })?;
    wait_kernel_ready(&session).await.map_err(|e| {
        emit_status(&app, &error_json(&e));
        e
    })?;

    // open the local forward.
    emit_status(&app, r#"{"state":"forwarding"}"#);
    let listener = TcpListener::bind(("127.0.0.1", LOCAL_FORWARD_PORT))
        .await
        .map_err(|e| {
            let msg = format!("bind local :{LOCAL_FORWARD_PORT} failed: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?;

    let accept_loop = tokio::spawn(run_forward(listener, session.clone()));

    if let Some(state) = app.try_state::<SshState>() {
        *state.0.lock().unwrap() = Some(Conn {
            session,
            accept_loop,
        });
    }

    emit_status(&app, r#"{"state":"connected"}"#);
    Ok(())
}

#[tauri::command]
pub async fn ssh_disconnect(app: AppHandle) -> Result<(), String> {
    if let Some(state) = app.try_state::<SshState>() {
        state.shutdown();
    }
    emit_status(&app, r#"{"state":"disconnected"}"#);
    Ok(())
}

fn error_json(detail: &str) -> String {
    serde_json::json!({ "state": "error", "detail": detail }).to_string()
}
