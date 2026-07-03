use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{Emitter, Manager};

// The app owns the kernel: spawn on launch (unless one is already serving :8000 — dev/attach
// mode), kill on exit. Lifecycle coupling is the whole point (STUDIO_PRODUCTION_PLAN P1').
struct Kernel(Mutex<Option<Child>>);

fn kernel_running() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        Duration::from_millis(300),
    )
    .is_ok()
}

fn studio_home() -> PathBuf {
    std::env::var("PARAMETIC_STUDIO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".parametic_studio")
        })
}

/// ~/.parametic_studio/config.json: {"python_path": "...", "kernel_dir": "...", "model": "..."}
/// (system-python packaging: the kernel source lives in the repo checkout, python is the user's)
fn load_config() -> (String, Option<PathBuf>, Option<String>) {
    let p = studio_home().join("config.json");
    if let Ok(mut f) = std::fs::File::open(&p) {
        let mut s = String::new();
        if f.read_to_string(&mut s).is_ok() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                return (
                    v["python_path"].as_str().unwrap_or("python").to_string(),
                    v["kernel_dir"].as_str().map(PathBuf::from),
                    v["model"].as_str().map(String::from),
                );
            }
        }
    }
    ("python".into(), None, None)
}

fn spawn_kernel() -> Option<Child> {
    if kernel_running() {
        log::info!("kernel already on :8000 — attach mode, not spawning");
        return None;
    }
    let (python, kernel_dir, model) = load_config();
    // dev fallback: `tauri dev` runs with cwd = studio_web/src-tauri → repo root is two levels up
    let dir = kernel_dir.or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|d| d.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
    })?;
    let mut cmd = Command::new(&python);
    cmd.args(["-m", "parametic_studio.api"])
        .current_dir(&dir)
        .env("PARAMETIC_STUDIO_PARENT_WATCH", "1"); // kernel self-exits if the app dies uncleanly
    if let Some(m) = model {
        cmd.env("PARAMETIC_STUDIO_MODEL", m);
    }
    match cmd.spawn() {
        Ok(child) => {
            log::info!("kernel spawned (pid {}) via {} in {:?}", child.id(), python, dir);
            Some(child)
        }
        Err(e) => {
            log::error!("kernel spawn failed ({} in {:?}): {e}", python, dir);
            None // frontend shows "kernel offline · reconnecting…" — fix config.json and relaunch
        }
    }
}

/// Splash → main handoff: the frontend calls this once the kernel is reachable
/// (or after a timeout). Show + focus the main window, close the splash.
#[tauri::command]
fn close_splash(app: tauri::AppHandle) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.close();
    }
}

/// Native menubar. Custom items emit `menu` events to the webview so App.tsx drives the UI.
fn build_menu(app: &tauri::AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let app_menu = Submenu::with_items(
        app,
        "Parametic Studio",
        true,
        &[
            &PredefinedMenuItem::about(app, None, None)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "settings", "Settings…", true, Some("Cmd+,"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    let model_menu = Submenu::with_items(
        app,
        "Model",
        true,
        &[
            &MenuItem::with_id(app, "load-model", "Load Model…", true, Some("Cmd+L"))?,
            &MenuItem::with_id(app, "unload-all", "Unload All", true, None::<&str>)?,
        ],
    )?;
    let view_menu = Submenu::with_items(
        app,
        "View",
        true,
        &[
            &MenuItem::with_id(app, "toggle-theme", "Toggle Theme", true, Some("Cmd+Shift+T"))?,
            &MenuItem::with_id(app, "toggle-explorer", "Toggle Explorer", true, Some("Cmd+B"))?,
        ],
    )?;
    Menu::with_items(app, &[&app_menu, &model_menu, &view_menu])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![close_splash])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            app.manage(Kernel(Mutex::new(spawn_kernel())));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(k) = app.try_state::<Kernel>() {
                    if let Some(mut child) = k.0.lock().unwrap().take() {
                        let _ = child.kill(); // window closed → kernel goes with it
                        let _ = child.wait();
                    }
                }
            }
        });
}
