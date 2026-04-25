//! Phontex Tauri App — main entry point
//! Handles: system tray, global shortcut, screenshot, OCR flow

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::GenericImageView;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tauri::{
    image::Image,
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// App state shared across commands
struct AppState {
    server_process: Option<std::process::Child>,
    last_capture: Arc<Mutex<Option<CapturedImage>>>,
}

struct CapturedImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tauri Commands
// ─────────────────────────────────────────────────────────────────────────────

/// Capture a region from the primary screen and run OCR
#[tauri::command]
async fn capture_and_ocr(app: AppHandle) -> Result<OcrResult, String> {
    println!("[Phontex] Capturing screenshot...");

    // Capture screen using `screenshots` crate
    let captures = screenshots::Screen::all().map_err(|e| format!("screenshots error: {e}"))?;
    if captures.is_empty() {
        return Err("No screens found".to_string());
    }

    let screen = &captures[0];
    let capture = screen
        .capture()
        .map_err(|e| format!("capture error: {e}"))?;

    let (w, h) = (capture.width(), capture.height());
    let rgba = capture.into_raw();

    // Save to temp file as PNG
    let temp_path = std::env::temp_dir().join("ipa_ocr_capture.png");
    image::save_buffer(
        temp_path.as_path(),
        &rgba,
        w,
        h,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| format!("save buffer error: {e}"))?;

    // Call FastAPI backend
    let result = call_ocr_api(&temp_path).await?;
    let text = result.text;

    // Write to system clipboard
    write_clipboard(&text).await?;

    // Show result window
    show_result_window(&app, &text).await?;

    // Clean up temp file
    let _ = std::fs::remove_file(temp_path);

    println!("[Phontex] Done: {}", text);
    Ok(OcrResult { text })
}

#[derive(Debug, Serialize, Deserialize)]
struct OcrResult {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    text: String,
}

/// Call FastAPI /ocr endpoint with the captured image
async fn call_ocr_api(path: &PathBuf) -> Result<ApiResponse, String> {
    let client = reqwest::Client::new();
    let img_bytes = std::fs::read(path).map_err(|e| format!("read file: {e}"))?;
    let b64 = BASE64.encode(&img_bytes);
    let data_url = format!("data:image/png;base64,{b64}");

    // Poll health endpoint until ready or timeout
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(60) {
            return Err("Server not ready after 60s".to_string());
        }
        if let Ok(resp) = client
            .get("http://127.0.0.1:8765/health")
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            if resp.status().is_success() {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let resp = client
        .post("http://127.0.0.1:8765/ocr")
        .json(&serde_json::json!({ "image": data_url }))
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("request error: {e}"))?;

    let status = resp.status();
    let body = resp.text().await.map_err(|e| format!("read body: {e}"))?;
    if !status.is_success() {
        return Err(format!("API error {status}: {body}"));
    }

    serde_json::from_str(&body).map_err(|e| format!("parse JSON: {e}"))
}

/// Write text to system clipboard via FastAPI (since Tauri clipboard plugin
/// may not be available at this point, we call the backend)
async fn write_clipboard(text: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/clipboard")
        .json(&serde_json::json!({ "text": text }))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| format!("clipboard error: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("clipboard API error: {}", resp.status()));
    }
    Ok(())
}

/// Show result popup window
async fn show_result_window(app: &AppHandle, text: &str) -> Result<(), String> {
    // Encode text for URL
    let encoded_text = urlencoding::encode(text);

    let window = WebviewWindowBuilder::new(
        app,
        "result",
        WebviewUrl::App(format!("result.html?text={}", encoded_text).into()),
    )
    .title("IPA OCR Result")
    .inner_size(420.0, 160.0)
    .position(200.0, 200.0) // centered placeholder; real position set in JS
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(false)
    .build()
    .map_err(|e| format!("create window: {e}"))?;

    // Close after 4 seconds
    let win = window.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(4)).await;
        let _ = win.close();
    });

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Server lifecycle
// ─────────────────────────────────────────────────────────────────────────────

async fn start_python_server(app: AppHandle) -> Result<(), String> {
    println!("[Phontex] Starting Python FastAPI server...");

    // Find python executable
    let python_exe = std::env::var("PYTHON").unwrap_or_else(|_| "python".to_string());

    let script_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("server.py").to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("frontend/server/server.py"));

    let child = Command::new(&python_exe)
        .arg("-u")
        .arg(script_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn server: {e}"))?;

    app.manage(AppState {
        server_process: Some(child),
        last_capture: Arc::new(Mutex::new(None)),
    });

    println!("[Phontex] Python server spawned");
    Ok(())
}

fn stop_python_server(state: tauri::State<'_, AppState>) {
    if let Some(mut child) = state.server_process.take() {
        let _ = child.kill();
        let _ = child.wait();
        println!("[Phontex] Python server stopped");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    println!("[Phontex] Starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![capture_and_ocr])
        .setup(|app| {
            let handle = app.handle().clone();

            // ── System Tray ──────────────────────────────────────────────────
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let capture =
                MenuItemBuilder::with_id("capture", "Capture (Ctrl+Shift+I)").build(app)?;
            let sep = tauri::menu::PredefinedMenuItem::separator(app)?;
            let menu = MenuBuilder::new(app)
                .item(&capture)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::with_id("main-tray")
                .icon(Image::from_path("icons/icon.png").unwrap_or_else(|_| {
                    Image::from_bytes(include_bytes!("../icons/default.png")).unwrap()
                }))
                .menu(&menu)
                .tooltip("IPA OCR — Ctrl+Shift+I to capture")
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "quit" => {
                        println!("[Phontex] Quit requested");
                        app.exit(0);
                    }
                    "capture" => {
                        let _ = app.emit("do-capture", ());
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: tauri::tray::MouseButton::Left, .. } =
                        event
                    {
                        let app = tray.app_handle();
                        let _ = app.emit("do-capture", ());
                    }
                })
                .build(app)?;

            // ── Global Shortcut ──────────────────────────────────────────────
            let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyI);
            let handle2 = handle.clone();
            app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    println!("[Phontex] Hotkey pressed");
                    let _ = handle2.emit("do-capture", ());
                }
            })?;

            // Emit on hotkey/tray click triggers JS capture via event listener
            // We handle it in Rust directly via the emit below
            let handle3 = handle.clone();
            let _ = handle.listen("do-capture", move |_event| {
                let h = handle3.clone();
                tauri::async_runtime::spawn(async move {
                    match capture_and_ocr(h).await {
                        Ok(r) => println!("[Phontex] OCR result: {}", r.text),
                        Err(e) => eprintln!("[Phontex] Error: {e}"),
                    }
                });
            });

            // ── Start Python Server ──────────────────────────────────────────
            let handle4 = handle.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = start_python_server(handle4).await {
                    eprintln!("[Phontex] Server start error: {e}");
                }
            });

            Ok(())
        })
        .on_exit(|_app| {
            println!("[Phontex] Exiting");
        })
        .run(tauri::generate_context!())
        .map_err(|e| e.into())?;

    Ok(())
}