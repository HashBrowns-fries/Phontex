# Phontex Frontend

Tauri (Rust) + FastAPI (Python) desktop app for IPA OCR.

## Architecture

```
┌──────────────────────┐    HTTP (base64)    ┌─────────────────────────┐
│  Tauri App (Rust)    │◄──────────────────►│  FastAPI (Python)        │
│  ├─ Tray Icon         │                    │  └─ TrOCR+LoRA model    │
│  ├─ Global Hotkey     │                    └─────────────────────────┘
│  ├─ Screenshot        │
│  └─ Result Popup      │
└──────────────────────┘
```

## Quick Start

### 1. Install system dependencies

```bash
# Linux
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev

# macOS
brew install gtk+3 webkit2gtk-4.1

# Windows
# Install Visual Studio Build Tools with C++ workload
```

### 2. Install Node.js deps

```bash
cd frontend/tauri && npm install
```

### 3. Start Python backend (standalone, for testing)

```bash
cd frontend/server && uv sync
uv run uvicorn server:app --port 8765 --reload
```

### 4. Run Tauri dev

```bash
cd frontend/tauri && npm run tauri dev
```

## Build

```bash
cd frontend/tauri && npm run tauri build
```

The executable will be in `tauri/target/release/bundle/`.

## Keyboard Shortcut

`Ctrl+Shift+I` — capture screen region → OCR → copy to clipboard

## Project Structure

```
frontend/
├── server/          # Python FastAPI backend
│   ├── server.py    # API + TrOCR+LoRA inference
│   └── pyproject.toml
└── tauri/           # Rust desktop app
    ├── src-tauri/
    │   ├── src/main.rs       # Tray, hotkey, screenshot, IPC
    │   ├── tauri.conf.json  # Window, tray, permissions
    │   └── Cargo.toml
    └── web/          # HTML/CSS popup UI
        ├── index.html
        └── result.html
```