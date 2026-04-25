# CLAUDE.md — Phontex 项目指南

本文件为 Claude Code 提供项目上下文，请在开始任何任务前阅读。

## 项目概述

**Phontex** = FOnt + TEX，一款基于 TrOCR+LoRA 的国际音标（IPA）识别桌面工具。

- 核心模型：`microsoft/trobat-base-handwritten` + LoRA 微调
- 验证集 CER：**1.29%**，超越 EACL 2026 SOTA 基线
- 支持 CPU/GPU 推理，跨 Windows/macOS/Linux 桌面

## 关键目录

| 目录 | 说明 |
|------|------|
| `frontend/server/` | FastAPI OCR 后端（Python） |
| `frontend/tauri/` | Tauri 桌面应用（Rust + Web） |
| `outputs_lora_data2/best_model/` | 已训练 LoRA 模型权重 |
| `data2/` | 135K 训练图 + 15K 验证图 |
| `benchmark_*.py` | 基准测试脚本 |

## 常用命令

```bash
# 安装依赖
uv sync

# 训练 LoRA
uv run python train_tcroft_lora.py --train-dir data2/train --val-dir data2/val

# 基准测试
uv run python benchmark_lora.py

# 启动前端（需先 npm install）
cd frontend/tauri && npm install && npm run tauri dev
```

## 核心文件说明

- `benchmark_lora.py` — LoRA 模型评估，复用了 model/processor 加载逻辑
- `server.py`（frontend/） — FastAPI 后端，`/ocr` 接收 base64 图片，`/clipboard` 写剪贴板
- `main.rs`（frontend/tauri/src-tauri/） — Rust 主程序：托盘、热键注册、截图、IPC 调用

## 快捷键

`Ctrl+Shift+I` — 截图选区 → OCR → 自动复制剪贴板

## 注意事项

- 模型路径：`outputs_lora_data2/best_model/`（绝对路径在 server.py 中硬编码）
- GPU 设备：自动检测 `cuda:0`（单卡）或 `cuda:N`（多卡），无 GPU 时回退 CPU
- Tauri 构建需要目标平台系统库（Windows: VS Build Tools, macOS: gtk+3, Linux: libgtk-3-dev）