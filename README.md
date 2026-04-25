# Phontex

**Fontex** = FOnt + TEX（音标识别，桌面端截图 OCR）

基于 TrOCR + LoRA 的国际音标（IPA）识别桌面工具。一键截图，自动识别 IPA 字符并复制到剪贴板。

---

## 功能

- `Ctrl+Shift+I` 快捷键触发截图
- 拖拽选择屏幕区域
- TrOCR+LoRA 离线 OCR（支持 CPU/GPU）
- 结果自动复制到剪贴板 + 毛玻璃弹窗

---

## 项目结构

```
phontex/
├── pyproject.toml          # 主项目依赖
├── uv.lock
├── README.md
├── CLAUDE.md               # Claude Code 项目指南
├── .gitignore
│
├── generate_training_data.py   # 合成 IPA 数据生成
├── train_tcroft_lora.py        # LoRA 微调训练
│
├── data/                       # 数据目录（gitignore）
├── outputs_lora_data2/          # 训练输出
│
└── frontend/
    ├── server/                  # Python FastAPI 后端
    │   ├── server.py            # OCR + 剪贴板 API
    │   └── pyproject.toml
    └── tauri/                   # Tauri 桌面应用
        ├── src-tauri/           # Rust 代码
        └── web/                 # HTML/CSS 前端
```

---

## 快速开始

### 安装依赖

```bash
uv sync
```

### 数据生成（可选）

```bash
uv run python generate_training_data.py --output-dir data/
```

### 训练 LoRA 模型（可选）

```bash
uv run python train_tcroft_lora.py \
    --train-dir data2/train \
    --val-dir data2/val \
    --output-dir outputs_lora_data2 \
    --num-epochs 10 \
    --batch-size 48
```

### 基准测试

```bash
uv run python benchmark_lora.py
uv run python benchmark_qwen.py
```

---

## 前端（Tauri 桌面应用）

```bash
cd frontend/tauri && npm install
npm run tauri dev      # 开发模式
npm run tauri build    # 打包
```

详见 [frontend/README.md](frontend/README.md)

---

## 技术栈

| 组件 | 技术 |
|------|------|
| OCR 模型 | TrOCR-base + LoRA (r=8, α=16) |
| 训练框架 | HuggingFace Transformers + PEFT |
| 数据生成 | 音系学约束的 IPA 合成器 |
| 桌面前端 | Tauri (Rust) + FastAPI (Python) |
| 前端 UI | HTML/CSS（毛玻璃风格） |

---

## 基准结果

| 模型 | In-Domain CER | Cross-Domain CER |
|------|-------------|------------------|
| TrOCR 零样本 | 124.75% | — |
| Qwen2.5-VL-3B 零样本 | 38.43% | — |
| TUM Calamari | 6.04% | 3.92% |
| **Phontex (Ours)** | **1.29%** | **3.30%** |
| EACL 2026 Calamari | 1.41% | — |

---

## 参考

- Li, M., et al. (2026). *OCR for IPA: A Benchmark and Strong Baseline.* EACL 2026.
- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation.* ICLR 2022.
- TUM-NLP/ocr-ipa: https://github.com/TUM-NLP/ocr-ipa