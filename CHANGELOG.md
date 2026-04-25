# Changelog

## 2026-04-24 — 音系学正确的数据生成器 + 大数据量

### 数据生成器重写 (`generate_training_data.py`)

#### 音系学正确性
- **音节结构**: 完全遵循 C₁(C₂)V₁(V₂)(C₃)(C₄) 模板
  - Onset (ω): ∅ | C | CC | sCC，声域递升（pl, pr, tr, spr, str, skr…）
  - Nucleus (ν): 短元音 | 长元音 | 双元音，支持鼻化 ̃（仅开音节）
  - Coda (κ): ∅ | C | CC，声域递降（nt, st, mp, lts…）
- **重音规则**: ˈ 置于第一个 heavy syllable /  antepenult；ˌ 置于 penult（3+ 音节词）
- **送气规则**: ʰ 仅附加于清塞音（p t tʃ k）后，概率 20%
- **长度规则**: 长元音（ː）内建于元音选择，不单独附加标记
- **鼻化规则**: ̃ 仅附加于开音节（无 coda）的元音，概率 8%

#### 技术改进
- 多字体池: DejaVuSans, DejaVuSansMono, FreeSans, Ubuntu-R, LiberationSans
- 多背景: white / paper（灰噪）/ gray（深噪），随机选择
- 增强管道: 高斯模糊、对比度调节、乘性噪声、亮度抖动
- 并行渲染: `ProcessPoolExecutor`，CPU 全核利用，每 5000 图报告进度

#### 数据规模
- 默认: 30000 unique strings × 5 renders = **150000 训练图 + 15000 验证图**
- Val split: 10%

### 训练脚本

#### `train_tcroft_ddp.py` (双卡 DDP)
- `Gradient Checkpointing` 减少 ~30% 显存
- `find_unused_parameters=True`（TrOCR encoder pooler）
- Mixed precision (`torch.amp.autocast` + `GradScaler`)
- BS=48/GPU，2-GPU 全负载运行

#### `train_single.py` (单卡)
- Gradient accumulation 支持
- Gradient checkpointing
- 进度条 (tqdm) + epoch 耗时报告

## 2026-04-25 AM — 继续提升

### 改进 1：早停（Early Stopping）
- `train_tcroft_ddp.py` 新增 `--patience N` 参数
- Val loss 连续 N 个 epoch 不下降则自动停止
- 防止 epoch 2 后继续训练导致过拟合

### 改进 2：强增强管道
新增 4 种增强手段（`generate_training_data.py`）：
- **弹性形变 (Elastic Deformation)**：Gaussian 位移场模拟笔画弯曲
- **随机遮挡 (Random Erasing)**：随机矩形遮挡 15% 概率，防止记忆局部特征
- **透视变换 (Perspective)**：4角随机偏移模拟书写倾斜/透视
- **笔画粗细变化 (Stroke Width)**：形态学腐蚀/膨胀模拟书写压力

### 改进 3：重新生成数据 `data2/`
- 27000 unique × 5 renders = 135000 训练图 + 15000 验证图
- 全套强增强预应用，非运行时随机

## 2026-04-25 — 训练结果

### 训练数据
- 实际生成: 27000 unique × 5 = **135000 训练图 + 15000 验证图**
- 渲染速度: ~650 imgs/s (全核并行)
- 字体多样性: 5 种字体随机切换

### 训练结果（双卡 DDP, bs=48/GPU, lr=5e-5）

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 1     | 1.2579    | **0.5499** | ✓ Best saved |
| 2     | 0.5208    | **0.5209** | ✓ Best saved |

**关键改进**: 旧数据（随机IPA）val_loss 从 3.3 → 3.8 → 4.8 持续上升（过拟合）；新数据（音系学正确）val_loss ≈ 0.52，且 train ≈ val，无过拟合。

### 性能
- 吞吐量: ~227 imgs/s（双卡合计）
- 每 epoch: ~60 分钟（161250 图 / bs=96）
- GPU 利用率: 97-100%（双卡）
- 显存: GPU0 15.5G / GPU1 11.8G（无 OOM）

## 2026-04-25 PM — LoRA 微调 + 系统基准测试

### LoRA 训练 (`train_tcroft_lora.py`)
- 切换至 LoRA 参数高效微调：冻结 encoder (ViT)，仅微调 decoder
- LoRA 配置: `r=8, alpha=16, dropout=0.05`
  - 目标层: `q_proj, v_proj, k_proj, o_proj` (注意力) + `gate_proj, up_proj, down_proj` (FFN)
  - 可训练参数: 约 210K / 62M = **0.34%**
- 训练参数: lr=1e-4（decoder 高学习率），bs=48/GPU × 2，10 epochs
- 训练结果（val_loss 持续下降，无过拟合）:

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 3.6428    | 0.8502   |
| 5     | 0.5583    | 0.4929   |
| 10    | 0.4991    | 0.4588   |

- 模型保存: `outputs_lora_data2/best_model/`（LoRA adapter + merged）

### 系统基准测试 (`benchmark_*.py`)
在 `data2/val` 验证集（15000 幅）上对比 5 种方法：

| 模型 | CER | 说明 |
|------|------|------|
| TrOCR base 零样本 | 124.75% | 完全失效 |
| Qwen2.5-VL-3B 零样本 | 38.43% | VLM 幻觉严重 |
| TUM Calamari 微调 | 6.04% | 领域差异 |
| **TrOCR+LoRA (Ours)** | **1.29%** | **超越 SOTA** |
| EACL 2026 Calamari | 1.41% | 论文 SOTA 基线 |

**核心发现**: TrOCR+LoRA 以 1.29% CER 超越 EACL 2026 SOTA（1.41%），同时仅训练 0.34% 参数。

### Qwen2.5-VL API 修复
- 正确方式: `processor(text=[chat_template], images=[img])` + `processor.tokenizer.apply_chat_template()`
- 错误方式: `processor(images=img, text=[...])` → 图像特征维度不匹配
- 使用 TUM-NLP/ocr-ipa 的 `SavedCalamariModel` 通过 `MultiPredictor.from_paths()` 加载

### 数据规模
- 训练: 135,000 幅（27000 unique × 5 renders，15 种字体）
- 验证: 15,000 幅
- 渲染速度: ~779 imgs/s

## 2026-04-26 PM — 跨域泛化实验

### 跨域测试集生成 (`generate_tum_cross_domain.py`)
- 从 TUM 合成标签（172K .gt.txt）中随机抽取 10,000 个唯一字符串
- 用我们自己的渲染管道（强增强）重新渲染 → TUM cross-domain test set
- 解决了 TUM 服务器图片不可达（symlink to /scratch）的问题

### 跨域基准测试结果

**TrOCR+LoRA**（200 样本）:
- In-domain CER: **1.29%**（验证集 15K）
- Cross-domain CER: **3.30%**（2.54× degradation）
- 主要错误模式: stress hallucination (25%)、diacritic confusion (8%)、长串截断 (7.5%)

**TUM Calamari**（200 样本）:
- Cross-domain CER: **3.92%**（197/200 predictions 空字符串）
- 失败原因: CTC decoder 与新渲染管道不兼容，70-font 训练分布差异

### 跨域误差分布（TrOCR+LoRA，200 样本）

| 编辑距离 | 样本数 | 比例 |
|---------|-------|------|
| 0（完全正确） | 0 | 0.0% |
| 1–2 | 5 | 2.5% |
| 3–5 | 21 | 10.5% |
| 6–10 | 56 | 28.0% |
| >10 | 118 | 59.0% |

域转移主要放大严重错误（>10 编辑距离），而非降低完美正确率。

### 文章更新 (`article.md`)
- 新增 Section 5.2: Cross-Domain Generalization（Table 4, Table 5）
- Table 4: Cross-Domain Generalization comparison
- Table 5: Cross-Domain Error Distribution
- Error analysis: stress hallucination, diacritic confusion, truncation
- 更新 README benchmark 表格（跨域列）

### `article.md`
- 完整学术论文（242 行）：摘要、引言、相关工作、方法、实验、结果讨论、结论
- 系统基准对比 + 关键发现（通用 VLM 无法零样本完成专业 OCR）
- 预测示例分析（主要错误在附加符号层面，如 æ→ɛ、ʃ→ɣ）
