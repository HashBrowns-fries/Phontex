"""TrOCR fine-tuning for IPA OCR — 预训练Vision+GPT2模型直接迁移到IPA字符识别"""

import os
import sys

# 抑制transformers加载噪声
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from tqdm import tqdm


class IPATRROCDataset(Dataset):
    """IPA TrOCR数据集"""

    def __init__(self, data_dir: str, processor, max_length: int = 64):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length

        # 加载标签
        label_paths = [
            self.data_dir / "labels.json",
            self.data_dir / "annotations.json",
        ]
        label_path = next((p for p in label_paths if p.exists()), None)

        if label_path:
            with open(label_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                # labels.json: {filename: text}
                self.samples = [
                    (str(self.data_dir / "images" / fname), text)
                    for fname, text in raw.items()
                    if (self.data_dir / "images" / fname).exists()
                ]
        else:
            # Fallback: 从目录结构加载
            img_dir = self.data_dir / "images"
            self.samples = [
                (str(p), p.stem)
                for p in sorted(img_dir.glob("*.png"))
            ]

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        img_path, text = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        # 用processor处理图像和文本
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # 编码标签
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids.squeeze(0)

        # 将padding token替换为-100（loss不计）
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn_trocr(batch: List[Dict], processor) -> Dict[str, torch.Tensor]:
    """合并batch"""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


class TrOCRTrainer:
    """TrOCR训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # 加载processor和模型
        print("Loading TrOCR processor and model...")
        self.processor = TrOCRProcessor.from_pretrained(args.model_name)

        if args.pretrained_path:
            print(f"Resuming from {args.pretrained_path}")
            self.model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_path)
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

        # 配置解码策略
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        # generation_config for generation
        self.model.generation_config.max_length = args.max_length
        self.model.generation_config.early_stopping = True
        self.model.generation_config.num_beams = args.num_beams
        self.model.generation_config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.processor.tokenizer.eos_token_id

        self.model.to(self.device)
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params")

    def create_dataloaders(self):
        """创建数据加载器"""
        train_dataset = IPATRROCDataset(
            self.args.train_dir, self.processor, max_length=self.args.max_length
        )
        val_dataset = IPATRROCDataset(
            self.args.val_dir, self.processor, max_length=self.args.max_length
        )

        def collate(batch):
            return collate_fn_trocr(batch, self.processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate,
        )
        return train_loader, val_loader

    def compute_accuracy(self, pred_ids, label_ids):
        """计算字符级准确率"""
        # label_ids中-100是padding
        correct, total = 0, 0
        for pred, label in zip(pred_ids, label_ids):
            # 移除-100和eos
            pred_clean = [p for p in pred if p not in (-100, self.processor.tokenizer.eos_token_id)]
            label_clean = [l for l in label if l != -100]
            min_len = min(len(pred_clean), len(label_clean))
            correct += sum(p == l for p, l in zip(pred_clean[:min_len], label_clean[:min_len]))
            total += max(len(pred_clean), len(label_clean))
        return correct / total if total > 0 else 0.0

    def evaluate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                total_loss += outputs.loss.item()

                # 解码预测
                pred_ids = self.model.generate(
                    pixel_values,
                    max_length=self.args.max_length,
                    num_beams=self.args.num_beams,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                    eos_token_id=self.model.config.eos_token_id,
                )
                all_preds.extend(pred_ids.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(len(val_loader), 1)
        acc = self.compute_accuracy(all_preds, all_labels)
        return avg_loss, acc

    def train(self):
        """训练"""
        train_loader, val_loader = self.create_dataloaders()
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )

        # 学习率调度
        total_steps = len(train_loader) * self.args.num_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        scaler = GradScaler()
        best_val_loss = float("inf")
        os.makedirs(self.args.output_dir, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_loss = 0
            epoch_start = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with autocast():
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 验证
            val_loss, val_acc = self.evaluate(val_loader)
            avg_train_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]

            # 记录历史
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(lr)

            print(f"\nEpoch {epoch + 1}/{self.args.num_epochs} — {epoch_time:.0f}s")
            print(f"  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr:.2e}")

            # 保存best和定期checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(self.args.output_dir) / "best_model"
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                print(f"  Best model saved: {save_path}")

            if (epoch + 1) % self.args.save_interval == 0:
                ckpt_path = Path(self.args.output_dir) / f"checkpoint_epoch_{epoch + 1}"
                self.model.save_pretrained(ckpt_path)
                self.processor.save_pretrained(ckpt_path)

        # 保存最终模型
        final_path = Path(self.args.output_dir) / "final_model"
        self.model.save_pretrained(final_path)
        self.processor.save_pretrained(final_path)

        # 保存历史
        with open(Path(self.args.output_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete! Models saved to {self.args.output_dir}")
        return history


def main():
    parser = argparse.ArgumentParser("TrOCR IPA Fine-tuning")

    # 数据
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--val-dir", type=str, default="data/val")
    parser.add_argument("--output-dir", type=str, default="outputs_trocr")

    # 模型
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)

    # 训练
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=5)

    args = parser.parse_args()

    print("=" * 60)
    print("TrOCR IPA Fine-tuning")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    trainer = TrOCRTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
