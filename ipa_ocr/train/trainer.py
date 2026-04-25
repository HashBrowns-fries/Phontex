"""训练器"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)
from torch.cuda.amp import autocast, GradScaler

from ipa_ocr.model import create_model, CTCLoss
from ipa_ocr.train.config import TrainConfig
from ipa_ocr.utils.characters import idx_to_char


class Trainer:
    """训练器"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 模型
        self.model = create_model(
            backbone=config.backbone,
            hidden_dim=config.hidden_dim,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout,
            use_v2=config.use_v2,
            use_v3=config.use_v3,
            use_v4=config.use_v4,
            use_v5=config.use_v5,
            use_attention=config.use_attention,
        ).to(self.device)
        print(self.model)
        # 损失函数
        self.criterion = CTCLoss(blank=0).to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None

        # 状态
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer == "AdamW":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == "CosineAnnealingLR":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.scheduler_params.get("eta_min", 1e-6),
            )
        elif self.config.scheduler == "CosineAnnealingWarmRestarts":
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        elif self.config.scheduler == "StepLR":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            return None

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch — 简洁版本"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images, targets, target_lengths = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    log_probs = self.model(images)
                    loss, _ = self._compute_loss(
                        log_probs, targets, target_lengths, images.size(0)
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs = self.model(images)
                loss, _ = self._compute_loss(
                    log_probs, targets, target_lengths, images.size(0)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _compute_loss(self, log_probs, targets, target_lengths, batch_size):
        import os
        if os.environ.get("DEBUG_LOSS") == "1":
            print(f"DEBUG: log_probs={log_probs.shape}, targets={targets.shape}, "
                  f"target_lengths={target_lengths.shape if hasattr(target_lengths,'shape') else target_lengths}, "
                  f"batch_size={batch_size}")
        if log_probs.dim() == 2:
            # v1 处理（不应使用，但保留）
            log_probs = (
                log_probs.unsqueeze(1).permute(1, 0, 2).permute(0, 2, 1)
            )  # (1, C, B)
            input_length = 1
        else:
            # v4/v5: (B, T, C) → CTC需要 (T, B, C)
            log_probs = log_probs.permute(1, 0, 2)
            input_length = log_probs.size(0)

        # 确保 input_lengths 在正确的设备上
        device = log_probs.device
        input_lengths = torch.full(
            size=(batch_size,), fill_value=input_length, dtype=torch.long, device=device
        )
        loss = self.criterion(log_probs, targets, target_lengths, input_lengths)

        if (target_lengths > input_length).any():
            print(f"警告：存在 target_length > input_length ({input_length})")
            raise ValueError("target_length 超过 input_length")
        return loss, input_lengths

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        for batch in val_loader:
            images, targets, target_lengths = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # 前向传播
            log_probs = self.model(images)

            # 损失
            loss, _ = self._compute_loss(
                log_probs, targets, target_lengths, images.size(0)
            )
            total_loss += loss.item()
            num_batches += 1

            # 准确率
            predictions = self.model.decode(log_probs, method=self.config.decode_method)
            target_texts = self._decode_targets(targets, target_lengths)

            for pred, target in zip(predictions, target_texts):
                correct_chars = sum(pc == tc for pc, tc in zip(pred, target))
                total_chars = max(len(pred), len(target))
                correct += correct_chars
                total += total_chars

        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0

        return {"loss": avg_loss, "accuracy": accuracy}

    def _decode_targets(
        self, targets: torch.Tensor, target_lengths: torch.Tensor
    ) -> List[str]:
        """解码目标文本"""
        texts = []
        targets = targets.cpu()
        target_lengths = target_lengths.cpu()

        idx = 0
        for length in target_lengths:
            text = ""
            for i in range(length.item()):
                char_idx = targets[idx + i].item()
                if char_idx in idx_to_char:
                    text += idx_to_char[char_idx]
            texts.append(text)
            idx += length.item()

        return texts

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ):
        """训练"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 早停计数器
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {"loss": 0, "accuracy": 0}

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录历史
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(0)  # 训练时不计算准确率
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # 保存最佳模型
            if val_metrics["loss"] < self.best_loss - self.config.min_delta:
                self.best_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.config.patience and self.config.patience > 0:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # 定期保存
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # 保存最终模型
        self.save_checkpoint("final_model.pth")

        # 保存训练历史
        self.save_history()

        print("Training completed!")

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
            "config": self.config.to_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = Path(self.config.output_dir) / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded: {checkpoint_path}")

    def save_history(self):
        """保存训练历史"""
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


def train_with_config(
    config: TrainConfig, train_loader: DataLoader, val_loader: DataLoader
):
    """使用配置训练"""
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    return trainer
