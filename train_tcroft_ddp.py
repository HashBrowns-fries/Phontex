"""TrOCR fine-tuning for IPA OCR — 双卡DDP并行版本"""
import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)


class IPATRROCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor, max_length: int = 64):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length

        label_path = next(
            (p for p in [self.data_dir / "labels.json", self.data_dir / "annotations.json"] if p.exists()),
            None
        )
        if label_path:
            with open(label_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.samples = [
                    (str(self.data_dir / "images" / fname), text)
                    for fname, text in raw.items()
                    if (self.data_dir / "images" / fname).exists()
                ]
        else:
            img_dir = self.data_dir / "images"
            self.samples = [(str(p), p.stem) for p in sorted(img_dir.glob("*.png"))]

        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_loop(rank: int, world_size: int, args):
    setup(rank, world_size)
    device = torch.device(rank)
    is_main = rank == 0

    # Processor — load once per process
    processor = TrOCRProcessor.from_pretrained(args.model_name)

    # Model
    if args.pretrained_path and is_main:
        model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_path)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Gradient checkpointing — 减少约30%显存
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if is_main:
            print("Gradient checkpointing enabled")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Datasets
    train_ds = IPATRROCDataset(args.train_dir, processor, max_length=args.max_length)
    val_ds = IPATRROCDataset(args.val_dir, processor, max_length=args.max_length)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    if is_main:
        print(f"\n{'='*60}")
        print(f"TrOCR IPA Fine-tuning — {world_size}-GPU DDP")
        print(f"{'='*60}")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        print(f"Batch/GPU: {args.batch_size}  Total: {args.batch_size * world_size}")
        print(f"Epochs: {args.num_epochs}")
        print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = args.patience
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Validation — only rank 0
        if is_main:
            model_eval = model.module
            model_eval.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    pv = batch["pixel_values"].to(device, non_blocking=True)
                    lbl = batch["labels"].to(device, non_blocking=True)
                    with autocast(device_type='cuda'):
                        out = model_eval(pixel_values=pv, labels=lbl)
                    val_loss += out.loss.item()

            avg_train = total_loss / len(train_loader)
            avg_val = val_loss / max(len(val_loader), 1)
            lr = optimizer.param_groups[0]["lr"]

            log_line = (f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | "
                        f"lr={lr:.2e}")
            print(log_line)
            if is_main:
                log_file = Path(args.output_dir) / "train.log"
                with open(log_file, "a") as lf:
                    lf.write(log_line + "\n")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                save_path = Path(args.output_dir) / "best_model"
                model_eval.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print("  ✓ Best model saved")
                if is_main:
                    with open(Path(args.output_dir) / "train.log", "a") as lf:
                        lf.write("  ✓ Best model saved\n")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    print("\nTraining complete!")
                    with open(Path(args.output_dir) / "train.log", "a") as lf:
                        lf.write(f"Early stopping at epoch {epoch+1}\n")
                        lf.write("Training complete!\n")
                    cleanup()
                    return

            if (epoch + 1) % args.save_interval == 0:
                ckpt = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}"
                model_eval.save_pretrained(ckpt)
                processor.save_pretrained(ckpt)

    if is_main:
        final_path = Path(args.output_dir) / "final_model"
        model.module.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        print("\nTraining complete!")
        with open(Path(args.output_dir) / "train.log", "a") as lf:
            lf.write("Training complete!\n")

    cleanup()


def main():
    parser = argparse.ArgumentParser("TrOCR IPA Fine-tuning DDP")
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--val-dir", type=str, default="data/val")
    parser.add_argument("--output-dir", type=str, default="outputs_trocr")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    # Auto-detect number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"WARNING: Only {world_size} GPU(s) found. Using single-GPU mode.")
        world_size = 1

    print(f"Launching {world_size}-GPU training...")

    if world_size > 1:
        torch.multiprocessing.spawn(
            train_loop, args=(world_size, args), nprocs=world_size, join=True
        )
    else:
        train_loop(0, 1, args)


if __name__ == "__main__":
    main()
