"""TrOCR fine-tuning — 单卡版本，先跑通"""
import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import argparse
from pathlib import Path
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

MODEL_NAME = "microsoft/trocr-base-handwritten"


class IPATRROCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor, max_length: int = 64):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length

        label_path = next(
            (p for p in [self.data_dir / "labels.json", self.data_dir / "annotations.json"] if p.exists()), None
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
        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pv = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--output-dir", default="outputs_tcroft")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"\n{'='*60}")
    print("TrOCR IPA Fine-tuning — single GPU")
    print(f"{'='*60}")

    print(f"Loading processor: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Gradient checkpointing to save memory
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_ds = IPATRROCDataset(args.train_dir, processor, max_length=args.max_length)
    val_ds = IPATRROCDataset(args.val_dir, processor, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                out = model(pixel_values=pv, labels=lbl)
                loss = out.loss / args.gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation
            pbar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    out = model(pixel_values=pv, labels=lbl)
                val_loss += out.loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / max(len(val_loader), 1)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{args.epochs} | {elapsed:.0f}s | "
              f"train={avg_train:.4f} | val={avg_val:.4f} | lr={lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            path = Path(args.output_dir) / "best_model"
            model.save_pretrained(path)
            processor.save_pretrained(path)
            print("  ✓ Best model saved")

        if (epoch + 1) % args.save_interval == 0:
            ckpt = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}"
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)

    # Final save
    path = Path(args.output_dir) / "final_model"
    model.save_pretrained(path)
    processor.save_pretrained(path)

    # Save history
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
