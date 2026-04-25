"""TrOCR fine-tuning with LoRA — parameter-efficient fine-tuning on IPA OCR.

LoRA strategy (from research):
  - Apply LoRA to decoder (cross-attention + ML LLaMA-style layers)
  - Freeze encoder (ViT) — preserves pretrained vision features
  - Rank r=8-16, alpha=16, dropout=0.05
  - Reduces overfitting on small synthetic datasets
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType

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


def build_lora_config(rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    """LoRA config for TrOCR decoder.

    TaskType: SEQ2SEQ_LM for encoder-decoder models.
    Applies LoRA to cross-attention (q_proj, v_proj) in decoder layers.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "v_proj",          # cross-attention
            "k_proj", "o_proj",          # self-attention
            "gate_proj", "up_proj", "down_proj",  # FFN
        ],
        bias="none",
    )


def build_encoder_lora_config(rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    """LoRA config for TrOCR encoder (ViT) — standard LoRA (not DoRA).

    ViT attention module names:
      encoder.layer.N.attention.attention.query  → query
      encoder.layer.N.attention.attention.key    → key
      encoder.layer.N.attention.attention.value  → value
      encoder.layer.N.attention.output.dense     → dense
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
        use_dora=False,
    )


def train_loop(rank: int, world_size: int, args):
    setup(rank, world_size)
    device = torch.device(rank)
    is_main = rank == 0

    processor = TrOCRProcessor.from_pretrained(args.model_name)

    # ── Extend tokenizer with IPA characters ────────────────────────────
    IPA_TOKENS = ["ˈ", "ˌ", "ː", "ʰ", "̃", "ʷ", "ʲ", "ɥ"]
    num_added = processor.tokenizer.add_tokens(IPA_TOKENS)
    if is_main:
        print(f"[Tokenizer] Added {num_added} IPA tokens: {IPA_TOKENS}")

    # Load base model
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Resize decoder token embeddings to accommodate new tokens
    model.decoder.model.decoder.resize_token_embeddings(len(processor.tokenizer))
    if is_main:
        print(f"[Tokenizer] Decoder embed resized to {model.decoder.model.decoder.embed_tokens.weight.shape[0]} tokens")

    # Apply encoder DoRA (if enabled)
    if args.use_encoder_lora:
        enc_lora = build_encoder_lora_config(
            rank=args.encoder_lora_rank,
            alpha=args.encoder_lora_alpha,
            dropout=args.lora_dropout,
        )
        model = get_peft_model(model, enc_lora)
        if is_main:
            enc_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad and any(n.startswith("encoder.") and "lora_" in n for n, _ in model.named_parameters()))
            print(f"[DoRA] Encoder DoRA: r={args.encoder_lora_rank}, alpha={args.encoder_lora_alpha}, use_dora=True")

    # Freeze encoder if NOT using encoder LoRA
    if not args.use_encoder_lora:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if is_main:
            enc_freeze_params = sum(p.numel() for p in model.encoder.parameters())
            print(f"[LoRA] Encoder frozen: {enc_freeze_params/1e6:.1f}M params")
    elif is_main:
        enc_train_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"[DoRA] Encoder trainable: {enc_train_params/1e6:.1f}M params")

    # Apply decoder LoRA
    dec_lora = build_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = get_peft_model(model, dec_lora)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Datasets
    train_ds = IPATRROCDataset(args.train_dir, processor, max_length=args.max_length)
    val_ds   = IPATRROCDataset(args.val_dir,   processor, max_length=args.max_length)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    if is_main:
        print(f"\n{'='*60}")
        print(f"TrOCR IPA Fine-tuning — {world_size}-GPU DDP + LoRA")
        print(f"{'='*60}")
        if args.use_encoder_lora:
            print(f"Encoder: DoRA r={args.encoder_lora_rank}, alpha={args.encoder_lora_alpha}")
        else:
            print(f"Encoder: frozen (ViT pretrained)")
        print(f"Decoder: LoRA r={args.lora_rank}, alpha={args.lora_alpha}")
        print(f"Tokenizer: {num_added} IPA tokens added")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        print(f"Batch/GPU: {args.batch_size}  Total: {args.batch_size * world_size}")
        print(f"Epochs: {args.num_epochs}")
        print(f"{'='*60}\n")

    # Separate learning rates for encoder DoRA vs decoder LoRA
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder." in name and "lora_" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    if is_main and args.use_encoder_lora:
        enc_m = sum(p.numel() for p in encoder_params) / 1e6
        dec_m = sum(p.numel() for p in decoder_params) / 1e6
        print(f"[Optimizer] Encoder DoRA: {enc_m:.1f}M params @ lr={args.encoder_lr} | "
              f"Decoder LoRA: {dec_m:.1f}M params @ lr={args.learning_rate}")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr} if encoder_params else {"params": [], "lr": 0},
        {"params": decoder_params, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step-level log file ─────────────────────────────────────────────
    step_log_path = Path(args.output_dir) / "step.log"
    step_log_file = None
    if is_main:
        step_log_file = open(step_log_path, "a")

    # ── Resume from checkpoint ─────────────────────────────────────────
    start_epoch = 0
    ckpt_path = Path(args.output_dir) / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        if is_main:
            print(f"[Resume] Loaded checkpoint — resuming from epoch {start_epoch}/{args.num_epochs}")
            print(f"         best_val_loss={best_val_loss:.4f}, epochs_no_improve={epochs_no_improve}")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}",
                    ncols=100, leave=True)
        for step, batch in pbar:
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
            avg_loss = total_loss / (step + 1)
            lr_now = optimizer.param_groups[-1]["lr"]
            pbar.set_postfix(avg=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")

            # ── Write step log every 50 steps ────────────────────────
            if is_main and step % 50 == 0:
                ts = time.strftime("%H:%M:%S")
                line = f"{ts} epoch={epoch+1} step={step}/{len(train_loader)} loss={loss.item():.4f} avg_loss={avg_loss:.4f} lr={lr_now:.2e}\n"
                step_log_file.write(line)
                step_log_file.flush()

        # Validation — rank 0 only
        if is_main:
            model_eval = model.module
            model_eval.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val", ncols=80, leave=False):
                    pv  = batch["pixel_values"].to(device, non_blocking=True)
                    lbl = batch["labels"].to(device, non_blocking=True)
                    with autocast(device_type='cuda'):
                        out = model_eval(pixel_values=pv, labels=lbl)
                    val_loss += out.loss.item()

            avg_train = total_loss / len(train_loader)
            avg_val   = val_loss / max(len(val_loader), 1)
            lr        = optimizer.param_groups[-1]["lr"]  # decoder lr (always present)

            log_line = (f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | "
                        f"lr={lr:.2e}")
            print(log_line)

            with open(Path(args.output_dir) / "train.log", "a") as lf:
                lf.write(log_line + "\n")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                model_eval.save_pretrained(Path(args.output_dir) / "best_model")
                processor.save_pretrained(Path(args.output_dir) / "best_model")
                print("  ✓ Best model saved")
                with open(Path(args.output_dir) / "train.log", "a") as lf:
                    lf.write("  ✓ Best model saved\n")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{args.patience})")

            # ── Save checkpoint after each epoch ──────────────────────
            ckpt = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
            }
            torch.save(ckpt, Path(args.output_dir) / "checkpoint.pt")
            if is_main:
                print(f"  ✓ Checkpoint saved (epoch {epoch+1})")

            # Broadcast early-stop decision to all ranks so they all exit together
            stop_flag = torch.tensor([1 if epochs_no_improve >= args.patience else 0],
                                    dtype=torch.int8, device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
            if stop_flag.item() > 0 and is_main:
                print(f"  Early stopping at epoch {epoch+1}")
                with open(Path(args.output_dir) / "train.log", "a") as lf:
                    lf.write(f"Early stopping at epoch {epoch+1}\n")
                step_log_file.close()
                cleanup()
                return
            elif stop_flag.item() > 0:
                cleanup()
                return

    if is_main:
        model.module.save_pretrained(Path(args.output_dir) / "final_model")
        processor.save_pretrained(Path(args.output_dir) / "final_model")
        print("\nTraining complete!")
        with open(Path(args.output_dir) / "train.log", "a") as lf:
            lf.write("Training complete!\n")
        step_log_file.close()

    cleanup()


def main():
    parser = argparse.ArgumentParser("TrOCR IPA Fine-tuning DDP + LoRA")
    parser.add_argument("--train-dir",      type=str, default="data/train")
    parser.add_argument("--val-dir",        type=str, default="data/val")
    parser.add_argument("--output-dir",     type=str, default="outputs_tcroft_lora")
    parser.add_argument("--model-name",     type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--max-length",     type=int, default=128)
    parser.add_argument("--batch-size",    type=int, default=48)
    parser.add_argument("--num-epochs",    type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # higher LR for LoRA
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1)
    parser.add_argument("--num-workers",    type=int, default=4)
    parser.add_argument("--save-interval",  type=int, default=5)
    parser.add_argument("--patience",       type=int, default=3)
    # LoRA params
    parser.add_argument("--lora-rank",     type=int, default=8)
    parser.add_argument("--lora-alpha",    type=int, default=16)
    parser.add_argument("--lora-dropout",  type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=None, help="Force specific GPU index")
    # Encoder DoRA params
    parser.add_argument("--use-encoder-lora", action="store_true", default=False,
                       help="Apply DoRA to encoder (ViT) for cross-domain generalization")
    parser.add_argument("--encoder-lora-rank", type=int, default=64)
    parser.add_argument("--encoder-lora-alpha", type=int, default=128)
    parser.add_argument("--encoder-lr", type=float, default=5e-5,
                       help="Learning rate for encoder DoRA (default: 5e-5)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume training from checkpoint.pt in output-dir")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    # Force single GPU if --gpu specified
    if hasattr(args, 'gpu') and args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        world_size = 1
        print(f"Using single GPU {args.gpu}")
    elif world_size > 1:
        print(f"Using {world_size} GPUs for training")
    else:
        print(f"WARNING: Only {world_size} GPU(s) available.")
        world_size = 1

    if world_size > 1:
        torch.multiprocessing.spawn(
            train_loop, args=(world_size, args), nprocs=world_size, join=True
        )
    else:
        train_loop(0, 1, args)


if __name__ == "__main__":
    main()
