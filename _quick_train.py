"""Quick 1-epoch test to verify TrOCR training loop works"""
import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
from train_trocr import IPATRROCDataset, collate_fn_trocr
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.to(DEVICE)
model.train()
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

train_dataset = IPATRROCDataset('data/train', processor, max_length=64)
val_dataset = IPATRROCDataset('data/val', processor, max_length=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda b: collate_fn_trocr(b, processor))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=lambda b: collate_fn_trocr(b, processor))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader))
scaler = GradScaler()

print("Starting 1-epoch test...")
t0 = time.time()
total_loss = 0

for batch in tqdm(train_loader, desc="Training"):
    pixel_values = batch["pixel_values"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    optimizer.zero_grad()
    with autocast('cuda'):
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    total_loss += loss.item()

print("\n1 epoch done in {:.0f}s, avg loss: {:.4f}".format(time.time()-t0, total_loss/len(train_loader)))

model.eval()
val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with autocast('cuda'):
            outputs = model(pixel_values=pixel_values, labels=labels)
        val_loss += outputs.loss.item()
        gen_ids = model.generate(pixel_values[:2], max_length=64, num_beams=4,
            decoder_start_token_id=model.config.decoder_start_token_id,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id)
        for gen, label in zip(gen_ids, labels[:2]):
            gt = processor.tokenizer.decode(label[label != -100], skip_special_tokens=True)
            pred = processor.tokenizer.decode(gen, skip_special_tokens=True)
            print(f"  GT: {gt!r}  PD: {pred!r}")

print("\nVal loss: {:.4f}".format(val_loss/len(val_loader)))
print("\nTraining loop works!")
