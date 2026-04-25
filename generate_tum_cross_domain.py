"""Generate cross-domain test set from TUM synth labels using our rendering pipeline.

This tests our TrOCR+LoRA model's generalization to IPA strings from
TUM's character distribution (synth dataset), rendered with our pipeline.
"""
import os
import json
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 10_000          # subset for fast evaluation
OUTPUT_DIR  = Path("/home/chenhao/ipa_ocr/tum_data/cross_domain_test")
FONT_POOL  = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────────────
def load_font(size=24):
    for path in FONT_POOL:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

# ── Augmentations ─────────────────────────────────────────────────────────────
def random_brightness(img, delta=20):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-delta, delta + 1, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def random_contrast(img, factor_range=(0.85, 1.15)):
    arr = np.array(img).astype(np.float32)
    factor = random.uniform(*factor_range)
    mean = arr.mean()
    arr = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def random_blur(img, p=0.3, radius=1):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

# ── Core rendering ──────────────────────────────────────────────────────────────
def render_ipa_string(args):
    idx, ipa_str, out_fname = args
    try:
        font = load_font(random.randint(20, 28))
        # estimate width
        dummy = Image.new("RGB", (1, 1))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), ipa_str, font=font)
        w = bbox[2] - bbox[0] + random.randint(16, 32)
        h = 48 + random.randint(0, 16)
        img = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(img)
        x = random.randint(6, 14)
        y = random.randint(4, 8)
        draw.text((x, y), ipa_str, font=font, fill=0)
        # letter spacing perturbation
        # (already rendered as single text, close enough)
        img = ImageOps.invert(img.convert("RGB"))
        img = random_brightness(img, delta=random.randint(5, 15))
        if random.random() > 0.5:
            img = random_contrast(img)
        img = random_blur(img, p=0.25)
        # resize to consistent height
        target_h = 48
        ratio = target_h / img.height
        new_w = max(int(img.width * ratio), 64)
        img = img.resize((new_w, target_h), Image.LANCZOS)
        img = img.convert("RGB")
        img.save(out_fname, "PNG", quality=85)
        return True
    except Exception:
        return False

# (need ImageOps)
# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Collect all unique TUM synth labels
    import glob
    labels_map = {}
    for f in glob.glob("/home/chenhao/ipa_ocr/tum_data/synth_test_labels/**/*.gt.txt", recursive=True):
        with open(f, encoding="utf-8") as fh:
            content = fh.read().strip()
        if content and len(content) > 0:
            labels_map[content] = content  # dedup

    unique_strings = list(labels_map.values())
    random.seed(42)
    selected = random.sample(unique_strings, min(NUM_SAMPLES, len(unique_strings)))
    print(f"Selected {len(selected)} unique strings from {len(unique_strings)} total")

    tasks = []
    for i, s in enumerate(selected):
        fname = f"tum_ipa_{i:06d}.png"
        tasks.append((i, s, str(IMG_DIR / fname)))

    print(f"Rendering {len(tasks)} images...")
    os.makedirs(IMG_DIR, exist_ok=True)

    # Parallel rendering
    n_workers = min(mp.cpu_count(), 16)
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(render_ipa_string, t): t for t in tasks}
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(tasks)} done")

    success = sum(1 for r in results if r)
    print(f"Rendered {success}/{len(tasks)} images successfully")

    # Save labels.json
    labels_out = {}
    for i, s in enumerate(selected):
        fname = f"tum_ipa_{i:06d}.png"
        labels_out[fname] = s

    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(labels_out)} labels to {labels_path}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
