"""Benchmark our TrOCR+LoRA model on IPA val set."""
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import PeftModel
import editdistance


DATA_DIR = Path("/home/chenhao/ipa_ocr/data2/val")
MODEL_DIR = Path("/home/chenhao/ipa_ocr/outputs_lora_data2/best_model")
OUTPUT = Path("/home/chenhao/ipa_ocr/outputs_benchmark/lora_cer.json")
N_SAMPLES = 500


def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def main():
    OUTPUT.parent.mkdir(exist_ok=True)

    with open(DATA_DIR / "labels.json") as f:
        labels = json.load(f)

    print("Loading TrOCR+LoRA model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    IPA_TOKENS = ["ˈ", "ˌ", "ː", "ʰ", "̃", "ʷ", "ʲ", "ɥ"]
    processor.tokenizer.add_tokens(IPA_TOKENS)
    print(f"[Benchmark] Added {len(IPA_TOKENS)} IPA tokens")
    base = VisionEncoderDecoderModel.from_pretrained("microsoft/tcroft-base-handwritten")
    model = PeftModel.from_pretrained(base, str(MODEL_DIR))
    model = model.merge_and_unload()
    model = model.to("cuda:1")
    model.eval()

    image_files = sorted((DATA_DIR / "images").glob("*.png"))[:N_SAMPLES]

    results = []
    total_dist = 0
    total_len = 0

    for img_path in tqdm(image_files, desc="TrOCR+LoRA"):
        fname = img_path.name
        if fname not in labels:
            continue
        ref = labels[fname]

        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to("cuda:1")
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
        hyp = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        dist = cer(ref, hyp)
        total_dist += dist
        total_len += len(ref)
        results.append({"file": fname, "ref": ref, "pred": hyp, "dist": dist})

    cer_score = total_dist / max(total_len, 1)
    print(f"\nTrOCR+LoRA CER: {cer_score:.4f} ({total_dist}/{total_len})")
    print(f"Samples: {len(results)}")

    with open(OUTPUT, "w") as f:
        json.dump({"cer": cer_score, "n": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()