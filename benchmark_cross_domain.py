"""Evaluate TrOCR+LoRA on TUM cross-domain test set."""
import json
import editdistance
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import PeftModel


DATA_DIR  = Path("/home/chenhao/ipa_ocr/tum_data/cross_domain_test")
MODEL_DIR = Path("/home/chenhao/ipa_ocr/outputs_lora_data2/best_model")
N_SAMPLES = 10_000


def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def main():
    with open(DATA_DIR / "labels.json") as f:
        labels = json.load(f)

    print("Loading TrOCR+LoRA model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    IPA_TOKENS = ["ˈ", "ˌ", "ː", "ʰ", "̃", "ʷ", "ʲ", "ɥ"]
    processor.tokenizer.add_tokens(IPA_TOKENS)
    print(f"[Benchmark] Added {len(IPA_TOKENS)} IPA tokens")

    HF_CACHE = "/mnt/models/chenhao/huggingface/hub/models--microsoft--trocr-base-handwritten/snapshots/eaacaf452b06415df8f10bb6fad3a4c11e609406"
    base = VisionEncoderDecoderModel.from_pretrained(HF_CACHE)
    model = PeftModel.from_pretrained(base, str(MODEL_DIR))
    model = model.merge_and_unload()
    model = model.to("cuda:1")
    model.eval()

    image_files = sorted((DATA_DIR / "images").glob("*.png"))[:N_SAMPLES]
    results = []
    total_dist, total_len = 0, 0

    for img_path in tqdm(image_files, desc="TrOCR+LoRA (cross-domain)"):
        fname = img_path.name
        if fname not in labels:
            continue
        ref = labels[fname]
        img = Image.open(img_path).convert("RGB")
        pv = processor(img, return_tensors="pt").pixel_values.to("cuda:1")
        gen = model.generate(pv, max_new_tokens=128)
        hyp = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        dist = cer(ref, hyp)
        total_dist += dist
        total_len += len(ref)
        results.append({"file": fname, "ref": ref, "pred": hyp, "dist": dist})

    cer_score = total_dist / max(total_len, 1)
    print(f"\nTrOCR+LoRA (cross-domain) CER: {cer_score:.4f}")
    print(f"Samples: {len(results)}")

    out_path = Path("/home/chenhao/ipa_ocr/outputs_benchmark/cross_domain_cer.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"cer": cer_score, "n": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()