"""Benchmark Qwen2.5-VL-3B on IPA OCR val set."""
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import editdistance


DATA_DIR = Path("/home/chenhao/ipa_ocr/data2/val")
OUTPUT = Path("/home/chenhao/ipa_ocr/outputs_benchmark/qwen25vl_cer.json")
N_SAMPLES = 200  # fast benchmark

def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(len(ref), 1)

def main():
    OUTPUT.parent.mkdir(exist_ok=True)

    with open(DATA_DIR / "labels.json") as f:
        labels = json.load(f)

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    # Build messages template (image + prompt)
    def make_prompt():
        return [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe the IPA text in this image exactly."}
            ]}
        ]

    results = []
    total_dist = 0
    total_len = 0

    image_files = sorted((DATA_DIR / "images").glob("*.png"))[:N_SAMPLES]

    for img_path in tqdm(image_files, desc="Qwen2.5-VL"):
        fname = img_path.name
        if fname not in labels:
            continue
        ref = labels[fname]

        img = Image.open(img_path).convert("RGB")
        msgs = make_prompt()
        text = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        decoded = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        dist = cer(ref, decoded)
        total_dist += dist
        total_len += len(ref)
        results.append({"file": fname, "ref": ref, "pred": decoded, "dist": dist})

    cer_score = total_dist / max(total_len, 1)
    print(f"\nQwen2.5-VL-3B Zero-shot CER: {cer_score:.4f} ({total_dist}/{total_len})")
    print(f"Samples: {len(results)}")

    with open(OUTPUT, "w") as f:
        json.dump({"cer": cer_score, "n": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
