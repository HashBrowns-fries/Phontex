"""Benchmark TUM Calamari on TUM cross-domain test set."""
import json
import editdistance
from pathlib import Path
from tqdm import tqdm
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams

DATA_DIR  = Path("/home/chenhao/ipa_ocr/tum_data/cross_domain_test")
MODEL_PATH = "/home/chenhao/ipa_ocr/ocr-ipa/model/calamari/best.ckpt"
OUTPUT    = Path("/home/chenhao/ipa_ocr/outputs_benchmark/calamari_xd_cer.json")
N_SAMPLES = 200

def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


with open(DATA_DIR / "labels.json") as f:
    labels = json.load(f)

print("Loading TUM Calamari...")
predictor = MultiPredictor.from_paths(checkpoints=[str(MODEL_PATH)], auto_update_checkpoints=False)

image_files = sorted((DATA_DIR / "images").glob("*.png"))[:N_SAMPLES]
batch_size = 64
results = []
total_dist, total_len = 0, 0

for i in tqdm(range(0, len(image_files), batch_size), desc="Calamari (cross-domain)"):
    batch = image_files[i:i+batch_size]
    params = FileDataParams(images=[str(p) for p in batch])
    preds_raw = predictor.predict(params)
    for img_path, pred_obj in zip(batch, preds_raw):
        fname = img_path.name
        if fname not in labels:
            continue
        ref = labels[fname]
        hyp = pred_obj.outputs[0][0].sentence
        d = cer(ref, hyp)
        total_dist += d
        total_len += len(ref)
        results.append({"file":fname,"ref":ref,"pred":hyp,"dist":d})

cer_score = total_dist / max(total_len, 1)
print(f"\nTUM Calamari (cross-domain) CER: {cer_score:.4f}")
print(f"Samples: {len(results)}")
with open(OUTPUT, "w") as f:
    json.dump({"cer": cer_score, "n": len(results), "results": results}, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUTPUT}")
