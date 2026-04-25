"""IPA OCR FastAPI backend — TrOCR+LoRA inference + clipboard."""
import base64
import io
from threading import Lock
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pyperclip

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent.parent / "outputs_lora_data2" / "best_model"
PORT = 8765

# Validate model path at startup
if not MODEL_DIR.exists():
    raise RuntimeError(f"[Phontex] Model directory not found: {MODEL_DIR}")

# ── Model (lazy init) ────────────────────────────────────────────────────────
_model_lock = Lock()
_model = None
_processor = None


def _get_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    return "cpu"


def _load_model():
    global _model, _processor
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    from peft import PeftModel

    print(f"[Phontex] Loading model from {MODEL_DIR} ...")
    device = _get_device()
    print(f"[Phontex] Device: {device}")

    _processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    IPA_TOKENS = ["ˈ", "ˌ", "ː", "ʰ", "̃", "ʷ", "ʲ", "ɥ"]
    _processor.tokenizer.add_tokens(IPA_TOKENS)
    print(f"[Phontex] Added {len(IPA_TOKENS)} IPA tokens")

    base = VisionEncoderDecoderModel.from_pretrained("microsoft/tcroft-base-handwritten")
    _model = PeftModel.from_pretrained(base, str(MODEL_DIR))
    _model = _model.merge_and_unload()
    _model = _model.to(device)
    _model.eval()
    print(f"[Phontex] Model ready on {device}")


# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Phontex] FastAPI server started")
    yield
    print("[Phontex] FastAPI server shutting down")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://127.0.0.1:8765"], allow_methods=["*"], allow_headers=["*"]
)

# ── Pydantic schemas ─────────────────────────────────────────────────────────
class OCRRequest(BaseModel):
    image: str  # base64 data URL: "data:image/png;base64,..."


class ClipboardRequest(BaseModel):
    text: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _model is None:
        return {"status": "loading", "ready": False}
    return {"status": "ready", "ready": True}


@app.post("/ocr")
async def ocr(req: OCRRequest):
    # Lazy load model
    with _model_lock:
        if _model is None:
            _load_model()

    # Decode base64 → PIL Image
    try:
        if req.image.startswith("data:"):
            b64 = req.image.split(",", 1)[1]
        else:
            b64 = req.image
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # TrOCR inference
    device = _get_device()
    pixel_values = _processor(img, return_tensors="pt").pixel_values.to(device)
    generated_ids = _model.generate(pixel_values, max_new_tokens=128)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return {"text": text}


@app.post("/clipboard")
async def write_clipboard(req: ClipboardRequest):
    try:
        pyperclip.copy(req.text)
        return {"ok": True}
    except Exception as py_err:
        # Fallback: try tkinter
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(req.text)
            root.update()
            root.destroy()
            return {"ok": True}
        except Exception:
            raise HTTPException(status_code=500, detail=f"clipboard error (pyperclip: {py_err})")


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")