from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = "scripts/distilbert_yelp"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=4)
def load(model_dir: str = DEFAULT_MODEL_DIR):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"{model_dir} not found — run scripts/distilbert_train.py first")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device()).eval()
    return tok, model


def _encode(tok, texts: list[str], dev: str):
    return tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)


def predict(texts: list[str], model_dir: str = DEFAULT_MODEL_DIR, batch_size: int = 32) -> list[int]:
    tok, model = load(model_dir)
    dev = device()
    out: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            logits = model(**_encode(tok, texts[i : i + batch_size], dev)).logits
            out.extend((torch.argmax(logits, dim=-1) + 1).cpu().tolist())
    return out


def predict_proba(texts: list[str], model_dir: str = DEFAULT_MODEL_DIR, batch_size: int = 32) -> np.ndarray:
    tok, model = load(model_dir)
    dev = device()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            logits = model(**_encode(tok, texts[i : i + batch_size], dev)).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def predict_one(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    probs = predict_proba([text], model_dir)[0]
    pred = int(np.argmax(probs)) + 1
    return {
        "stars": pred,
        "confidence": float(probs[pred - 1]),
        "probs": {int(i + 1): float(p) for i, p in enumerate(probs)},
    }
