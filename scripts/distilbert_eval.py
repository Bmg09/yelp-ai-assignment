"""Evaluate a fine-tuned distilbert on multiple JSONL eval sets.

Usage:
    python python/distilbert_eval.py --model python/distilbert_yelp \
        --sets data/yelp_eval.jsonl data/amazon_eval.jsonl data/imdb_eval.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def predict(texts, model, tok, dev: str, batch_size: int = 32) -> list[int]:
    model.eval()
    out: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            out.extend(preds)
    return out


def eval_set(path: str, model, tok, dev: str) -> dict:
    rows = load_jsonl(path)
    texts = [r["text"] for r in rows]
    truth = np.array([r["stars"] - 1 for r in rows])
    preds = np.array(predict(texts, model, tok, dev))

    classes = sorted(set(truth.tolist()))
    return {
        "n": len(rows),
        "classes": [c + 1 for c in classes],
        "accuracy": float(accuracy_score(truth, preds)),
        "macro_f1": float(f1_score(truth, preds, average="macro", labels=classes)),
        "mae": float(mean_absolute_error(truth, preds)),
        "confusion": confusion_matrix(truth, preds, labels=list(range(5))).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", default="results/04b-distilbert.json")
    args = ap.parse_args()

    dev = device()
    print(f"device={dev}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev)

    results: dict[str, dict] = {}
    for p in args.sets:
        name = Path(p).stem
        print(f"\n--- {name} ---")
        r = eval_set(p, model, tok, dev)
        print(json.dumps({k: v for k, v in r.items() if k != "confusion"}, indent=2))
        results[name] = r

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
