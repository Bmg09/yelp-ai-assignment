"""Fine-tune distilbert on Yelp (1-5 stars) using M4 MPS backend.

Usage:
    python python/distilbert_train.py --n 10000 --epochs 3
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def stratified_sample(ds, n_per_class: int, label_field: str, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in range(n_classes)}
    for row in ds:
        by_class[row[label_field]].append(row)
    out = []
    for c in range(n_classes):
        idx = rng.choice(len(by_class[c]), size=min(n_per_class, len(by_class[c])), replace=False)
        out.extend([by_class[c][i] for i in idx])
    rng.shuffle(out)
    return Dataset.from_list(out)


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="train sample per class x 5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out", default="python/distilbert_yelp")
    args = ap.parse_args()

    seed_all(42)
    print(f"device={device()}")

    print("loading yelp_review_full ...")
    yelp = load_dataset("Yelp/yelp_review_full")

    per_class = args.n // 5
    train = stratified_sample(yelp["train"], per_class, "label", 5, 42)
    val = stratified_sample(yelp["test"], 100, "label", 5, 7)
    print(f"train n={len(train)}  val n={len(val)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(batch):
        return tok(batch["text"], truncation=True, padding=False, max_length=256)

    train = train.map(enc, batched=True, remove_columns=["text"])
    val = val.map(enc, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=5)

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "mae": mean_absolute_error(labels, preds),
        }

    tr_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        bf16=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
