from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class Prediction:
    pred: int | None
    truth: int
    extra: dict | None = None


def valid(preds: list[Prediction]) -> list[Prediction]:
    return [p for p in preds if p.pred is not None]


def compliance(preds: list[Prediction]) -> float:
    return len(valid(preds)) / max(1, len(preds))


def accuracy(preds: list[Prediction]) -> float:
    v = valid(preds)
    if not v:
        return 0.0
    return float(accuracy_score([p.truth for p in v], [p.pred for p in v]))


def macro_f1(preds: list[Prediction], classes: list[int]) -> float:
    v = valid(preds)
    if not v:
        return 0.0
    return float(f1_score([p.truth for p in v], [p.pred for p in v], labels=classes, average="macro", zero_division=0))


def mae(preds: list[Prediction]) -> float:
    v = valid(preds)
    if not v:
        return 0.0
    return float(np.mean([abs(p.pred - p.truth) for p in v]))


def confusion(preds: list[Prediction], classes: list[int]) -> list[list[int]]:
    v = valid(preds)
    if not v:
        return [[0] * len(classes) for _ in classes]
    mat = confusion_matrix([p.truth for p in v], [p.pred for p in v], labels=classes)
    return mat.tolist()


def report(preds: list[Prediction], classes: list[int]) -> dict:
    return {
        "n": len(preds),
        "compliance": compliance(preds),
        "accuracy": accuracy(preds),
        "macro_f1": macro_f1(preds, classes),
        "mae": mae(preds),
        "confusion": confusion(preds, classes),
    }


def fmt(r: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [
        f"n={r['n']}  compliance={pct(r['compliance'])}  accuracy={pct(r['accuracy'])}  "
        f"macroF1={r['macro_f1']:.3f}  MAE={r['mae']:.3f}",
        "confusion (rows=truth, cols=pred):",
    ]
    for row in r["confusion"]:
        lines.append("  " + " ".join(f"{v:4d}" for v in row))
    return "\n".join(lines)
