from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def confusion_heatmap(mat: list[list[int]], classes: list[int], title: str = "", save: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = np.array(mat)
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{c}★" for c in classes],
        yticklabels=[f"{c}★" for c in classes],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("pred")
    ax.set_ylabel("truth")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120)
    return fig


def bar_compare(data: dict[str, dict[str, float]], metric: str, title: str = "", save: str | None = None):
    strategies = list(next(iter(data.values())).keys())
    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, strat in enumerate(strategies):
        vals = [data[m][strat] for m in models]
        ax.bar(x + i * width, vals, width, label=strat)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120)
    return fig
