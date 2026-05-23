"""Test-set evaluator (Req 9).

Computes:

* The confusion matrix (TN, FP, FN, TP) — visualized via ``matplotlib``.
* The full ``sklearn.classification_report`` (precision / recall / F1 /
  support per class + macro averages).

The evaluator returns the raw predictions and probabilities so downstream
error analysis (Req 9 part 2 — FP/FN inspection) can reuse them without
re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from imdb_gru.data.loader import LABEL_NAMES


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids `exp(large_positive)` overflow.

    For z >= 0:  1 / (1 + exp(-z))   — the standard form, safe.
    For z <  0:  exp(z) / (1 + exp(z)) — rewrite, also safe.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class EvaluationResult:
    y_true: np.ndarray  # shape (N,)
    y_pred: np.ndarray  # shape (N,)
    y_proba: np.ndarray  # shape (N,) — sigmoid probabilities
    confusion: np.ndarray  # shape (2, 2)
    report: str  # formatted classification report
    accuracy: float
    precision: float
    recall: float
    f1: float


class Evaluator:
    """Compute and report metrics on a held-out ``DataLoader`` (e.g. test)."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvaluationResult:
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            lengths = batch["lengths"]
            labels = batch["labels"]
            logits = self.model(input_ids, lengths).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

        logits = torch.cat(all_logits).numpy()
        probs = _stable_sigmoid(logits)
        preds = (probs > 0.5).astype(np.int64)
        y_true = torch.cat(all_labels).numpy().astype(np.int64)

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        report = classification_report(y_true, preds, target_names=list(LABEL_NAMES), digits=4)
        acc = float((preds == y_true).mean())
        return EvaluationResult(
            y_true=y_true,
            y_pred=preds,
            y_proba=probs,
            confusion=cm,
            report=report,
            accuracy=acc,
            precision=float(precision_score(y_true, preds)),
            recall=float(recall_score(y_true, preds)),
            f1=float(f1_score(y_true, preds)),
        )

    # ---------------------------------------------------------------- plotting

    @staticmethod
    def plot_confusion_matrix(
        result: EvaluationResult,
        *,
        save_path: str | Path | None = None,
        normalize: bool = False,
        show: bool = False,
    ) -> plt.Figure:
        cm = result.confusion.astype(np.float64)
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES)
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(
            f"Confusion Matrix{' (row-normalized)' if normalize else ''}\n"
            f"acc={result.accuracy:.4f}  F1={result.f1:.4f}"
        )

        fmt = ".2f" if normalize else "d"
        threshold = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontweight="bold",
                )
        fig.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[plot] saved {save_path}")
        if show:
            plt.show()
        return fig
