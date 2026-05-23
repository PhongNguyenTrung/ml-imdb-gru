"""False-Positive / False-Negative error analysis (Req 9, part 2).

The Req brief asks for an *analysis* of the model's mistakes — not just
counts. This module surfaces the actual misclassified reviews so the
researcher can characterise WHICH inputs trip the GRU, e.g.:

* sarcasm / mixed-sentiment reviews → FP/FN of "negative" predictions;
* very long reviews → degraded due to fixed 256-token truncation;
* short reviews with strong but isolated cue words → over-confident.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np

from imdb_gru.data.loader import LABEL_NAMES
from imdb_gru.evaluation.evaluator import EvaluationResult


@dataclass
class MisclassifiedSample:
    index: int          # position in the test set
    text: str
    true_label: int
    predicted_label: int
    probability: float  # sigmoid prob of class 1

    @property
    def error_type(self) -> str:
        if self.true_label == 0 and self.predicted_label == 1:
            return "False Positive (pred=positive, true=negative)"
        if self.true_label == 1 and self.predicted_label == 0:
            return "False Negative (pred=negative, true=positive)"
        return "Correct"


class ErrorAnalyzer:
    """Surface and pretty-print misclassified reviews from a test pass."""

    def __init__(self, result: EvaluationResult, texts: list[str]) -> None:
        if len(texts) != len(result.y_true):
            raise ValueError("`texts` length must match the number of evaluation samples.")
        self.result = result
        self.texts = texts

    def collect_errors(
        self,
        kind: str = "both",
        *,
        top_k: int | None = None,
        by_confidence: bool = True,
    ) -> list[MisclassifiedSample]:
        """Return misclassified samples.

        Parameters
        ----------
        kind : {"fp", "fn", "both"}
        top_k : int | None
            If given, return only the ``top_k`` most-confident errors
            (largest distance from the 0.5 decision boundary).
        by_confidence : bool
            If True, sort errors by descending confidence (model was *most
            wrong*); otherwise sort by index.
        """
        y_true = self.result.y_true
        y_pred = self.result.y_pred
        probs = self.result.y_proba

        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        if kind == "fp":
            mask = fp_mask
        elif kind == "fn":
            mask = fn_mask
        elif kind == "both":
            mask = fp_mask | fn_mask
        else:
            raise ValueError("kind must be 'fp', 'fn', or 'both'.")

        idxs = np.where(mask)[0]
        if by_confidence:
            idxs = sorted(idxs, key=lambda i: -abs(probs[i] - 0.5))
        if top_k is not None:
            idxs = idxs[:top_k]

        return [
            MisclassifiedSample(
                index=int(i),
                text=self.texts[int(i)],
                true_label=int(y_true[i]),
                predicted_label=int(y_pred[i]),
                probability=float(probs[i]),
            )
            for i in idxs
        ]

    def print_report(
        self,
        *,
        n_per_class: int = 5,
        max_chars: int = 500,
    ) -> dict[str, list[MisclassifiedSample]]:
        """Print the ``n_per_class`` most-confident FP and FN samples."""
        fps = self.collect_errors("fp", top_k=n_per_class)
        fns = self.collect_errors("fn", top_k=n_per_class)

        n_total = len(self.result.y_true)
        n_errors = int((self.result.y_pred != self.result.y_true).sum())
        print(f"\n=== Error Analysis ===")
        print(f"Test samples: {n_total}")
        print(f"Total errors: {n_errors}  (error rate = {n_errors / n_total:.2%})")
        print(f"False Positives (true={LABEL_NAMES[0]}, pred={LABEL_NAMES[1]}): {self.result.confusion[0, 1]}")
        print(f"False Negatives (true={LABEL_NAMES[1]}, pred={LABEL_NAMES[0]}): {self.result.confusion[1, 0]}")

        self._print_group("Top False Positives (most-confident wrongs)", fps, max_chars)
        self._print_group("Top False Negatives (most-confident wrongs)", fns, max_chars)
        return {"fp": fps, "fn": fns}

    @staticmethod
    def _print_group(title: str, samples: list[MisclassifiedSample], max_chars: int) -> None:
        print(f"\n--- {title} ---")
        if not samples:
            print("  (none)")
            return
        for rank, s in enumerate(samples, start=1):
            preview = s.text[:max_chars]
            ellipsis = "..." if len(s.text) > max_chars else ""
            wrapped = textwrap.fill(preview + ellipsis, width=100, subsequent_indent="    ")
            print(
                f"[{rank}] idx={s.index}  true={LABEL_NAMES[s.true_label]}  "
                f"pred={LABEL_NAMES[s.predicted_label]}  p(pos)={s.probability:.4f}"
            )
            print(f"    text: {wrapped}\n")
