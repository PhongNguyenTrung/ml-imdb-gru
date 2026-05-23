"""Lightweight metric utilities (Req 5)."""

from __future__ import annotations

import torch


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return mean classification accuracy for single-logit binary outputs.

    A logit > 0 corresponds to :math:`\\sigma(z) > 0.5`, i.e. predicted positive.
    """
    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")
    preds = (logits > 0.0).to(labels.dtype)
    correct = (preds == labels).float().sum().item()
    total = labels.numel()
    return correct / max(total, 1)


class RunningMean:
    """Numerically stable streaming mean (Welford-lite)."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0

    def update(self, value: float, weight: int = 1) -> None:
        if weight <= 0:
            return
        self._n += weight
        # incremental mean: m_n = m_{n-1} + w * (value - m_{n-1}) / n
        self._mean += weight * (value - self._mean) / self._n

    @property
    def value(self) -> float:
        return self._mean

    @property
    def count(self) -> int:
        return self._n

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
