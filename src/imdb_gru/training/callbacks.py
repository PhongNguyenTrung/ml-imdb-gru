"""Training callbacks: Early Stopping + Model Checkpoint (Req 6).

Regularization summary
----------------------
The full Req-6 regularization stack used in this project:

1. **Dropout** — applied (a) inside the GRU stack between layers when
   ``num_layers > 1`` and (b) on the pooled final hidden state before the
   linear head. See ``GRUClassifier``.
2. **Weight decay (L2)** — added as an extra term :math:`\\lambda \\|\\theta\\|_2^2`
   to the Adam update. See ``build_optimizer``.
3. **Early stopping** — implemented here; halts training when the chosen
   validation metric has not improved for ``patience`` consecutive epochs.

Why early stopping is *regularization* in the statistical-learning sense:
running ``epochs → ∞`` would let the network memorise the train set; the
patience mechanism implicitly bounds the optimisation horizon to the best
empirical risk on held-out val data, producing the same bias-variance
trade-off that an explicit penalty on model complexity would yield
(Goodfellow et al., 2016 — §7.8).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn


class EarlyStopping:
    """Stop training when ``monitor`` hasn't improved for ``patience`` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for an improvement before stopping.
    monitor : str
        Key into the metrics dict passed to ``__call__``. Typical: ``"val_loss"``.
    mode : {"min", "max"}
        Whether improvement means *decrease* (loss) or *increase* (accuracy).
    min_delta : float
        Minimum absolute change to count as improvement.
    """

    def __init__(
        self,
        patience: int,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best: float = math.inf if mode == "min" else -math.inf
        self.counter: int = 0
        self.should_stop: bool = False

    def _improved(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def __call__(self, metrics: dict[str, float]) -> bool:
        if self.monitor not in metrics:
            raise KeyError(f"Monitor key {self.monitor!r} not found in metrics: {list(metrics)}")
        current = float(metrics[self.monitor])
        if self._improved(current):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Persist the model whenever the monitored metric improves."""

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        filename: str = "best.pt",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.path = self.directory / filename
        self.best: float = math.inf if mode == "min" else -math.inf
        self.best_epoch: int | None = None

    def _improved(self, current: float) -> bool:
        return current < self.best if self.mode == "min" else current > self.best

    def __call__(self, *, epoch: int, model: nn.Module, metrics: dict[str, float]) -> bool:
        current = float(metrics[self.monitor])
        if self._improved(current):
            self.best = current
            self.best_epoch = epoch
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics},
                self.path,
            )
            print(f"[checkpoint] saved {self.path} ({self.monitor}={current:.4f}, epoch={epoch})")
            return True
        return False
