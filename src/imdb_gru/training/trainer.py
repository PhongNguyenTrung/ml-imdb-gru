"""Trainer (Req 5 + Req 6 integration).

Implements an explicit train / validate loop that:

* Iterates ``epochs`` epochs over the train ``DataLoader``.
* Tracks running mean Loss and Accuracy on TRAIN within each epoch.
* Runs a full validation pass at the end of each epoch and tracks the same
  metrics on VAL.
* Persists per-epoch logs as a JSON ``history.json`` plus per-step tensorboard
  scalars (best-effort — disabled silently if TB is unavailable).
* Hands the history off to the ``EarlyStopping`` callback (Req 6) and the
  ``ModelCheckpoint`` callback for best-model bookkeeping.

The training step is intentionally written *explicitly* (no fancy
``pytorch-lightning`` wrappers) per the Req 5 brief:

    "Viết vòng lặp huấn luyện tường minh, tính toán và lưu log (Loss, Accuracy)
     của cả tập Train và Validation sau mỗi epoch."
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from imdb_gru.training.callbacks import EarlyStopping, ModelCheckpoint
from imdb_gru.training.metrics import RunningMean, binary_accuracy_from_logits


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    duration_sec: float


@dataclass
class TrainerConfig:
    epochs: int = 10
    grad_clip: float | None = 1.0
    log_every_n_steps: int = 50
    log_dir: str = "artifacts/runs"
    experiment_name: str = "base"
    save_best: bool = True
    monitor: str = "val_loss"  # "val_loss" (minimise) or "val_acc" (maximise)
    early_stopping_patience: int | None = 3

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            # Treat 0 (or any non-positive value) as "early stopping disabled".
            self.early_stopping_patience = None
        if self.monitor not in {"val_loss", "val_acc"}:
            raise ValueError(f"monitor must be 'val_loss' or 'val_acc', got {self.monitor!r}")


@dataclass
class TrainingHistory:
    epochs: list[EpochLog] = field(default_factory=list)

    def append(self, log: EpochLog) -> None:
        self.epochs.append(log)

    def to_dict(self) -> dict:
        return {"epochs": [asdict(e) for e in self.epochs]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class Trainer:
    """Explicit train/val loop with logging, checkpointing, and early-stopping."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        self.run_dir = Path(config.log_dir) / config.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = (
            ModelCheckpoint(
                directory=self.run_dir,
                monitor=config.monitor,
                mode="min" if config.monitor.endswith("loss") else "max",
            )
            if config.save_best
            else None
        )
        self.early_stopping = (
            EarlyStopping(
                patience=config.early_stopping_patience,
                monitor=config.monitor,
                mode="min" if config.monitor.endswith("loss") else "max",
            )
            if config.early_stopping_patience is not None
            else None
        )

        self._tb_writer = self._maybe_create_tb_writer()

    # ----------------------------------------------------------------- public

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        history = TrainingHistory()
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = self._run_one_epoch(train_loader, training=True, epoch=epoch)
            val_loss, val_acc = self._run_one_epoch(val_loader, training=False, epoch=epoch)
            dur = time.perf_counter() - t0

            log = EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                duration_sec=dur,
            )
            history.append(log)
            self._log_epoch(log)
            history.save(self.run_dir / "history.json")

            metrics = {"val_loss": val_loss, "val_acc": val_acc}

            if self.checkpoint is not None:
                self.checkpoint(epoch=epoch, model=self.model, metrics=metrics)

            if self.early_stopping is not None and self.early_stopping(metrics):
                print(
                    f"[trainer] Early stopping at epoch {epoch} "
                    f"(no improvement in {self.config.monitor} for "
                    f"{self.early_stopping.patience} epochs)."
                )
                break

        if self._tb_writer is not None:
            self._tb_writer.close()
        return history

    # ---------------------------------------------------------------- internals

    def _run_one_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        epoch: int,
    ) -> tuple[float, float]:
        self.model.train(mode=training)
        loss_meter, acc_meter = RunningMean(), RunningMean()
        step = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                lengths = batch["lengths"]  # stays on CPU for pack_padded_sequence
                labels = batch["labels"].to(self.device, non_blocking=True)

                logits = self.model(input_ids, lengths)
                loss = self.loss_fn(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()

                bsz = labels.size(0)
                loss_meter.update(loss.item(), weight=bsz)
                acc_meter.update(binary_accuracy_from_logits(logits.detach(), labels), weight=bsz)
                step += 1

                if (
                    training
                    and self._tb_writer is not None
                    and step % self.config.log_every_n_steps == 0
                ):
                    global_step = (epoch - 1) * len(loader) + step
                    self._tb_writer.add_scalar("train/loss_step", loss.item(), global_step)

        return loss_meter.value, acc_meter.value

    def _log_epoch(self, log: EpochLog) -> None:
        print(
            f"[epoch {log.epoch:02d}] "
            f"train_loss={log.train_loss:.4f} train_acc={log.train_acc:.4f} | "
            f"val_loss={log.val_loss:.4f} val_acc={log.val_acc:.4f} | "
            f"{log.duration_sec:.1f}s"
        )
        if self._tb_writer is not None:
            w = self._tb_writer
            w.add_scalar("train/loss", log.train_loss, log.epoch)
            w.add_scalar("train/acc", log.train_acc, log.epoch)
            w.add_scalar("val/loss", log.val_loss, log.epoch)
            w.add_scalar("val/acc", log.val_acc, log.epoch)

    def _maybe_create_tb_writer(self) -> Any | None:
        """Return a TensorBoard SummaryWriter, or None if TB is unavailable.

        Imported lazily because ``tensorboard`` is an optional dep — the rest
        of the trainer must still work if it's missing or fails to initialise.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.run_dir / "tb"))
        except Exception:  # pragma: no cover — TB missing or init failed
            return None
