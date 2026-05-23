"""Smoke tests for the Trainer (Req 5).

We exercise the training loop on a tiny synthetic dataset so the test runs in
a few seconds on CPU — proves the train/val/checkpoint/early-stop control
flow is wired correctly without depending on the real IMDB download.
"""

from __future__ import annotations

import json

import pytest
import torch
from imdb_gru.models import GRUClassifier, GRUClassifierConfig
from imdb_gru.training import Trainer, TrainerConfig, build_loss, build_optimizer
from torch.utils.data import DataLoader, TensorDataset


def _synthetic_dataloaders(
    *,
    vocab_size: int = 50,
    seq_len: int = 16,
    n_train: int = 40,
    n_val: int = 16,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Generate a tiny but learnable signal: label = (mean(token_id) > vocab_size/2)."""
    torch.manual_seed(0)

    def _make(n: int) -> DataLoader:
        ids = torch.randint(low=2, high=vocab_size, size=(n, seq_len))
        labels = (ids.float().mean(dim=1) > (vocab_size / 2)).float()
        lengths = torch.full((n,), seq_len, dtype=torch.long)

        def collate(batch):
            xs, ys, ls = zip(*batch, strict=True)
            return {
                "input_ids": torch.stack(xs),
                "labels": torch.stack(ys),
                "lengths": torch.stack(ls),
            }

        ds = TensorDataset(ids, labels, lengths)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return _make(n_train), _make(n_val)


def _make_model(vocab_size: int = 50) -> GRUClassifier:
    cfg = GRUClassifierConfig(vocab_size=vocab_size, embed_dim=8, hidden_dim=8, dropout=0.0)
    return GRUClassifier(cfg)


# --------------------------------------------------------------------- tests


def test_trainer_fit_runs_and_persists_history(tmp_path) -> None:
    train_loader, val_loader = _synthetic_dataloaders()
    model = _make_model()
    cfg = TrainerConfig(
        epochs=2,
        log_dir=str(tmp_path),
        experiment_name="smoke",
        early_stopping_patience=None,
        log_every_n_steps=999,  # silence step-level TB logging
    )
    trainer = Trainer(
        model=model,
        loss_fn=build_loss("bce_with_logits"),
        optimizer=build_optimizer(model.parameters(), lr=1e-2),
        config=cfg,
        device="cpu",
    )
    history = trainer.fit(train_loader, val_loader)

    assert len(history.epochs) == 2
    for log in history.epochs:
        assert log.train_loss > 0
        assert 0.0 <= log.train_acc <= 1.0
        assert 0.0 <= log.val_acc <= 1.0

    hist_path = tmp_path / "smoke" / "history.json"
    assert hist_path.exists()
    payload = json.loads(hist_path.read_text())
    assert len(payload["epochs"]) == 2


def test_trainer_creates_checkpoint(tmp_path) -> None:
    train_loader, val_loader = _synthetic_dataloaders()
    model = _make_model()
    trainer = Trainer(
        model=model,
        loss_fn=build_loss("bce_with_logits"),
        optimizer=build_optimizer(model.parameters(), lr=1e-2),
        config=TrainerConfig(
            epochs=2,
            log_dir=str(tmp_path),
            experiment_name="ckpt_smoke",
            early_stopping_patience=None,
        ),
        device="cpu",
    )
    trainer.fit(train_loader, val_loader)
    assert (tmp_path / "ckpt_smoke" / "best.pt").exists()
    assert trainer.checkpoint is not None
    assert trainer.checkpoint.best_epoch in (1, 2)


def test_trainer_early_stops_on_plateau(tmp_path) -> None:
    """Use lr=0 so val_loss never improves → early stopping must fire by epoch ~ patience+1."""
    train_loader, val_loader = _synthetic_dataloaders()
    model = _make_model()
    trainer = Trainer(
        model=model,
        loss_fn=build_loss("bce_with_logits"),
        optimizer=build_optimizer(model.parameters(), lr=0.0),  # nothing learns
        config=TrainerConfig(
            epochs=10,
            log_dir=str(tmp_path),
            experiment_name="es_smoke",
            early_stopping_patience=2,
        ),
        device="cpu",
    )
    history = trainer.fit(train_loader, val_loader)
    # Patience=2, baseline epoch 1, fail-counters at 2 & 3 → stop after epoch 3.
    assert len(history.epochs) <= 4


def test_trainer_config_treats_zero_patience_as_disabled() -> None:
    cfg = TrainerConfig(early_stopping_patience=0)
    assert cfg.early_stopping_patience is None


def test_trainer_config_rejects_bad_monitor() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(monitor="bogus")


def test_trainer_config_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(epochs=0)
