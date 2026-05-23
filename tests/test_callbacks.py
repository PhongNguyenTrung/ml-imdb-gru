"""Tests for EarlyStopping + ModelCheckpoint (Req 6)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from imdb_gru.training.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------- EarlyStopping


def test_early_stopping_triggers_after_patience() -> None:
    es = EarlyStopping(patience=2, monitor="val_loss", mode="min")
    assert not es({"val_loss": 1.0})  # baseline
    assert not es({"val_loss": 0.9})  # improved
    assert not es({"val_loss": 0.95})  # worse 1
    stop = es({"val_loss": 0.95})  # worse 2 → patience exhausted
    assert stop is True
    assert es.should_stop is True


def test_early_stopping_resets_counter_on_improvement() -> None:
    es = EarlyStopping(patience=2, monitor="val_loss", mode="min")
    es({"val_loss": 1.0})
    es({"val_loss": 0.99})  # plateau-ish
    es({"val_loss": 0.5})  # big improvement → reset counter
    assert es.counter == 0
    assert es.best == 0.5
    assert es.should_stop is False


def test_early_stopping_max_mode_uses_higher_is_better() -> None:
    es = EarlyStopping(patience=1, monitor="val_acc", mode="max")
    assert not es({"val_acc": 0.80})
    assert not es({"val_acc": 0.85})  # improved
    stop = es({"val_acc": 0.83})  # worse → patience=1 exhausted
    assert stop is True


def test_early_stopping_min_delta_filters_noise() -> None:
    es = EarlyStopping(patience=1, monitor="val_loss", mode="min", min_delta=0.05)
    es({"val_loss": 1.0})
    es({"val_loss": 0.98})  # improvement of 0.02 < min_delta → counter increments
    assert es.counter == 1


def test_early_stopping_invalid_patience() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_early_stopping_missing_monitor_key_raises() -> None:
    es = EarlyStopping(patience=2, monitor="val_loss")
    with pytest.raises(KeyError):
        es({"val_acc": 0.9})  # wrong key


# ---------------------------------------------------------------- ModelCheckpoint


def test_model_checkpoint_saves_only_on_improvement(tmp_path) -> None:
    ckpt = ModelCheckpoint(directory=tmp_path, monitor="val_loss", mode="min")
    model = nn.Linear(4, 1)

    saved1 = ckpt(epoch=1, model=model, metrics={"val_loss": 1.0})
    assert saved1 is True
    assert ckpt.best == 1.0
    assert ckpt.best_epoch == 1
    assert (tmp_path / "best.pt").exists()

    saved2 = ckpt(epoch=2, model=model, metrics={"val_loss": 0.5})  # better
    assert saved2 is True
    assert ckpt.best == 0.5
    assert ckpt.best_epoch == 2

    saved3 = ckpt(epoch=3, model=model, metrics={"val_loss": 0.7})  # worse
    assert saved3 is False
    assert ckpt.best == 0.5  # unchanged
    assert ckpt.best_epoch == 2


def test_model_checkpoint_payload_contents(tmp_path) -> None:
    ckpt = ModelCheckpoint(directory=tmp_path, monitor="val_acc", mode="max")
    model = nn.Linear(3, 1)
    ckpt(epoch=5, model=model, metrics={"val_acc": 0.9})

    payload = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 5
    assert payload["metrics"] == {"val_acc": 0.9}
    assert "state_dict" in payload
