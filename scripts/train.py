"""End-to-end training entry-point.

Loads a YAML config, builds the data module + model + loss + optimizer +
trainer, then fits and persists artifacts to ``configs.logging.log_dir``.

Usage
-----
    python -m scripts.train --config configs/base.yaml
    python -m scripts.train --config configs/base.yaml --override configs/exp_lr_high.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from imdb_gru.data import IMDBLoader, build_dataloaders
from imdb_gru.models import GRUClassifier, GRUClassifierConfig
from imdb_gru.training import Trainer, TrainerConfig, build_loss, build_optimizer
from imdb_gru.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the IMDB-GRU classifier.")
    p.add_argument("--config", required=True, help="Path to a base YAML config.")
    p.add_argument(
        "--override",
        default=None,
        help="Optional path to an experiment override YAML (deep-merged on top of --config).",
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect by default).")
    return p.parse_args()


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device)

    # ----- Data ------------------------------------------------------------
    print("[train] loading IMDB dataset...")
    loader = IMDBLoader(seed=cfg.get("seed", 42))
    train_split, test_split = loader.train, loader.test

    dm = build_dataloaders(
        train_split,
        test_split,
        val_ratio=cfg["data"]["val_split"],
        max_len=cfg["data"]["max_len"],
        vocab_size=cfg["data"]["vocab_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg.get("seed", 42),
    )
    print(f"[train] vocab size = {len(dm.vocabulary)}")
    print(
        f"[train] batches: train={len(dm.train_loader)} val={len(dm.val_loader)} "
        f"test={len(dm.test_loader)}"
    )

    # ----- Model -----------------------------------------------------------
    model_cfg = GRUClassifierConfig(
        vocab_size=len(dm.vocabulary),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    )
    model = GRUClassifier(model_cfg)
    print(f"[train] model params: {model.count_parameters()}")

    # ----- Loss / Optimizer -----------------------------------------------
    loss_fn = build_loss(cfg["training"]["loss"])
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=cfg["training"]["optimizer"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ----- Trainer ---------------------------------------------------------
    trainer_cfg = TrainerConfig(
        epochs=cfg["training"]["epochs"],
        grad_clip=cfg["training"].get("grad_clip"),
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        log_dir=cfg["logging"]["log_dir"],
        experiment_name=cfg["logging"]["experiment_name"],
        early_stopping_patience=cfg["training"].get("early_stopping_patience"),
    )
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer, config=trainer_cfg, device=device
    )

    print(f"[train] device={device} | experiment={trainer_cfg.experiment_name}")
    trainer.fit(dm.train_loader, dm.val_loader)

    # Persist run config + vocabulary alongside the model.
    run_dir = Path(trainer_cfg.log_dir) / trainer_cfg.experiment_name
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    dm.vocabulary.save(run_dir / "vocab.json")
    print(f"[train] artifacts saved to {run_dir}")
    print(
        f"[train] best metric so far: epoch={trainer.checkpoint.best_epoch if trainer.checkpoint else 'n/a'}"
    )


if __name__ == "__main__":
    main()
