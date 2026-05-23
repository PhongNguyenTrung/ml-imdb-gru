"""Learning-curve plots (Req 8).

Functions
---------
- ``plot_learning_curves``: side-by-side Loss & Accuracy curves (train vs val)
  for a single experiment's training history.
- ``plot_experiment_comparison``: overlay Loss/Accuracy curves of multiple
  experiments — used by Req 7's reporting.

History format expected
-----------------------
The ``Trainer`` writes ``history.json`` with shape::

    {"epochs": [{"epoch": 1, "train_loss": ..., "train_acc": ...,
                 "val_loss": ..., "val_acc": ..., "duration_sec": ...}, ...]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt


def _load_history(history_or_path) -> list[dict]:
    if isinstance(history_or_path, (str, Path)):
        payload = json.loads(Path(history_or_path).read_text(encoding="utf-8"))
        return payload["epochs"]
    if isinstance(history_or_path, dict) and "epochs" in history_or_path:
        return history_or_path["epochs"]
    raise TypeError("history must be a path or a dict with key 'epochs'")


def plot_learning_curves(
    history_or_path,
    *,
    title: str = "Learning Curves",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot Loss and Accuracy curves (Train vs Val) for a single run.

    Returns the ``matplotlib.figure.Figure`` for further customisation.
    """
    epochs_data = _load_history(history_or_path)
    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_acc = [e["train_acc"] for e in epochs_data]
    val_acc = [e["val_acc"] for e in epochs_data]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, train_loss, marker="o", label="Train", linewidth=1.8)
    ax_loss.plot(epochs, val_loss, marker="s", label="Validation", linewidth=1.8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (BCE)")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend()

    ax_acc.plot(epochs, train_acc, marker="o", label="Train", linewidth=1.8)
    ax_acc.plot(epochs, val_acc, marker="s", label="Validation", linewidth=1.8)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {save_path}")
    if show:
        plt.show()
    return fig


def plot_experiment_comparison(
    histories: Iterable[tuple[str, str | Path]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Overlay validation curves from multiple experiments (Req 7 reporting).

    Parameters
    ----------
    histories : iterable of (label, path)
        Each entry is a ``(display_label, path_to_history_json)`` pair.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, path in histories:
        epochs_data = _load_history(path)
        if not epochs_data:
            continue
        epochs = [e["epoch"] for e in epochs_data]
        ax_loss.plot(epochs, [e["val_loss"] for e in epochs_data], marker="o", label=label, linewidth=1.6)
        ax_acc.plot(epochs, [e["val_acc"] for e in epochs_data], marker="o", label=label, linewidth=1.6)

    ax_loss.set_title("Validation Loss across experiments")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend()

    ax_acc.set_title("Validation Accuracy across experiments")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {save_path}")
    if show:
        plt.show()
    return fig
