"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    Parameters
    ----------
    seed : int
    deterministic : bool
        If True, force CuDNN to deterministic algorithms — slower but
        repeatable. Off by default to keep training fast.

    Notes
    -----
    Setting ``PYTHONHASHSEED`` after process start does NOT affect the current
    interpreter's hash randomisation (the hash seed is read at startup). It is
    propagated to any subprocesses we spawn (e.g. via ``run_experiments``), so
    fixing it is still useful for *reproducibility across child processes*.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
