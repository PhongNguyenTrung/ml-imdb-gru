"""YAML config loader with shallow merge support (Req 7 sweeps).

Experiment configs override the ``base.yaml`` keys section-by-section via
``load_config(base_path, override_path)``. We keep the merge logic
intentionally simple — a 1-level deep dict merge is enough for our schema.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive shallow-key merge: override wins; nested dicts are merged."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(loaded).__name__}")
    return loaded


def load_config(base_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    """Load a base YAML config and optionally merge an experiment override on top."""
    cfg = load_yaml(base_path)
    if override_path is not None:
        override = load_yaml(override_path)
        cfg = _deep_merge(cfg, override)
    return cfg
