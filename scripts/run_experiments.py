"""Run the 3 Req-7 hyperparameter experiments sequentially.

Usage
-----
    python -m scripts.run_experiments

Each experiment override is merged on top of ``configs/base.yaml`` and writes
its artifacts to ``artifacts/runs/<experiment_name>/``. A combined summary is
printed at the end (best val_acc / val_loss across all 3 runs).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Resolve repo root from this file so the script works no matter where it's invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
EXPERIMENTS = [
    REPO_ROOT / "configs" / "exp_lr_high.yaml",
    REPO_ROOT / "configs" / "exp_lr_low.yaml",
    REPO_ROOT / "configs" / "exp_bs_large.yaml",
]
DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "runs"


def run_one(override_path: Path) -> None:
    print(f"\n{'=' * 70}\n>>> Running experiment: {override_path.name}\n{'=' * 70}")
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", str(BASE_CONFIG),
        "--override", str(override_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def summarise(log_dir: Path = DEFAULT_LOG_DIR) -> None:
    print(f"\n{'=' * 70}\n>>> Summary across experiments\n{'=' * 70}")
    print(f"{'experiment':<24} {'best_val_loss':>14} {'best_val_acc':>14} {'epochs':>8}")
    print("-" * 64)
    for hist_path in sorted(log_dir.glob("*/history.json")):
        name = hist_path.parent.name
        history = json.loads(hist_path.read_text(encoding="utf-8"))["epochs"]
        if not history:
            continue
        best_loss = min(e["val_loss"] for e in history)
        best_acc = max(e["val_acc"] for e in history)
        print(f"{name:<24} {best_loss:>14.4f} {best_acc:>14.4f} {len(history):>8}")


def main() -> None:
    for cfg in EXPERIMENTS:
        run_one(cfg)
    summarise()


if __name__ == "__main__":
    main()
