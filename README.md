# IMDB Sentiment Analysis with GRU

> A PyTorch R&D project implementing a Gated Recurrent Unit (GRU) classifier for binary sentiment analysis on the IMDB Reviews dataset.

[![CI](https://github.com/PhongNguyenTrung/ml-imdb-gru/actions/workflows/ci.yml/badge.svg)](https://github.com/PhongNguyenTrung/ml-imdb-gru/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

This repository contains the implementation of a sentiment classifier on the IMDB Reviews
dataset using a **GRU (Gated Recurrent Unit)** architecture in PyTorch. The project is
designed as an academic R&D deliverable.

The codebase follows an **object-oriented, modular layout** suitable for reproducible
machine-learning research.

## Status

✅ **All 10 Requirements implemented.** The reference notebook
[`SourceCode.ipynb`](SourceCode.ipynb) walks through every requirement and
serves as the executable submission deliverable (run on Google Colab or
local Jupyter to capture output cells).

## Project Structure

```
ml-imdb-gru/
├── SourceCode.ipynb        # Submission notebook — runs end-to-end on Colab/Jupyter
├── configs/                # YAML configs (base + 3 Req-7 hyperparameter overrides)
├── src/imdb_gru/           # Importable Python package
│   ├── data/               # Loader, preprocessing, vocabulary, dataset    (Req 1, 2)
│   ├── models/             # GRU classifier + attention extension          (Req 3, 10)
│   ├── training/           # Trainer, losses, optimizer, callbacks         (Req 4, 5, 6)
│   ├── evaluation/         # Evaluator + FP/FN error analysis              (Req 9)
│   ├── visualization/      # Learning-curve & comparison plots             (Req 8)
│   └── utils/              # Seeding, config loader
├── scripts/                # CLI entry points (explore_data, train, run_experiments, evaluate)
├── tests/                  # pytest — preprocessing, vocabulary, dataset, model, trainer, evaluator
├── pyproject.toml          # Build + tooling (ruff, black, mypy, pytest)
├── .github/                # CI workflow + Dependabot + auto-merge
└── .pre-commit-config.yaml
```

## Requirements Coverage

| # | Topic | Implementation | Status |
|---|-------|----------------|--------|
| 1 | Data loading & EDA (≥5 samples) | [`data/loader.py`](src/imdb_gru/data/loader.py), [`scripts/explore_data.py`](scripts/explore_data.py) | ✅ |
| 2 | Tokenization + Vocabulary(10k) + Pad/Trunc(256), leakage-safe | [`data/preprocessing.py`](src/imdb_gru/data/preprocessing.py), [`data/vocabulary.py`](src/imdb_gru/data/vocabulary.py), [`data/dataset.py`](src/imdb_gru/data/dataset.py) | ✅ |
| 3 | `Embedding → GRU → Linear` + GRU mathematics + parameter count | [`models/gru_classifier.py`](src/imdb_gru/models/gru_classifier.py) | ✅ |
| 4 | `BCEWithLogitsLoss` + `Adam` with mathematical rationale | [`training/losses.py`](src/imdb_gru/training/losses.py), [`training/optimizer.py`](src/imdb_gru/training/optimizer.py) | ✅ |
| 5 | Explicit train/val loop, per-epoch Loss & Accuracy logging | [`training/trainer.py`](src/imdb_gru/training/trainer.py), [`training/metrics.py`](src/imdb_gru/training/metrics.py) | ✅ |
| 6 | Dropout + Weight Decay (L2) + Early Stopping | [`models/gru_classifier.py`](src/imdb_gru/models/gru_classifier.py), [`training/optimizer.py`](src/imdb_gru/training/optimizer.py), [`training/callbacks.py`](src/imdb_gru/training/callbacks.py) | ✅ |
| 7 | 3 hyperparameter experiments (LR high / LR low / BS large) | [`configs/exp_*.yaml`](configs/), [`scripts/run_experiments.py`](scripts/run_experiments.py) | ✅ |
| 8 | Learning curve plots (Train vs Val, Loss + Acc) + cross-experiment overlay | [`visualization/plots.py`](src/imdb_gru/visualization/plots.py) | ✅ |
| 9 | Confusion matrix + classification_report (P/R/F1) + FP/FN error analysis | [`evaluation/evaluator.py`](src/imdb_gru/evaluation/evaluator.py), [`evaluation/error_analysis.py`](src/imdb_gru/evaluation/error_analysis.py) | ✅ |
| 10| Self-Attention + Transformer Encoder extension (theory + working skeleton) | [`models/attention_extension.py`](src/imdb_gru/models/attention_extension.py) | ✅ |

## Installation

```bash
# Clone
git clone https://github.com/PhongNguyenTrung/ml-imdb-gru.git
cd ml-imdb-gru

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install (editable mode with dev extras)
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Usage

All four CLIs are functional. The submission notebook [`SourceCode.ipynb`](SourceCode.ipynb) calls the same Python API end-to-end.

```bash
# Req 1 — load IMDB and show 5+ sample reviews + length statistics
python -m scripts.explore_data --n-samples 5

# Req 2–6 — train the baseline GRU classifier on the IMDB train split
python -m scripts.train --config configs/base.yaml

# Req 7 — run the 3 hyperparameter experiments sequentially (LR high / LR low / BS large)
python -m scripts.run_experiments

# Req 8–9 — load a trained checkpoint, evaluate on the test split, plot CM + learning curves,
#           and print FP / FN error analysis
python -m scripts.evaluate --run-dir artifacts/runs/base
```

Artifacts (checkpoints, history, vocab, TensorBoard logs) are written to `artifacts/runs/<experiment_name>/`.

## Development

```bash
# Format & lint
ruff check --fix .
black .

# Run tests
pytest

# Type-check
mypy src/
```

## License

[MIT](LICENSE)

## Citation

If you use this codebase in academic work, please cite:

```bibtex
@misc{nguyen2026imdbgru,
  author       = {Nguyen Trung Phong},
  title        = {IMDB Sentiment Analysis with GRU: A PyTorch R\&D Study},
  year         = {2026},
  howpublished = {\url{https://github.com/PhongNguyenTrung/ml-imdb-gru}}
}
```
