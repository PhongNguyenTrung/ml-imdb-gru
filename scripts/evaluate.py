"""End-to-end test-set evaluation entry-point (Req 8 + Req 9).

Usage
-----
    python -m scripts.evaluate --run-dir artifacts/runs/base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from imdb_gru.data import IMDBLoader, RegexTokenizer, Vocabulary
from imdb_gru.data.dataset import IMDBDataset, collate_batch
from imdb_gru.evaluation import ErrorAnalyzer, Evaluator
from imdb_gru.models import GRUClassifier, GRUClassifierConfig
from imdb_gru.utils import set_seed
from imdb_gru.visualization import plot_learning_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained IMDB-GRU model on the test split.")
    p.add_argument("--run-dir", required=True, help="Run directory containing best.pt + config.json + vocab.json.")
    p.add_argument("--checkpoint-name", default="best.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k-errors", type=int, default=5, help="FP/FN samples to print per class.")
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
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    device = pick_device(args.device)

    # 1) Load test split + the *training-time* vocabulary (no leakage).
    loader = IMDBLoader()
    test_split = loader.test
    vocab = Vocabulary.load(run_dir / "vocab.json")
    tokenizer = RegexTokenizer()
    test_texts = list(test_split["text"])
    test_labels = list(test_split["label"])
    test_ds = IMDBDataset(test_texts, test_labels, tokenizer, vocab, max_len=cfg["data"]["max_len"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # 2) Reconstruct model & load weights.
    model_cfg = GRUClassifierConfig(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    )
    model = GRUClassifier(model_cfg)
    ckpt = torch.load(run_dir / args.checkpoint_name, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"[eval] loaded {args.checkpoint_name} (epoch={ckpt.get('epoch')})")

    # 3) Run evaluation.
    evaluator = Evaluator(model, device=device)
    result = evaluator.evaluate(test_loader)

    print("\n=== Classification Report (test) ===")
    print(result.report)
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1-score:  {result.f1:.4f}")

    # 4) Persist plots alongside the run artifacts (gitignored).
    figures = run_dir / "figures"
    Evaluator.plot_confusion_matrix(result, save_path=figures / "confusion_matrix.png")
    Evaluator.plot_confusion_matrix(result, save_path=figures / "confusion_matrix_normalized.png", normalize=True)
    plot_learning_curves(run_dir / "history.json", title=f"Learning Curves — {run_dir.name}",
                          save_path=figures / "learning_curves.png")

    # 5) Error analysis.
    analyzer = ErrorAnalyzer(result, test_texts)
    analyzer.print_report(n_per_class=args.top_k_errors)


if __name__ == "__main__":
    main()
