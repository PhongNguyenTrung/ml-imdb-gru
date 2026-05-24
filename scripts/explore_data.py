"""CLI entry-point for Req 1 — load IMDB and print exploratory diagnostics.

Usage
-----
    python -m scripts.explore_data --n-samples 5

This will:
1. Download (or fetch from HF cache) the IMDB dataset.
2. Print split sizes and class balance.
3. Print N representative reviews (balanced between pos/neg).
4. Print character-length statistics.
"""

from __future__ import annotations

import argparse

from imdb_gru.data import IMDBLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDB EDA driver (Req 1).")
    parser.add_argument(
        "--n-samples", type=int, default=5, help="Number of samples to display (>=5)."
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        default="data",
        help="HF dataset cache directory (project-local by default; set 'None' to use ~/.cache).",
    )
    args = parser.parse_args()

    if args.n_samples < 5:
        raise SystemExit("Req 1 mandates at least 5 displayed samples.")

    loader = IMDBLoader(cache_dir=args.cache_dir, seed=args.seed)
    loader.summary()
    loader.show_samples(n=args.n_samples, split=args.split)
    loader.text_length_stats(split=args.split)


if __name__ == "__main__":
    main()
