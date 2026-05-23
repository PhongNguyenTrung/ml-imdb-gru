"""IMDB Reviews loader (Req 1).

Loads the IMDB binary-sentiment dataset via the HuggingFace ``datasets`` library
and provides utilities for inspecting raw samples (Exploratory Data Analysis).

Dataset reference
-----------------
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
*Learning Word Vectors for Sentiment Analysis*. ACL 2011.

The HF mirror provides:

* ``train``: 25,000 reviews (balanced 12,500 pos / 12,500 neg)
* ``test``:  25,000 reviews (same balance)
* features: ``{"text": str, "label": ClassLabel(0=neg, 1=pos)}``
"""

from __future__ import annotations

import random
import statistics
import textwrap
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_DATASET_NAME = "imdb"
LABEL_NAMES: tuple[str, str] = ("negative", "positive")


@dataclass(frozen=True)
class RawSample:
    """Lightweight container for a single raw IMDB review."""

    index: int
    text: str
    label: int

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]


class IMDBLoader:
    """High-level loader for the IMDB Reviews corpus.

    Parameters
    ----------
    dataset_name : str
        HF dataset identifier. Defaults to ``"imdb"``.
    cache_dir : str | None
        Optional cache directory for HF datasets. ``None`` uses the HF default.
    seed : int
        Seed for any random sampling performed by EDA helpers.

    Examples
    --------
    >>> loader = IMDBLoader()
    >>> loader.summary()        # prints split sizes and label distribution
    >>> loader.show_samples(5)  # prints 5 representative reviews
    """

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        cache_dir: str | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.seed = seed
        self._dataset: DatasetDict | None = None

    # ------------------------------------------------------------------ load

    def load(self) -> DatasetDict:
        """Load (or return cached) IMDB ``DatasetDict`` with ``train``/``test``."""
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        assert isinstance(self._dataset, DatasetDict)
        return self._dataset

    @property
    def train(self) -> Dataset:
        return self.load()["train"]

    @property
    def test(self) -> Dataset:
        return self.load()["test"]

    # ----------------------------------------------------------- inspection

    def summary(self) -> dict[str, dict[str, int]]:
        """Return per-split sizes and class distribution; also prints a table."""
        ds = self.load()
        out: dict[str, dict[str, int]] = {}
        print(f"\n=== IMDB Dataset Summary ({self.dataset_name}) ===")
        print(f"{'split':<8} {'size':>8} {'neg':>8} {'pos':>8}")
        print("-" * 36)
        for split_name, split in ds.items():
            labels = split["label"]
            n_neg = sum(1 for y in labels if y == 0)
            n_pos = sum(1 for y in labels if y == 1)
            out[split_name] = {"size": len(split), "negative": n_neg, "positive": n_pos}
            print(f"{split_name:<8} {len(split):>8} {n_neg:>8} {n_pos:>8}")
        print()
        return out

    def show_samples(
        self,
        n: int = 5,
        split: str = "train",
        balanced: bool = True,
        max_chars: int = 400,
    ) -> list[RawSample]:
        """Print ``n`` raw review samples and return them as ``RawSample`` objects.

        Parameters
        ----------
        n : int
            Number of samples to show. The Req requires at least 5.
        split : {"train", "test"}
            Which split to draw from.
        balanced : bool
            If True, half the samples are drawn from each class so the user
            inspects both polarities.
        max_chars : int
            Truncate the printed text to this many characters for readability.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        ds = self.load()[split]
        rng = random.Random(self.seed)

        if balanced:
            pos_idx = [i for i, y in enumerate(ds["label"]) if y == 1]
            neg_idx = [i for i, y in enumerate(ds["label"]) if y == 0]
            half = n // 2
            picks = rng.sample(pos_idx, half) + rng.sample(neg_idx, n - half)
            rng.shuffle(picks)
        else:
            picks = rng.sample(range(len(ds)), n)

        samples: list[RawSample] = []
        print(f"\n=== {n} sample reviews from `{split}` split ===\n")
        for rank, idx in enumerate(picks, start=1):
            row = ds[int(idx)]
            sample = RawSample(index=int(idx), text=row["text"], label=int(row["label"]))
            samples.append(sample)
            preview = sample.text[:max_chars]
            ellipsis = "..." if len(sample.text) > max_chars else ""
            wrapped = textwrap.fill(preview + ellipsis, width=100, subsequent_indent="    ")
            print(f"[{rank}] idx={sample.index}  label={sample.label} ({sample.label_name})")
            print(f"    len={len(sample.text)} chars")
            print(f"    text: {wrapped}\n")
        return samples

    def text_length_stats(self, split: str = "train") -> dict[str, float]:
        """Return character-length statistics for a split (min/median/mean/max)."""
        ds = self.load()[split]
        lengths = [len(t) for t in ds["text"]]
        stats = {
            "min": float(min(lengths)),
            "median": float(statistics.median(lengths)),
            "mean": float(statistics.mean(lengths)),
            "max": float(max(lengths)),
            "stdev": float(statistics.stdev(lengths)),
        }
        print(f"\n=== Character-length stats for `{split}` split ===")
        for k, v in stats.items():
            print(f"  {k:>7}: {v:,.1f}")
        print()
        return stats
