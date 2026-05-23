"""Torch ``Dataset`` and ``DataLoader`` factory for IMDB (Req 2).

Pipeline
--------
raw text → ``RegexTokenizer`` → ``Vocabulary.encode`` → pad/truncate to
``max_len`` → ``torch.LongTensor``.

The encoding step happens **lazily** in ``__getitem__`` to keep memory
overhead small (≈ holding the raw strings in RAM only). For 25 k reviews
this is fine; the bottleneck is the GPU, not tokenization.

Data-leakage guard
------------------
``build_dataloaders`` is the canonical entry point. It:

1. Splits the HF ``train`` into ``train``/``val`` deterministically.
2. Tokenizes only the training texts.
3. Fits the ``Vocabulary`` exclusively on those training tokens.
4. Wraps train/val/test in ``IMDBDataset`` instances that share the *same
   fitted* vocabulary, so val/test never influence the mapping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from imdb_gru.data.preprocessing import RegexTokenizer
from imdb_gru.data.vocabulary import PAD_INDEX, Vocabulary

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset


@dataclass
class EncodedSample:
    input_ids: torch.Tensor  # shape (max_len,), dtype long
    length: int              # true (pre-pad) length, ∈ [1, max_len]
    label: int               # 0 or 1


class IMDBDataset(Dataset[EncodedSample]):
    """Torch dataset over an iterable of (text, label) pairs.

    Parameters
    ----------
    texts : list[str]
        Raw review texts.
    labels : list[int]
        Binary labels (0 or 1) aligned with ``texts``.
    tokenizer : RegexTokenizer
        Shared tokenizer instance.
    vocabulary : Vocabulary
        Already-fitted vocabulary (fitted on train tokens only).
    max_len : int
        Pad/truncate length. Req 2 mandates ``256``.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: RegexTokenizer,
        vocabulary: Vocabulary,
        max_len: int = 256,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if max_len < 1:
            raise ValueError("max_len must be >= 1.")
        if not vocabulary.is_fitted:
            raise RuntimeError("Vocabulary must be fitted before constructing IMDBDataset.")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> EncodedSample:
        tokens = self.tokenizer.tokenize(self.texts[idx])
        ids = self.vocabulary.encode(tokens)

        # Truncate then pad to fixed max_len.
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        true_len = max(1, len(ids))  # avoid 0-length (would break packed RNN)
        if len(ids) < self.max_len:
            ids = ids + [PAD_INDEX] * (self.max_len - len(ids))

        return EncodedSample(
            input_ids=torch.tensor(ids, dtype=torch.long),
            length=true_len,
            label=int(self.labels[idx]),
        )


def collate_batch(batch: Iterable[EncodedSample]) -> dict[str, torch.Tensor]:
    """Stack ``EncodedSample`` objects into a batch dict ready for the model."""
    batch = list(batch)
    input_ids = torch.stack([b.input_ids for b in batch], dim=0)
    lengths = torch.tensor([b.length for b in batch], dtype=torch.long)
    labels = torch.tensor([b.label for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


# ---------------------------------------------------------------------------
# DataLoader factory (handles the leakage-safe split + vocab fitting)
# ---------------------------------------------------------------------------


@dataclass
class IMDBDataModule:
    """Bundle of train/val/test DataLoaders + the fitted vocabulary."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocabulary: Vocabulary
    tokenizer: RegexTokenizer


def build_dataloaders(
    train_split: "HFDataset",
    test_split: "HFDataset",
    *,
    val_ratio: float = 0.1,
    max_len: int = 256,
    vocab_size: int = 10_000,
    min_freq: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> IMDBDataModule:
    """Create train/val/test DataLoaders with a leakage-safe vocabulary.

    The HF ``train_split`` (25 000 reviews) is partitioned deterministically
    into a model-train partition (``1 - val_ratio``) and a held-out validation
    partition. The ``Vocabulary`` is fitted *only* on the model-train partition;
    val and test are encoded with that fitted vocab and so cannot leak.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie in (0, 1).")

    # 1) Pull raw texts/labels into Python lists (cheap; ~50 k strings total).
    train_texts: list[str] = list(train_split["text"])
    train_labels: list[int] = list(train_split["label"])
    test_texts: list[str] = list(test_split["text"])
    test_labels: list[int] = list(test_split["label"])

    # 2) Deterministic shuffle + split for train/val.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_texts), generator=g).tolist()
    cutoff = int(len(train_texts) * (1.0 - val_ratio))
    train_idx, val_idx = perm[:cutoff], perm[cutoff:]

    tr_texts = [train_texts[i] for i in train_idx]
    tr_labels = [train_labels[i] for i in train_idx]
    val_texts = [train_texts[i] for i in val_idx]
    val_labels = [train_labels[i] for i in val_idx]

    # 3) Tokenize train ONLY for vocab fitting (no leakage).
    tokenizer = RegexTokenizer()
    train_tokens = tokenizer.tokenize_batch(tr_texts)
    vocab = Vocabulary(max_size=vocab_size, min_freq=min_freq).fit(train_tokens)

    # 4) Wrap each split in IMDBDataset sharing the fitted vocab.
    train_ds = IMDBDataset(tr_texts, tr_labels, tokenizer, vocab, max_len=max_len)
    val_ds = IMDBDataset(val_texts, val_labels, tokenizer, vocab, max_len=max_len)
    test_ds = IMDBDataset(test_texts, test_labels, tokenizer, vocab, max_len=max_len)

    # Default `pin_memory=True` on CUDA hosts; False on CPU/MPS to avoid the
    # "Cannot pin 'torch.cuda.LongTensor' only dense CPU tensors..." warning.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return IMDBDataModule(
        train_loader=DataLoader(
            train_ds, batch_size=batch_size, num_workers=num_workers,
            collate_fn=collate_batch, shuffle=True, pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            val_ds, batch_size=batch_size, num_workers=num_workers,
            collate_fn=collate_batch, shuffle=False, pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            test_ds, batch_size=batch_size, num_workers=num_workers,
            collate_fn=collate_batch, shuffle=False, pin_memory=pin_memory,
        ),
        vocabulary=vocab,
        tokenizer=tokenizer,
    )
