"""Tests for IMDBDataset & collate_batch (Req 2)."""

from __future__ import annotations

import torch

from imdb_gru.data.dataset import IMDBDataset, collate_batch
from imdb_gru.data.preprocessing import RegexTokenizer
from imdb_gru.data.vocabulary import PAD_INDEX, Vocabulary


def _make_pipeline(max_len: int = 8) -> tuple[IMDBDataset, Vocabulary]:
    texts = ["this is good", "this is bad bad bad", "okay film"]
    labels = [1, 0, 1]
    tok = RegexTokenizer()
    train_tokens = [tok(t) for t in texts]
    vocab = Vocabulary(max_size=50).fit(train_tokens)
    ds = IMDBDataset(texts, labels, tok, vocab, max_len=max_len)
    return ds, vocab


def test_dataset_len_matches_inputs() -> None:
    ds, _ = _make_pipeline()
    assert len(ds) == 3


def test_padding_to_max_len() -> None:
    ds, _ = _make_pipeline(max_len=16)
    sample = ds[0]
    assert sample.input_ids.shape == (16,)
    assert sample.input_ids.dtype == torch.long
    assert sample.length == 3
    # tail must be padded with PAD_INDEX
    assert (sample.input_ids[sample.length :] == PAD_INDEX).all()


def test_truncation_to_max_len() -> None:
    ds, _ = _make_pipeline(max_len=2)
    sample = ds[1]  # "this is bad bad bad" → 5 tokens, truncated to 2
    assert sample.input_ids.shape == (2,)
    assert sample.length == 2


def test_collate_produces_expected_batch_shape() -> None:
    ds, _ = _make_pipeline(max_len=8)
    batch = collate_batch([ds[i] for i in range(len(ds))])
    assert batch["input_ids"].shape == (3, 8)
    assert batch["lengths"].shape == (3,)
    assert batch["labels"].shape == (3,)
    assert batch["labels"].dtype == torch.float32


def test_dataset_requires_fitted_vocab() -> None:
    import pytest

    vocab = Vocabulary(max_size=10)  # NOT fitted
    with pytest.raises(RuntimeError):
        IMDBDataset(["x"], [0], RegexTokenizer(), vocab)
