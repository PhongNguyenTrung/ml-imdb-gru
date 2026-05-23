"""Tests for the Vocabulary class (Req 2)."""

from __future__ import annotations

import pytest
from imdb_gru.data.vocabulary import (
    PAD_INDEX,
    PAD_TOKEN,
    UNK_INDEX,
    UNK_TOKEN,
    Vocabulary,
)


def test_specials_reserved_at_fixed_indices() -> None:
    vocab = Vocabulary(max_size=10).fit([["a", "b", "c"]])
    assert vocab.index_to_token[PAD_INDEX] == PAD_TOKEN
    assert vocab.index_to_token[UNK_INDEX] == UNK_TOKEN


def test_size_capped_to_max_size() -> None:
    """Asking for max_size=5 keeps <pad>, <unk>, and the 3 most-frequent tokens."""
    corpus = [["a"] * 5, ["b"] * 4, ["c"] * 3, ["d"] * 2, ["e"] * 1]
    vocab = Vocabulary(max_size=5).fit(corpus)
    assert len(vocab) == 5
    assert set(vocab.token_to_index) == {PAD_TOKEN, UNK_TOKEN, "a", "b", "c"}


def test_unknown_tokens_map_to_unk() -> None:
    vocab = Vocabulary(max_size=5).fit([["hello", "world"]])
    ids = vocab.encode(["hello", "rare_word", "world"])
    assert ids[0] != UNK_INDEX
    assert ids[1] == UNK_INDEX
    assert ids[2] != UNK_INDEX


def test_min_freq_filter() -> None:
    corpus = [["a"] * 5, ["b"] * 5, ["c"] * 1]
    vocab = Vocabulary(max_size=100, min_freq=2).fit(corpus)
    assert "a" in vocab.token_to_index
    assert "b" in vocab.token_to_index
    assert "c" not in vocab.token_to_index


def test_encode_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        Vocabulary().encode(["a"])


def test_refit_raises() -> None:
    vocab = Vocabulary(max_size=10).fit([["a", "b"]])
    with pytest.raises(RuntimeError):
        vocab.fit([["c", "d"]])


def test_max_size_too_small_raises() -> None:
    with pytest.raises(ValueError):
        Vocabulary(max_size=1)


def test_save_and_load_roundtrip(tmp_path) -> None:
    vocab = Vocabulary(max_size=50, min_freq=1).fit([["hello", "world"], ["hello", "again"]])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.token_to_index == vocab.token_to_index
    assert loaded.encode(["hello", "world"]) == vocab.encode(["hello", "world"])


def test_no_leakage_unseen_token_in_val_maps_to_unk() -> None:
    """If a token appears in val/test but not train, it MUST map to <unk>."""
    train_corpus = [["good", "movie"], ["bad", "film"]]
    vocab = Vocabulary(max_size=100).fit(train_corpus)
    # Val/test token "fantastic" never appeared during fit.
    ids = vocab.encode(["good", "fantastic", "movie"])
    assert ids[0] != UNK_INDEX  # "good" was in train
    assert ids[1] == UNK_INDEX  # "fantastic" never seen during fit → leakage-safe
    assert ids[2] != UNK_INDEX
