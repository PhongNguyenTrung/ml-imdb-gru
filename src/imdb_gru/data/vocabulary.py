"""Vocabulary (Req 2).

Builds a token-to-index mapping from a corpus, with two reserved special
tokens:

* ``<pad>`` (index 0): padding for variable-length sequences. The GRU model
  zeros-out the embedding for this index via ``nn.Embedding(padding_idx=0)``,
  so the recurrent update treats padded positions as no-op inputs.
* ``<unk>`` (index 1): out-of-vocabulary token for any word not seen during
  ``fit``, or trimmed by the ``max_size`` cap.

Critical invariant for Req 2 — *no data leakage*
------------------------------------------------
``Vocabulary.fit`` MUST be called on the **training tokens only**. The
validation and test sets are then encoded by ``encode`` using the fitted
mapping; any unseen tokens map to ``<unk>``. This guarantees that no
statistical information from val/test ever enters the model's input
representation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1


class Vocabulary:
    """Frequency-thresholded vocabulary with reserved ``<pad>`` and ``<unk>``.

    Parameters
    ----------
    max_size : int
        Hard cap on the total vocabulary size *including* the two specials.
        Per the Req 2 brief, the default is **10 000**.
    min_freq : int
        Minimum corpus frequency for a token to be included.

    Attributes
    ----------
    token_to_index : dict[str, int]
    index_to_token : list[str]
    """

    def __init__(self, max_size: int = 10_000, min_freq: int = 1) -> None:
        if max_size < 2:
            raise ValueError("max_size must be >= 2 (must fit <pad> and <unk>).")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1.")
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_index: dict[str, int] = {}
        self.index_to_token: list[str] = []
        self._fitted: bool = False

    # -------------------------------------------------------------- properties

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def pad_index(self) -> int:
        return PAD_INDEX

    @property
    def unk_index(self) -> int:
        return UNK_INDEX

    # --------------------------------------------------------------------- fit

    def fit(self, tokenized_corpus: Iterable[Iterable[str]]) -> Vocabulary:
        """Fit the vocabulary on an iterable of tokenized documents.

        IMPORTANT — must be called on TRAIN tokens only to avoid data leakage
        into the held-out validation/test partitions.
        """
        if self._fitted:
            raise RuntimeError("Vocabulary already fitted; create a new instance to refit.")
        counter: Counter[str] = Counter()
        for tokens in tokenized_corpus:
            counter.update(tokens)

        # Reserve specials.
        self.index_to_token = [PAD_TOKEN, UNK_TOKEN]
        budget = self.max_size - 2  # remaining slots after specials

        # most_common returns deterministically by descending frequency.
        for token, freq in counter.most_common():
            if budget <= 0:
                break
            if freq < self.min_freq:
                # Robust against future changes to iteration order: skip rather
                # than rely on monotone non-increasing freq from most_common().
                continue
            if token in (PAD_TOKEN, UNK_TOKEN):
                continue
            self.index_to_token.append(token)
            budget -= 1

        self.token_to_index = {t: i for i, t in enumerate(self.index_to_token)}
        self._fitted = True
        return self

    # ----------------------------------------------------------------- encode

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map an iterable of tokens to integer ids; unseen → ``<unk>``."""
        if not self._fitted:
            raise RuntimeError("Vocabulary must be fitted before encoding.")
        unk = UNK_INDEX
        t2i = self.token_to_index
        return [t2i.get(tok, unk) for tok in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Vocabulary must be fitted before decoding.")
        n = len(self.index_to_token)
        return [self.index_to_token[i] if 0 <= i < n else UNK_TOKEN for i in indices]

    # ------------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "index_to_token": self.index_to_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(max_size=payload["max_size"], min_freq=payload["min_freq"])
        vocab.index_to_token = list(payload["index_to_token"])
        vocab.token_to_index = {t: i for i, t in enumerate(vocab.index_to_token)}
        vocab._fitted = True
        return vocab
