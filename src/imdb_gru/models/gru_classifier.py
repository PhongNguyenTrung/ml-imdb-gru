r"""GRU-based binary sentiment classifier (Req 3).

Architecture
------------

    input_ids ∈ ℤ^{B × T}
        │
        ▼  nn.Embedding(V, E, padding_idx=0)
    E_t ∈ ℝ^{B × T × E}
        │
        ▼  nn.GRU(E, H, num_layers=L, bidirectional, dropout)
    H_T ∈ ℝ^{B × H · num_dirs}        (last hidden state, optionally concat over directions)
        │
        ▼  nn.Dropout(p)
        ▼  nn.Linear(H · num_dirs, 1)
    logit ∈ ℝ^{B}                     (raw score; pass through BCEWithLogits)


GRU mathematics (single layer, time step ``t``, hidden size ``H``)
------------------------------------------------------------------

Let :math:`x_t \in \mathbb{R}^{E}` be the embedding at time ``t``, and
:math:`h_{t-1} \in \mathbb{R}^{H}` the previous hidden state.

Reset gate
    :math:`r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})`

Update gate
    :math:`z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})`

Candidate (new) state
    :math:`n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))`

Hidden state
    :math:`h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}`

with :math:`\sigma(\cdot)` the sigmoid and :math:`\odot` element-wise product.

The update gate :math:`z_t \in (0,1)^H` learns *per-dimension* how much past
memory to carry forward — this is the key mechanism that lets the GRU
preserve sentiment signal across the long IMDB reviews (median ≈ 175 tokens)
without the vanishing-gradient pathology of vanilla RNNs.

Parameter counts (used in the Report)
-------------------------------------

* ``Embedding``: ``V · E`` parameters (here ``10 000 · 128 = 1 280 000``).
* ``GRU`` (single layer, unidirectional): ``3 · (E · H + H · H + 2H)``
  parameters. The factor 3 covers the (r, z, n) gates, and the ``+2H`` per
  gate accounts for the input- and hidden-bias vectors. For ``E=H=128``
  this is ``3 · (128·128 + 128·128 + 2·128) = 99 072``.
* ``Linear`` head: ``(H · num_dirs + 1)`` parameters (the ``+1`` is the bias).

"Number of neurons / kernel meaning" (Req 3 explicit ask)
---------------------------------------------------------

* ``embed_dim (E)`` — dimensionality of each token's learned dense vector.
  Larger E → richer lexical semantics but more parameters and more overfit
  risk on 25 k reviews. We default to 128, a common sweet spot for IMDB.
* ``hidden_dim (H)`` — number of GRU "neurons" (the size of :math:`h_t`).
  This is the *bottleneck* through which the entire review must be summarised
  before the linear classifier sees it. Larger H → more expressive memory.
* ``num_layers (L)`` — depth of stacked GRUs. Each layer's output sequence
  feeds the next as input. L > 1 lets the network compose temporal features
  at multiple abstraction levels; here we default to 1 for parameter efficiency.
* ``Linear(H·num_dirs, 1)`` — a single output neuron producing the logit
  :math:`\log \frac{P(y=1|x)}{P(y=0|x)}`. We use 1 output (not 2) because
  binary sentiment is naturally a Bernoulli distribution; the logit feeds
  ``BCEWithLogitsLoss`` (numerically stable softplus formulation).

Note on packed sequences
------------------------

We use ``nn.utils.rnn.pack_padded_sequence`` with the true ``lengths``
tensor. This skips the recurrent computation over padded positions, which
both (a) speeds up training and (b) **prevents <pad> embeddings from
contaminating the final hidden state** — semantically critical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class GRUClassifierConfig:
    """Hyperparameters for :class:`GRUClassifier`."""

    vocab_size: int = 10_000
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3
    padding_idx: int = 0


class GRUClassifier(nn.Module):
    """``Embedding → GRU → Dropout → Linear`` binary sentiment classifier.

    The model outputs a single logit per example (no sigmoid applied), to be
    consumed by ``nn.BCEWithLogitsLoss`` — see Req 4.

    Parameters
    ----------
    config : GRUClassifierConfig
        See the dataclass above for field semantics.
    """

    def __init__(self, config: GRUClassifierConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx,
        )

        # When num_layers == 1, PyTorch ignores the GRU `dropout` arg; that's fine —
        # we apply explicit dropout after pooling the final hidden state below.
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        num_dirs = 2 if config.bidirectional else 1
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * num_dirs, 1)

        self._init_weights()

    # ------------------------------------------------------------------ init

    def _init_weights(self) -> None:
        """Xavier-uniform for the embedding and classifier; orthogonal for GRU
        recurrent weights — a standard recipe that stabilises early training."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[self.config.padding_idx].fill_(0.0)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ---------------------------------------------------------------- forward

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute a single logit per example.

        Parameters
        ----------
        input_ids : torch.LongTensor, shape (B, T)
            Padded token ids; padding value must equal ``config.padding_idx``.
        lengths : torch.LongTensor, shape (B,)
            True (pre-pad) lengths in [1, T]. Must reside on CPU (PyTorch
            requirement for ``pack_padded_sequence``).

        Returns
        -------
        logits : torch.FloatTensor, shape (B,)
            Raw, un-sigmoided scores.
        """
        # (B, T) → (B, T, E)
        emb = self.embedding(input_ids)

        # `enforce_sorted=False` lets us pass an unsorted lengths tensor.
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # h_n: (num_layers * num_dirs, B, H)
        _, h_n = self.gru(packed)

        # Bidirectional: concat last layer's forward + backward final states (B, 2H).
        # Otherwise: just the top-layer hidden state (B, H).
        last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.config.bidirectional else h_n[-1]

        last = self.dropout(last)
        logits: torch.Tensor = self.classifier(last).squeeze(-1)  # (B,)
        return logits

    # -------------------------------------------------------------- utilities

    def count_parameters(self) -> dict[str, int]:
        """Return per-module trainable-parameter counts (useful for the report)."""
        return {
            "embedding": sum(p.numel() for p in self.embedding.parameters() if p.requires_grad),
            "gru": sum(p.numel() for p in self.gru.parameters() if p.requires_grad),
            "classifier": sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
