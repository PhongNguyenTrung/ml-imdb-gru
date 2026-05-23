r"""Theoretical extension: Self-Attention on top of GRU, plus a Transformer
Encoder reference implementation (Req 10).

Motivation — the sequentiality bottleneck of GRU
------------------------------------------------

The recurrent update :math:`h_t = (1-z_t) \odot n_t + z_t \odot h_{t-1}` is
**inherently serial** in ``t``. Two consequences:

1. **Computational** — for a sequence of length ``T``, the wall-clock depth
   of the forward pass is :math:`\Theta(T)`; no GPU parallelism helps.
2. **Modelling** — the path that information at position ``i`` must travel
   to influence the prediction has length :math:`T-i`, so very long-range
   dependencies are attenuated even with gating (the "diminishing gradient
   over depth" effect).

Self-Attention (Vaswani et al., 2017) replaces the recurrence with a
**single global pooling step** that is parallelisable over positions and
exposes every token-pair interaction directly. Concretely:

.. math::
    \text{Attention}(Q, K, V) =
        \operatorname{softmax}\!\Bigl(\frac{Q K^{\top}}{\sqrt{d_k}}\Bigr) V

where for each head :math:`h`:

* :math:`Q = X W_Q^{(h)}`, :math:`K = X W_K^{(h)}`, :math:`V = X W_V^{(h)}`
  are query / key / value projections of the input :math:`X \in \mathbb{R}^{T \times d}`.
* :math:`\sqrt{d_k}` is the scaling that prevents the softmax saturating
  when :math:`d_k` is large (its expected dot-product variance grows
  linearly in :math:`d_k`).
* Multiple heads :math:`h = 1, \dots, H` are concatenated and re-projected:
  :math:`\text{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H) W_O`.

Replacing or augmenting the GRU with self-attention therefore:

* drops the sequential depth from :math:`O(T)` to :math:`O(1)` (parallel),
* gives **direct** :math:`O(1)` connectivity between any pair of positions,
* trades the :math:`O(T \cdot d^2)` recurrent cost for an :math:`O(T^2 \cdot d)`
  cost — manageable at :math:`T = 256`.

Two upgrade paths are provided below
------------------------------------

* :class:`GRUAttentionClassifier` — keeps the GRU as a contextualiser but
  pools its outputs with a *learned additive attention* (Bahdanau-style):
  a minimal, mathematically grounded upgrade.
* :class:`TransformerEncoderClassifier` — replaces the GRU outright with a
  stack of standard Transformer encoder blocks. Provided as a *reference
  skeleton* for the Req-10 discussion; not trained in the main pipeline.

Both classes share the same I/O contract as :class:`GRUClassifier` so they
slot into the existing training loop without changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from imdb_gru.models.gru_classifier import GRUClassifierConfig

# =============================================================================
# 1) GRU + Additive Self-Attention pooling
# =============================================================================


class AdditiveAttentionPooling(nn.Module):
    r"""Single-head additive attention over a sequence (Bahdanau, 2015).

    Given hidden states :math:`h_1, \dots, h_T \in \mathbb{R}^{H}`:

    .. math::
        e_t &= v^{\top} \tanh(W h_t)  \\
        \alpha_t &= \frac{\exp(e_t)}{\sum_{t'} \exp(e_{t'})} \\
        c &= \sum_t \alpha_t \, h_t

    The pooled context :math:`c` replaces the GRU's last hidden state and
    captures *which positions mattered* for the classification (the
    :math:`\alpha_t` are interpretable saliency weights).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs
        ------
        hidden_states : (B, T, H)
        mask : bool (B, T)  -- True at valid positions, False at <pad>.

        Returns
        -------
        context : (B, H)
        attn_weights : (B, T)
        """
        scores = self.v(torch.tanh(self.w(hidden_states))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.einsum("bt,bth->bh", attn, hidden_states)
        return context, attn


class GRUAttentionClassifier(nn.Module):
    """GRU contextualiser + additive-attention pooling + linear head.

    This is the *minimal* concrete upgrade that addresses the Req-10 critique
    of pure-GRU sequentiality: positional weighting is learned rather than
    fixed to "use the last hidden state", which is particularly helpful for
    long reviews where the sentiment cue may not appear at the end.
    """

    def __init__(self, config: GRUClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx,
        )
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.attention = AdditiveAttentionPooling(out_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hiddens, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))

        mask = (input_ids != self.config.padding_idx)
        context, _ = self.attention(hiddens, mask)
        logits = self.classifier(self.dropout(context)).squeeze(-1)
        return logits


# =============================================================================
# 2) Transformer Encoder skeleton (reference)
# =============================================================================


@dataclass
class TransformerEncoderConfig:
    vocab_size: int = 10_000
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_len: int = 256
    padding_idx: int = 0


class SinusoidalPositionalEncoding(nn.Module):
    r"""Standard fixed sin/cos positional encoding (Vaswani et al., 2017).

    .. math::
        \mathrm{PE}_{(pos, 2i)}   &= \sin(pos / 10000^{2i / d_{\text{model}}}) \\
        \mathrm{PE}_{(pos, 2i+1)} &= \cos(pos / 10000^{2i / d_{\text{model}}})
    """

    def __init__(self, embed_dim: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderClassifier(nn.Module):
    """Reference Transformer-encoder skeleton — not trained in the main run.

    Architecture: ``Embedding + PosEnc → N × TransformerEncoderLayer → mean-pool → Linear``.

    Provided so the Req-10 report section is grounded in runnable PyTorch
    code, not just equations.
    """

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx,
        )
        self.pos_enc = SinusoidalPositionalEncoding(config.embed_dim, config.max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.embed_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths  # encoder uses key_padding_mask, not lengths
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        key_padding_mask = (input_ids == self.config.padding_idx)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Masked mean pool — ignore <pad> positions.
        valid = (~key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)
