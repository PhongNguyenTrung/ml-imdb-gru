r"""Loss factory for the IMDB-GRU project (Req 4).

Why ``BCEWithLogitsLoss`` (and not ``CrossEntropyLoss``)?
--------------------------------------------------------

IMDB sentiment is a *binary* classification problem; ``y ∈ {0, 1}`` follows
a Bernoulli distribution :math:`y \mid x \sim \text{Bernoulli}(p(x))`. The
correct negative log-likelihood is

.. math::
    \mathcal{L}_{\text{BCE}}(z, y) = -\bigl[y \cdot \log \sigma(z)
                                          + (1-y) \cdot \log(1-\sigma(z))\bigr]

with :math:`z` the raw logit and :math:`\sigma(z) = 1/(1+e^{-z})`.

PyTorch's ``BCEWithLogitsLoss`` computes this via the numerically stable
softplus formulation

.. math::
    \mathcal{L}_{\text{BCE}}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})

which never invokes ``log(0)`` even when :math:`\sigma(z)` saturates to 0 or 1.

Equivalence to ``CrossEntropyLoss``
-----------------------------------
For two classes, applying ``softmax`` to a 2-logit vector and then NLL is
*mathematically equivalent* to ``BCEWithLogitsLoss`` on a single logit
:math:`z = z_1 - z_0`. But the single-logit form has half the head
parameters, so we adopt it as the canonical Req 4 choice.

Class imbalance handling (optional)
-----------------------------------
The IMDB train set is perfectly balanced (12 500 pos / 12 500 neg) so the
default ``pos_weight=None`` suffices. The argument is exposed nonetheless,
because Req-7 sweeps or future extensions on imbalanced corpora may need it.
"""

from __future__ import annotations

import torch
from torch import nn


def build_loss(
    loss_name: str = "bce_with_logits",
    *,
    pos_weight: float | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Return the loss module specified in the YAML config.

    Parameters
    ----------
    loss_name : {"bce_with_logits", "cross_entropy"}
        Selects the loss function. We default to ``bce_with_logits`` for the
        single-logit GRU classifier.
    pos_weight : float | None
        Optional weighting of the positive class for BCE; useful only on
        imbalanced corpora.
    label_smoothing : float
        Only honored when ``loss_name == "cross_entropy"``.
    """
    loss_name = loss_name.lower()

    if loss_name == "bce_with_logits":
        pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    if loss_name == "cross_entropy":
        # Requires the model to output 2 logits — keep for completeness only.
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    raise ValueError(f"Unknown loss_name={loss_name!r}.")
