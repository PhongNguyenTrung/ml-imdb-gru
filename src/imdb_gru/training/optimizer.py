r"""Optimizer factory (Req 4, also Req 6 via ``weight_decay``).

Why Adam for a GRU sequence model?
----------------------------------

Adam (Kingma & Ba, 2015) couples per-parameter first- and second-moment
estimates of the stochastic gradient:

.. math::
    m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
    \hat{m}_t &= m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t) \\
    \theta_t &= \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)

Two properties make it the de-facto choice for recurrent models:

1. **Adaptive per-parameter step size.** GRU gradients are highly anisotropic
   across the embedding (sparse, large fan-out), recurrent (dense, moderate
   magnitude) and output-linear (dense, small) parameter groups. Adam's
   :math:`1/\sqrt{\hat{v}_t}` normalisation matches the per-parameter scale,
   so SGD-style fixed learning rates that work for the recurrent matrices
   would either fail to update the rare embeddings or blow up the linear
   head.

2. **Bias-corrected momentum.** Time-truncated BPTT over 256-token sequences
   produces gradients with substantial variance; the EMA in :math:`m_t`
   reduces step-noise without the long warm-up that vanilla momentum
   requires.

Weight decay (Req 6 — L2 regularisation)
----------------------------------------

Standard Adam mixes weight decay into the gradient *before* the
:math:`\sqrt{\hat{v}_t}` normalisation, which couples regularisation
strength to the gradient magnitude. ``torch.optim.Adam`` with
``weight_decay > 0`` implements this classic form, sufficient for our
purposes. (For larger Transformer-scale models the decoupled ``AdamW``
variant from Loshchilov & Hutter, 2019 is preferred — see Req 10.)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Construct the optimizer named by the YAML config."""
    optimizer_name = optimizer_name.lower()
    params = list(parameters)

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer_name={optimizer_name!r}.")
