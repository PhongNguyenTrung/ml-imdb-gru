"""Tests for the GRU classifier (Req 3)."""

from __future__ import annotations

import torch

from imdb_gru.models import GRUClassifier, GRUClassifierConfig


def _make_model(**kwargs) -> GRUClassifier:
    cfg = GRUClassifierConfig(vocab_size=100, embed_dim=16, hidden_dim=16, **kwargs)
    return GRUClassifier(cfg)


def test_forward_shape_unidirectional() -> None:
    model = _make_model()
    input_ids = torch.randint(low=0, high=100, size=(4, 20))
    lengths = torch.tensor([20, 15, 10, 5], dtype=torch.long)
    logits = model(input_ids, lengths)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_forward_shape_bidirectional() -> None:
    model = _make_model(bidirectional=True)
    input_ids = torch.randint(low=0, high=100, size=(2, 8))
    lengths = torch.tensor([8, 5], dtype=torch.long)
    logits = model(input_ids, lengths)
    assert logits.shape == (2,)


def test_forward_shape_multilayer() -> None:
    model = _make_model(num_layers=2)
    input_ids = torch.randint(low=0, high=100, size=(3, 10))
    lengths = torch.tensor([10, 7, 3], dtype=torch.long)
    logits = model(input_ids, lengths)
    assert logits.shape == (3,)


def test_padding_does_not_affect_output() -> None:
    """The forward pass on a packed sequence must ignore <pad> tokens.

    We compare logits for the same review with different amounts of trailing
    padding; outputs should be identical up to numerical noise.
    """
    torch.manual_seed(0)
    model = _make_model().eval()
    # Same first 5 tokens, but row 0 has 3 extra <pad> at the end.
    ids_short = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0]])
    ids_long = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0]])
    lengths = torch.tensor([5], dtype=torch.long)

    with torch.no_grad():
        out_short = model(ids_short, lengths)
        out_long = model(ids_long, lengths)
    assert torch.allclose(out_short, out_long, atol=1e-6)


def test_parameter_count_includes_all_modules() -> None:
    model = _make_model()
    counts = model.count_parameters()
    assert counts["embedding"] > 0
    assert counts["gru"] > 0
    assert counts["classifier"] > 0
    assert counts["total"] == counts["embedding"] + counts["gru"] + counts["classifier"]


def test_padding_embedding_stays_zero_after_init() -> None:
    model = _make_model()
    pad_vec = model.embedding.weight[model.config.padding_idx]
    assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))


def test_backward_runs() -> None:
    model = _make_model()
    input_ids = torch.randint(low=0, high=100, size=(2, 6))
    lengths = torch.tensor([6, 4], dtype=torch.long)
    logits = model(input_ids, lengths)
    loss = logits.sum()
    loss.backward()
    # at least one grad on the embedding (excluding pad row).
    assert model.embedding.weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert model.classifier.weight.grad is not None
