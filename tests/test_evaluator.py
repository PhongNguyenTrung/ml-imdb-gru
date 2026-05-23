"""Tests for Evaluator + ErrorAnalyzer (Req 9)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from imdb_gru.evaluation import ErrorAnalyzer, Evaluator
from imdb_gru.evaluation.evaluator import EvaluationResult, _stable_sigmoid


class _DummyModel(nn.Module):
    """Returns a fixed logit per example, regardless of input — for deterministic tests."""

    def __init__(self, fixed_logits: torch.Tensor) -> None:
        super().__init__()
        self._fixed = fixed_logits
        # so .to(device) calls do something
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self._fixed[: input_ids.size(0)].to(input_ids.device)


def _make_loader(input_ids, labels) -> DataLoader:
    lengths = torch.full((len(labels),), input_ids.size(1), dtype=torch.long)

    def collate(batch):
        xs, ys, ls = zip(*batch)
        return {"input_ids": torch.stack(xs), "labels": torch.stack(ys), "lengths": torch.stack(ls)}

    ds = TensorDataset(input_ids, labels, lengths)
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------ _stable_sigmoid


def test_stable_sigmoid_matches_naive_in_safe_range() -> None:
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    naive = 1.0 / (1.0 + np.exp(-z))
    assert np.allclose(_stable_sigmoid(z), naive)


def test_stable_sigmoid_handles_extreme_negatives_without_overflow() -> None:
    # Previously: np.exp(-(-1000)) overflows. The stable form does not.
    z = np.array([-1000.0, -500.0, 1000.0])
    out = _stable_sigmoid(z)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[2] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ Evaluator


def test_evaluator_computes_accuracy_precision_recall_f1() -> None:
    # 4 samples: predict positive(>0) for first 2, negative for last 2.
    # True labels: [1, 0, 1, 0]  → preds: [1, 1, 0, 0]
    #   TP=1, FP=1, TN=1, FN=1 → P=0.5, R=0.5, F1=0.5, Acc=0.5
    fixed_logits = torch.tensor([2.0, 2.0, -2.0, -2.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    input_ids = torch.zeros(4, 8, dtype=torch.long)

    loader = _make_loader(input_ids, labels)
    model = _DummyModel(fixed_logits)
    result = Evaluator(model, device="cpu").evaluate(loader)

    assert result.accuracy == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    np.testing.assert_array_equal(result.confusion, np.array([[1, 1], [1, 1]]))
    assert "negative" in result.report and "positive" in result.report


def test_evaluator_probabilities_within_unit_interval() -> None:
    fixed_logits = torch.tensor([0.0, 1.5, -1.5, 5.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0])
    input_ids = torch.zeros(4, 8, dtype=torch.long)
    loader = _make_loader(input_ids, labels)
    result = Evaluator(_DummyModel(fixed_logits), device="cpu").evaluate(loader)
    assert ((result.y_proba >= 0.0) & (result.y_proba <= 1.0)).all()


def test_confusion_matrix_plot_renders(tmp_path) -> None:
    """Plot helper must save a file without raising — render correctness is visual."""
    fixed_logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    result = Evaluator(_DummyModel(fixed_logits), device="cpu").evaluate(
        _make_loader(input_ids, labels)
    )
    path = tmp_path / "cm.png"
    Evaluator.plot_confusion_matrix(result, save_path=path, show=False)
    assert path.exists() and path.stat().st_size > 0


# -------------------------------------------------------------- ErrorAnalyzer


def _make_result(y_true: list[int], y_pred: list[int], probs: list[float]) -> EvaluationResult:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    from sklearn.metrics import confusion_matrix

    return EvaluationResult(
        y_true=yt,
        y_pred=yp,
        y_proba=np.array(probs),
        confusion=confusion_matrix(yt, yp, labels=[0, 1]),
        report="",
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
    )


def test_error_analyzer_separates_fp_and_fn() -> None:
    # true=[0, 1, 1, 0], pred=[1, 0, 1, 0] →
    #   idx 0: FP (true=neg, pred=pos)
    #   idx 1: FN (true=pos, pred=neg)
    #   idx 2: TP, idx 3: TN
    result = _make_result(
        y_true=[0, 1, 1, 0],
        y_pred=[1, 0, 1, 0],
        probs=[0.95, 0.10, 0.80, 0.05],
    )
    texts = ["fp text", "fn text", "tp text", "tn text"]
    analyzer = ErrorAnalyzer(result, texts)

    fps = analyzer.collect_errors("fp")
    fns = analyzer.collect_errors("fn")
    both = analyzer.collect_errors("both")
    assert [s.index for s in fps] == [0]
    assert [s.index for s in fns] == [1]
    assert sorted(s.index for s in both) == [0, 1]


def test_error_analyzer_top_k_by_confidence() -> None:
    # 3 FPs with probs 0.95 / 0.55 / 0.99 → most-confident first is idx with prob 0.99.
    result = _make_result(
        y_true=[0, 0, 0],
        y_pred=[1, 1, 1],
        probs=[0.95, 0.55, 0.99],
    )
    analyzer = ErrorAnalyzer(result, ["a", "b", "c"])
    top = analyzer.collect_errors("fp", top_k=2, by_confidence=True)
    assert [s.index for s in top] == [2, 0]  # 0.99 then 0.95


def test_error_analyzer_text_length_mismatch_raises() -> None:
    result = _make_result(y_true=[0], y_pred=[1], probs=[0.9])
    with pytest.raises(ValueError):
        ErrorAnalyzer(result, texts=["t1", "t2"])  # 2 texts vs 1 result


def test_error_type_string_classification() -> None:
    result = _make_result(
        y_true=[0, 1],
        y_pred=[1, 0],
        probs=[0.9, 0.1],
    )
    analyzer = ErrorAnalyzer(result, ["fp_text", "fn_text"])
    errs = analyzer.collect_errors("both")
    types = {s.error_type for s in errs}
    assert any("False Positive" in t for t in types)
    assert any("False Negative" in t for t in types)
