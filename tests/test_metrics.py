"""Unit tests for metrics module."""

import numpy as np
import pytest

from numcompute import accuracy, confusion_matrix, f1, mse, precision, recall
from numcompute.metrics import accuracy as accuracy_direct
from numcompute.metrics import confusion_matrix as confusion_matrix_direct
from numcompute.metrics import f1 as f1_direct
from numcompute.metrics import mse as mse_direct
from numcompute.metrics import precision as precision_direct
from numcompute.metrics import recall as recall_direct


def test_package_exports_match_metrics_module() -> None:
    assert accuracy is accuracy_direct
    assert precision is precision_direct
    assert recall is recall_direct
    assert f1 is f1_direct
    assert confusion_matrix is confusion_matrix_direct
    assert mse is mse_direct


def test_accuracy_computes_fraction_correct() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_precision_recall_f1_compute_expected_values() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    assert precision(y_true, y_pred) == pytest.approx(0.5)
    assert recall(y_true, y_pred) == pytest.approx(0.5)
    assert f1(y_true, y_pred) == pytest.approx(0.5)


def test_precision_recall_and_f1_handle_zero_denominators() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    assert precision(y_true, y_pred) == 0.0
    assert recall(y_true, y_pred) == 0.0
    assert f1(y_true, y_pred) == 0.0


def test_confusion_matrix_returns_expected_counts() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    expected = np.array([[1, 1], [1, 1]])

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)


def test_confusion_matrix_rejects_non_binary_labels() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])

    with pytest.raises(ValueError, match="labels must be binary"):
        confusion_matrix(y_true, y_pred)


@pytest.mark.parametrize(
    "func, y_true, y_pred, message",
    [
        (accuracy, np.array([]), np.array([]), "inputs must be non-empty"),
        (precision, np.array([0, 1]), np.array([0]), "inputs must be non-empty"),
        (recall, np.array([]), np.array([]), "inputs must be non-empty"),
        (f1, np.array([0, 1]), np.array([0]), "inputs must be non-empty"),
        (confusion_matrix, np.array([]), np.array([]), "inputs must be non-empty"),
        (mse, np.array([]), np.array([]), "inputs must be non-empty"),
    ],
)
def test_metrics_raise_for_empty_or_shape_mismatch(func, y_true, y_pred, message) -> None:
    with pytest.raises(ValueError, match=message):
        func(y_true, y_pred)


def test_mse_computes_expected_value() -> None:
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    assert mse(y_true, y_pred) == pytest.approx(0.375)