"""Unit tests for rank module."""

import numpy as np
import pytest

from numcompute import percentile_ranks, rank


def test_rank_average_ties() -> None:
    values = np.array([10.0, 20.0, 20.0, 30.0], dtype=float)
    result = rank(values, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0], dtype=float)
    np.testing.assert_allclose(result, expected)


def test_rank_min_and_max_ties() -> None:
    values = np.array([5.0, 5.0, 9.0], dtype=float)
    min_ranks = rank(values, method="min")
    max_ranks = rank(values, method="max")

    np.testing.assert_allclose(min_ranks, np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(max_ranks, np.array([2.0, 2.0, 3.0]))


def test_rank_dense_ties() -> None:
    values = np.array([100.0, 10.0, 10.0, 50.0], dtype=float)
    result = rank(values, method="dense")
    np.testing.assert_allclose(result, np.array([3.0, 1.0, 1.0, 2.0]))


def test_rank_ordinal_is_stable_for_ties() -> None:
    values = np.array([2.0, 1.0, 1.0, 3.0], dtype=float)
    result = rank(values, method="ordinal")
    # The two tied 1.0 values preserve original order with stable sort.
    np.testing.assert_allclose(result, np.array([3.0, 1.0, 2.0, 4.0]))


def test_rank_descending_order() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    result = rank(values, method="min", ascending=False)
    np.testing.assert_allclose(result, np.array([3.0, 2.0, 1.0]))


def test_rank_nan_keep_preserves_nan() -> None:
    values = np.array([3.0, np.nan, 1.0], dtype=float)
    result = rank(values, na_option="keep")
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[[0, 2]], np.array([2.0, 1.0]))


def test_rank_nan_top_and_bottom() -> None:
    values = np.array([3.0, np.nan, 1.0, np.nan], dtype=float)
    top = rank(values, na_option="top")
    bottom = rank(values, na_option="bottom")

    np.testing.assert_allclose(top, np.array([4.0, 1.0, 3.0, 2.0]))
    np.testing.assert_allclose(bottom, np.array([2.0, 3.0, 1.0, 4.0]))


def test_rank_empty_returns_empty() -> None:
    result = rank(np.array([], dtype=float))
    assert result.size == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "banana"}, "method must be one of"),
        ({"na_option": "drop"}, "na_option must be one of"),
        ({"start": 0}, "start must be >= 1"),
    ],
)
def test_rank_validation_errors(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rank(np.array([1.0, 2.0], dtype=float), **kwargs)


def test_rank_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="expected 1-D input"):
        rank(np.array([[1.0, 2.0]], dtype=float))


def test_percentile_ranks_scale_to_0_and_100() -> None:
    values = np.array([10.0, 20.0, 30.0], dtype=float)
    result = percentile_ranks(values)
    np.testing.assert_allclose(result, np.array([0.0, 50.0, 100.0]))


def test_percentile_ranks_with_ties_and_nan_keep() -> None:
    values = np.array([5.0, 5.0, np.nan, 10.0], dtype=float)
    result = percentile_ranks(values, method="average", na_option="keep")

    assert np.isnan(result[2])
    np.testing.assert_allclose(result[[0, 1, 3]], np.array([0.0, 0.0, 100.0]))
