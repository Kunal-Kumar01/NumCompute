"""Unit tests for stats module."""

import numpy as np
import pytest

from numcompute.stats import (
    WelfordStats,
    histogram,
    maximum,
    mean,
    median,
    minimum,
    percentile,
    std,
)


def test_welford_update_matches_numpy_population_stats() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=float)
    stats = WelfordStats()

    for value in values:
        stats.update(float(value))

    clean = values[~np.isnan(values)]
    assert stats.n == clean.size
    np.testing.assert_allclose(stats.mean, np.mean(clean), atol=1e-12)
    np.testing.assert_allclose(stats.variance, np.var(clean), atol=1e-12)
    np.testing.assert_allclose(stats.std, np.std(clean), atol=1e-12)
    assert stats.min == np.min(clean)
    assert stats.max == np.max(clean)


def test_welford_update_batch_and_summary_rounding() -> None:
    values = np.array([1.11111, 2.22222, np.nan, 3.33333], dtype=float)
    stats = WelfordStats()

    stats.update_batch(values)
    summary = stats.summary()

    clean = values[~np.isnan(values)]
    assert summary["n"] == clean.size
    assert summary["mean"] == round(float(np.mean(clean)), 4)
    assert summary["std"] == round(float(np.std(clean)), 4)
    assert summary["min"] == float(np.min(clean))
    assert summary["max"] == float(np.max(clean))


def test_mean_median_std_ignore_nan_with_axis_none() -> None:
    X = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=float)

    assert mean(X) == pytest.approx(np.nanmean(X))
    assert median(X) == pytest.approx(np.nanmedian(X))
    assert std(X) == pytest.approx(np.nanstd(X))


def test_axis_wise_stats_have_expected_shape_and_values() -> None:
    X = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=float)

    col_means = mean(X, axis=0)
    row_medians = median(X, axis=1)
    col_stds = std(X, axis=0)
    row_mins = minimum(X, axis=1)
    col_maxs = maximum(X, axis=0)

    assert col_means.shape == (3,)
    assert row_medians.shape == (2,)
    assert col_stds.shape == (3,)
    assert row_mins.shape == (2,)
    assert col_maxs.shape == (3,)

    np.testing.assert_allclose(col_means, np.array([2.5, 3.5, 6.0]))
    np.testing.assert_allclose(row_medians, np.array([1.5, 5.0]))
    np.testing.assert_allclose(row_mins, np.array([1.0, 4.0]))
    np.testing.assert_allclose(col_maxs, np.array([4.0, 5.0, 6.0]))


@pytest.mark.parametrize(
    "func, message",
    [
        (mean, "stats.mean: input array is empty."),
        (median, "stats.median: input array is empty."),
        (std, "stats.std: input array is empty."),
        (minimum, "stats.minimum: input array is empty."),
        (maximum, "stats.maximum: input array is empty."),
    ],
)
def test_descriptive_functions_raise_on_empty_input(func, message) -> None:
    with pytest.raises(ValueError, match=message):
        func(np.array([]))


def test_histogram_ignores_nan_and_counts_samples() -> None:
    X = np.array([0.0, 0.5, 1.0, 1.5, np.nan], dtype=float)

    counts, edges = histogram(X, bins=2, range_=(0.0, 2.0))

    assert counts.shape == (2,)
    assert edges.shape == (3,)
    assert counts.sum() == 4
    np.testing.assert_allclose(edges, np.array([0.0, 1.0, 2.0]))


def test_histogram_raises_for_empty_and_all_nan_inputs() -> None:
    with pytest.raises(ValueError, match="stats.histogram: input array is empty"):
        histogram(np.array([]))

    with pytest.raises(ValueError, match="stats.histogram: input contains only NaN values"):
        histogram(np.array([np.nan, np.nan]))


def test_percentile_scalar_and_multi_quantile_with_axis() -> None:
    X = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=float)

    p50 = percentile(X, 50)
    quartiles = percentile(X, [25, 75], axis=0)

    assert isinstance(p50, float)
    assert p50 == pytest.approx(4.0)
    assert quartiles.shape == (2, 3)
    np.testing.assert_allclose(quartiles[:, 0], np.array([1.75, 3.25]))


def test_percentile_validation_errors() -> None:
    X = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="stats.percentile: input array is empty"):
        percentile(np.array([]), 50)

    with pytest.raises(ValueError, match="all values in q must be between 0 and 100"):
        percentile(X, [-5, 50])

    with pytest.raises(ValueError, match="interpolation must be one of"):
        percentile(X, 50, interpolation="cubic")