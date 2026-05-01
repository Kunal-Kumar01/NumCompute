"""Unit tests for preprocessing module."""

import numpy as np
import pytest
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


# ── StandardScaler ────────────────────────────────────────────────────────────

def test_standard_scaler_zero_mean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

def test_standard_scaler_unit_std():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(result.std(axis=0), [1.0, 1.0], atol=1e-10)

def test_standard_scaler_constant_column():
    # Constant column should not cause division by zero
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    result = StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0])

def test_standard_scaler_inverse():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    np.testing.assert_allclose(scaler.fit_transform(X), scaler.transform(X) , atol=1e-10)

def test_standard_scaler_not_fitted_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.array([[1.0, 2.0]]))

def test_standard_scaler_wrong_dims_raises():
    with pytest.raises(ValueError):
        StandardScaler().fit(np.array([1.0, 2.0, 3.0]))


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

def test_minmax_scaler_range():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = MinMaxScaler().fit_transform(X)
    assert result.min() >= 0.0 and result.max() <= 1.0

def test_minmax_scaler_min_is_zero():
    X = np.array([[1.0], [2.0], [3.0]])
    result = MinMaxScaler().fit_transform(X)
    assert result.min() == 0.0

def test_minmax_scaler_max_is_one():
    X = np.array([[1.0], [2.0], [3.0]])
    result = MinMaxScaler().fit_transform(X)
    assert result.max() == 1.0

def test_minmax_scaler_constant_column():
    X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    result = MinMaxScaler().fit_transform(X)
    np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0])

def test_minmax_not_fitted_raises():
    with pytest.raises(RuntimeError):
        MinMaxScaler().transform(np.array([[1.0]]))

def test_minmax_custom_feature_range():
    X = np.array([[1.0], [2.0], [3.0]])
    result = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
    np.testing.assert_allclose(result.min(), -1.0)
    np.testing.assert_allclose(result.max(), 1.0)
    np.testing.assert_allclose(result[1, 0], 0.0)

def test_minmax_inverse_transform_with_custom_range_round_trips():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = MinMaxScaler(feature_range=(-5.0, 5.0))
    scaled = scaler.fit_transform(X)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), X, atol=1e-10)

def test_minmax_invalid_feature_range_raises():
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=(1.0, 0.0))
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=(0.0, 0.0))


# ── Imputer ───────────────────────────────────────────────────────────────────

def test_imputer_mean_fills_nan():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    result = Imputer(strategy="mean").fit_transform(X)
    assert not np.isnan(result).any()

def test_imputer_mean_correct_value():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    result = Imputer(strategy="mean").fit_transform(X)
    # mean of col 1 (ignoring nan) = (4+6)/2 = 5.0
    assert result[0, 1] == 5.0

def test_imputer_median():
    X = np.array([[1.0, np.nan], [2.0, 2.0], [3.0, 8.0]])
    result = Imputer(strategy="median").fit_transform(X)
    # median of col 1 = 5.0
    assert result[0, 1] == 5.0

def test_imputer_mode():
    X = np.array([[np.nan, np.nan], [2.0, 3.0], [2.0, 3.0]])
    result = Imputer(strategy="mode").fit_transform(X)
    # mode of col 0 = 2.0, mode of col 1 = 3.0
    assert result[0, 0] == 2.0
    assert result[0, 1] == 3.0

def test_imputer_invalid_strategy_raises():
    with pytest.raises(ValueError):
        Imputer(strategy="banana")

def test_imputer_not_fitted_raises():
    with pytest.raises(RuntimeError):
        Imputer().transform(np.array([[1.0, np.nan]]))


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

def test_onehot_output_shape():
    X = np.array([[0], [1], [2]])
    result = OneHotEncoder().fit_transform(X)
    assert result.shape == (3, 3)

def test_onehot_correct_encoding():
    X = np.array([[0], [1], [2]])
    result = OneHotEncoder().fit_transform(X)
    expected = np.eye(3)
    np.testing.assert_array_equal(result, expected)

def test_onehot_multi_column():
    X = np.array([[0, 1], [1, 0]])
    result = OneHotEncoder().fit_transform(X)
    # col 0 has 2 cats, col 1 has 2 cats → 4 output columns
    assert result.shape == (2, 4)

def test_onehot_not_fitted_raises():
    with pytest.raises(RuntimeError):
        OneHotEncoder().transform(np.array([[0], [1]]))