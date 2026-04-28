"""Unit tests for utils module."""

import numpy as np
import pytest
from numcompute.utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    pairwise_distances,
    sigmoid,
    relu,
    softmax,
    tanh,
    logsumexp,
    make_batches,
)


# ── euclidean_distance ────────────────────────────────────────────────────────

def test_euclidean_basic():
    assert euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)

def test_euclidean_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(a, a) == pytest.approx(0.0)

def test_euclidean_shape_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance(np.array([1.0, 2.0]), np.array([1.0]))

def test_euclidean_2d_raises():
    with pytest.raises(ValueError):
        euclidean_distance(np.array([[1.0]]), np.array([[1.0]]))


# ── manhattan_distance ────────────────────────────────────────────────────────

def test_manhattan_basic():
    assert manhattan_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(7.0)

def test_manhattan_identical():
    a = np.array([5.0, 5.0])
    assert manhattan_distance(a, a) == pytest.approx(0.0)

def test_manhattan_shape_mismatch_raises():
    with pytest.raises(ValueError):
        manhattan_distance(np.array([1.0]), np.array([1.0, 2.0]))


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_identical():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)

def test_cosine_orthogonal():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

def test_cosine_zero_vector():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0

def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0]), np.array([1.0, 2.0]))


# ── pairwise_distances ────────────────────────────────────────────────────────

def test_pairwise_euclidean_diagonal_zero():
    X = np.random.rand(4, 3)
    D = pairwise_distances(X, metric="euclidean")
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

def test_pairwise_euclidean_symmetric():
    X = np.random.rand(4, 3)
    D = pairwise_distances(X, metric="euclidean")
    np.testing.assert_allclose(D, D.T, atol=1e-10)

def test_pairwise_manhattan():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    D = pairwise_distances(X, metric="manhattan")
    assert D[0, 1] == pytest.approx(2.0)

def test_pairwise_invalid_metric_raises():
    with pytest.raises(ValueError):
        pairwise_distances(np.eye(3), metric="minkowski")


# ── sigmoid ───────────────────────────────────────────────────────────────────

def test_sigmoid_zero():
    np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5])

def test_sigmoid_large_positive():
    result = sigmoid(np.array([1000.0]))
    assert result[0] == pytest.approx(1.0, abs=1e-6)

def test_sigmoid_large_negative():
    result = sigmoid(np.array([-1000.0]))
    assert result[0] == pytest.approx(0.0, abs=1e-6)

def test_sigmoid_range():
    x = np.linspace(-10, 10, 100)
    result = sigmoid(x)
    assert np.all(result > 0) and np.all(result < 1)


# ── relu ──────────────────────────────────────────────────────────────────────

def test_relu_positive():
    np.testing.assert_array_equal(relu(np.array([1.0, 2.0])), [1.0, 2.0])

def test_relu_negative():
    np.testing.assert_array_equal(relu(np.array([-1.0, -5.0])), [0.0, 0.0])

def test_relu_zero():
    assert relu(np.array([0.0]))[0] == 0.0


# ── softmax ───────────────────────────────────────────────────────────────────

def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.sum(softmax(x)), 1.0, atol=1e-10)

def test_softmax_large_values_stable():
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-10)

def test_softmax_uniform():
    x = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(softmax(x), [1/3, 1/3, 1/3], atol=1e-10)


# ── logsumexp ─────────────────────────────────────────────────────────────────

def test_logsumexp_known_value():
    x = np.array([0.0, 0.0])
    assert logsumexp(x) == pytest.approx(np.log(2.0))

def test_logsumexp_large_stable():
    x = np.array([1000.0, 1000.0])
    result = logsumexp(x)
    assert not np.isnan(result)
    assert result == pytest.approx(1000.0 + np.log(2.0))


# ── make_batches ──────────────────────────────────────────────────────────────

def test_make_batches_count():
    X = np.arange(10).reshape(5, 2)
    batches = make_batches(X, batch_size=2)
    assert len(batches) == 3  # 2+2+1

def test_make_batches_last_smaller():
    X = np.arange(10).reshape(5, 2)
    batches = make_batches(X, batch_size=3)
    assert batches[-1].shape[0] == 2

def test_make_batches_shuffle():
    X = np.arange(20).reshape(10, 2)
    batches_s = make_batches(X, batch_size=10, shuffle=True, seed=42)
    batches_n = make_batches(X, batch_size=10, shuffle=False)
    assert not np.array_equal(batches_s[0], batches_n[0])

def test_make_batches_empty_raises():
    with pytest.raises(ValueError):
        make_batches(np.array([]).reshape(0, 2), batch_size=2)

def test_make_batches_invalid_size_raises():
    with pytest.raises(ValueError):
        make_batches(np.ones((5, 2)), batch_size=0)
