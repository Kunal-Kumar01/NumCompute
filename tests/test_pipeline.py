"""Unit tests for pipeline module."""

import numpy as np
import pytest
from numcompute.pipeline import Transformer, Estimator, Compose, FeatureUnion
from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer


# ── Transformer base class ────────────────────────────────────────────────────

def test_transformer_fit_raises():
    with pytest.raises(NotImplementedError):
        Transformer().fit(np.ones((3, 2)))

def test_transformer_transform_raises():
    with pytest.raises(NotImplementedError):
        Transformer().transform(np.ones((3, 2)))


# ── Estimator base class ──────────────────────────────────────────────────────

def test_estimator_fit_raises():
    with pytest.raises(NotImplementedError):
        Estimator().fit(np.ones((3, 2)), np.ones(3))

def test_estimator_predict_raises():
    with pytest.raises(NotImplementedError):
        Estimator().predict(np.ones((3, 2)))


# ── Compose ───────────────────────────────────────────────────────────────────

def test_compose_empty_steps_raises():
    with pytest.raises(ValueError):
        Compose([])

def test_compose_fit_transform_two_scalers():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pipe = Compose([("imputer", Imputer()), ("scaler", StandardScaler())])
    result = pipe.fit_transform(X)
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

def test_compose_transform_before_fit_raises():
    pipe = Compose([("scaler", StandardScaler())])
    with pytest.raises(RuntimeError):
        pipe.transform(np.ones((3, 2)))

def test_compose_repr():
    pipe = Compose([("scaler", StandardScaler())])
    assert "Compose" in repr(pipe)

def test_compose_preserves_shape():
    X = np.random.rand(10, 4)
    pipe = Compose([("s1", StandardScaler()), ("s2", MinMaxScaler())])
    result = pipe.fit_transform(X)
    assert result.shape == X.shape

def test_compose_output_range_after_minmax():
    X = np.random.rand(20, 3) * 100
    pipe = Compose([("mm", MinMaxScaler())])
    result = pipe.fit_transform(X)
    assert result.min() >= 0.0 and result.max() <= 1.0


# ── FeatureUnion ──────────────────────────────────────────────────────────────

def test_feature_union_empty_raises():
    with pytest.raises(ValueError):
        FeatureUnion([])

def test_feature_union_doubles_features():
    X = np.random.rand(10, 3)
    fu = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])
    result = fu.fit_transform(X)
    assert result.shape == (10, 6)

def test_feature_union_transform_before_fit_raises():
    fu = FeatureUnion([("scaler", StandardScaler())])
    with pytest.raises(RuntimeError):
        fu.transform(np.ones((3, 2)))

def test_feature_union_repr():
    fu = FeatureUnion([("scaler", StandardScaler())])
    assert "FeatureUnion" in repr(fu)

def test_feature_union_three_transformers():
    X = np.random.rand(8, 2)
    fu = FeatureUnion([
        ("s1", StandardScaler()),
        ("s2", MinMaxScaler()),
        ("s3", Imputer()),
    ])
    result = fu.fit_transform(X)
    assert result.shape == (8, 6)
