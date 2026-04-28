"""Unit tests for optim module."""

import numpy as np
import pytest
from numcompute.optim import finite_diff_gradient, finite_diff_jacobian, line_search


# ── finite_diff_gradient ──────────────────────────────────────────────────────

def test_gradient_quadratic_central():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad = finite_diff_gradient(f, np.array([1.0, 2.0]))
    np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-5)

def test_gradient_quadratic_forward():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad = finite_diff_gradient(f, np.array([1.0, 2.0]), method="forward")
    np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-4)

def test_gradient_quadratic_backward():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad = finite_diff_gradient(f, np.array([1.0, 2.0]), method="backward")
    np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-4)

def test_gradient_at_minimum():
    f = lambda x: (x[0] - 3.0) ** 2
    grad = finite_diff_gradient(f, np.array([3.0]))
    np.testing.assert_allclose(grad, [0.0], atol=1e-5)

def test_gradient_invalid_method_raises():
    with pytest.raises(ValueError):
        finite_diff_gradient(lambda x: x[0], np.array([1.0]), method="spooky")

def test_gradient_invalid_h_raises():
    with pytest.raises(ValueError):
        finite_diff_gradient(lambda x: x[0], np.array([1.0]), h=0.0)

def test_gradient_2d_input_raises():
    with pytest.raises(ValueError):
        finite_diff_gradient(lambda x: x[0, 0], np.array([[1.0, 2.0]]))

def test_gradient_shape():
    f = lambda x: np.sum(x)
    grad = finite_diff_gradient(f, np.ones(5))
    assert grad.shape == (5,)


# ── finite_diff_jacobian ──────────────────────────────────────────────────────

def test_jacobian_shape():
    f = lambda x: np.array([x[0] ** 2, x[0] * x[1]])
    J = finite_diff_jacobian(f, np.array([1.0, 2.0]))
    assert J.shape == (2, 2)

def test_jacobian_values():
    f = lambda x: np.array([x[0] ** 2, x[0] * x[1]])
    J = finite_diff_jacobian(f, np.array([1.0, 2.0]))
    # d(x0^2)/dx0=2, d(x0^2)/dx1=0, d(x0*x1)/dx0=2, d(x0*x1)/dx1=1
    expected = np.array([[2.0, 0.0], [2.0, 1.0]])
    np.testing.assert_allclose(J, expected, atol=1e-4)

def test_jacobian_identity_function():
    f = lambda x: x.copy()
    J = finite_diff_jacobian(f, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(J, np.eye(3), atol=1e-5)

def test_jacobian_invalid_method_raises():
    with pytest.raises(ValueError):
        finite_diff_jacobian(lambda x: x, np.array([1.0]), method="ghost")

def test_jacobian_invalid_h_raises():
    with pytest.raises(ValueError):
        finite_diff_jacobian(lambda x: x, np.array([1.0]), h=-1.0)


# ── line_search ───────────────────────────────────────────────────────────────

def test_line_search_returns_positive():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    step = line_search(f, np.array([1.0, 1.0]), np.array([-1.0, -1.0]))
    assert step > 0.0

def test_line_search_decreases_function():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    x = np.array([2.0, 2.0])
    d = np.array([-1.0, -1.0])
    step = line_search(f, x, d)
    assert f(x + step * d) < f(x)

def test_line_search_invalid_rho_raises():
    with pytest.raises(ValueError):
        line_search(lambda x: x[0] ** 2, np.array([1.0]), np.array([-1.0]), rho=1.5)

def test_line_search_invalid_alpha_raises():
    with pytest.raises(ValueError):
        line_search(lambda x: x[0] ** 2, np.array([1.0]), np.array([-1.0]), alpha=0.0)
