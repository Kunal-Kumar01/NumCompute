"""Optimisation Module

Finite-difference gradient and Jacobian estimation, plus an optional
backtracking line search. No external ML libraries used — only NumPy.
"""

import numpy as np


def finite_diff_gradient(
    f,
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Estimate the gradient of a scalar-valued function using finite differences.

    Three finite-difference schemes are supported:
        - central:  (f(x+h) - f(x-h)) / (2h)   — O(h^2) error, recommended
        - forward:  (f(x+h) - f(x))   / h        — O(h) error
        - backward: (f(x) - f(x-h))   / h        — O(h) error

    Args:
        f (callable): Scalar function f: R^n → R. Must accept a 1-D np.ndarray.
        x (np.ndarray): Point at which to estimate the gradient, shape (n,).
        h (float): Step size for finite differences. Default 1e-5.
        method (str): One of 'central', 'forward', 'backward'. Default 'central'.

    Returns:
        np.ndarray: Estimated gradient, shape (n,).

    Raises:
        ValueError: If x is not 1-D, h <= 0, or method is unsupported.

    Complexity:
        Time: O(n) function evaluations  Space: O(n)

    Examples:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> grad = finite_diff_gradient(f, np.array([1.0, 2.0]))
        >>> grad   # approximately [2.0, 4.0]
        array([2., 4.])
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"optim.finite_diff_gradient: expected 1-D array, got shape {x.shape}."
        )
    if h <= 0:
        raise ValueError(
            f"optim.finite_diff_gradient: h must be > 0, got {h}."
        )
    valid_methods = {"central", "forward", "backward"}
    if method not in valid_methods:
        raise ValueError(
            f"optim.finite_diff_gradient: method must be one of {valid_methods}, "
            f"got '{method}'."
        )

    n = x.shape[0]
    grad = np.zeros(n, dtype=float)
    # Use an identity matrix to perturb one dimension at a time — vectorised
    # in the sense that there are no inner loops; each step is a single np op.
    eye = np.eye(n, dtype=float) * h

    if method == "central":
        for i in range(n):
            grad[i] = (f(x + eye[i]) - f(x - eye[i])) / (2.0 * h)
    elif method == "forward":
        f0 = f(x)
        for i in range(n):
            grad[i] = (f(x + eye[i]) - f0) / h
    else:  # backward
        f0 = f(x)
        for i in range(n):
            grad[i] = (f0 - f(x - eye[i])) / h

    return grad


def finite_diff_jacobian(
    f,
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Estimate the Jacobian matrix of a vector-valued function using finite differences.

    The Jacobian J has shape (m, n) where m = len(f(x)) and n = len(x).
    J[i, j] = d(f_i) / d(x_j)

    Args:
        f (callable): Vector function f: R^n → R^m. Must accept a 1-D np.ndarray
            and return a 1-D np.ndarray.
        x (np.ndarray): Point at which to estimate the Jacobian, shape (n,).
        h (float): Step size. Default 1e-5.
        method (str): One of 'central', 'forward', 'backward'. Default 'central'.

    Returns:
        np.ndarray: Estimated Jacobian matrix, shape (m, n).

    Raises:
        ValueError: If x is not 1-D, h <= 0, or method is unsupported.

    Complexity:
        Time: O(n) function evaluations  Space: O(m * n)

    Examples:
        >>> f = lambda x: np.array([x[0]**2, x[0]*x[1]])
        >>> J = finite_diff_jacobian(f, np.array([1.0, 2.0]))
        >>> J   # approximately [[2., 0.], [2., 1.]]
        array([[2., 0.],
               [2., 1.]])
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"optim.finite_diff_jacobian: expected 1-D array, got shape {x.shape}."
        )
    if h <= 0:
        raise ValueError(
            f"optim.finite_diff_jacobian: h must be > 0, got {h}."
        )
    valid_methods = {"central", "forward", "backward"}
    if method not in valid_methods:
        raise ValueError(
            f"optim.finite_diff_jacobian: method must be one of {valid_methods}, "
            f"got '{method}'."
        )

    n = x.shape[0]
    f0 = np.asarray(f(x), dtype=float)
    m = f0.shape[0]
    jacobian = np.zeros((m, n), dtype=float)
    eye = np.eye(n, dtype=float) * h

    if method == "central":
        for j in range(n):
            jacobian[:, j] = (
                np.asarray(f(x + eye[j]), dtype=float)
                - np.asarray(f(x - eye[j]), dtype=float)
            ) / (2.0 * h)
    elif method == "forward":
        for j in range(n):
            jacobian[:, j] = (
                np.asarray(f(x + eye[j]), dtype=float) - f0
            ) / h
    else:  # backward
        for j in range(n):
            jacobian[:, j] = (
                f0 - np.asarray(f(x - eye[j]), dtype=float)
            ) / h

    return jacobian


def line_search(
    f,
    x: np.ndarray,
    direction: np.ndarray,
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Backtracking line search satisfying the Armijo sufficient-decrease condition.

    Starting from step size alpha, repeatedly shrinks by factor rho until
    the Armijo condition is met:
        f(x + alpha * d) <= f(x) + c * alpha * grad(f).T @ d

    Args:
        f (callable): Scalar function f: R^n → R.
        x (np.ndarray): Current point, shape (n,).
        direction (np.ndarray): Descent direction, shape (n,). Should satisfy
            grad(f).T @ direction < 0 for convergence.
        alpha (float): Initial step size. Default 1.0.
        rho (float): Shrinkage factor in (0, 1). Default 0.5.
        c (float): Armijo sufficient decrease constant in (0, 1). Default 1e-4.
        max_iter (int): Maximum number of shrinkage steps. Default 100.

    Returns:
        float: Accepted step size satisfying Armijo condition, or smallest
            alpha after max_iter steps.

    Raises:
        ValueError: If x or direction are not 1-D, or alpha/rho/c are invalid.

    Complexity:
        Time: O(max_iter) function evaluations  Space: O(n)

    Examples:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> x = np.array([1.0, 1.0])
        >>> d = np.array([-1.0, -1.0])
        >>> step = line_search(f, x, d)
        >>> step > 0
        True
    """
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            f"optim.line_search: x must be 1-D, got shape {x.shape}."
        )
    if direction.ndim != 1 or direction.shape != x.shape:
        raise ValueError(
            f"optim.line_search: direction must be 1-D with same shape as x."
        )
    if not (0 < rho < 1):
        raise ValueError(f"optim.line_search: rho must be in (0, 1), got {rho}.")
    if not (0 < c < 1):
        raise ValueError(f"optim.line_search: c must be in (0, 1), got {c}.")
    if alpha <= 0:
        raise ValueError(f"optim.line_search: alpha must be > 0, got {alpha}.")

    f0 = f(x)
    # Estimate directional derivative via forward difference
    grad = finite_diff_gradient(f, x, method="forward")
    slope = float(np.dot(grad, direction))

    for _ in range(max_iter):
        if f(x + alpha * direction) <= f0 + c * alpha * slope:
            break
        alpha *= rho

    return alpha


# Spec-aligned short names. The longer ones above are kept as the canonical
# implementation; these aliases let user code follow the spec wording exactly.
grad = finite_diff_gradient
jacobian = finite_diff_jacobian
