"""Utils Module

Helper functions for distances, activations, numerical stability,
and batching. All operations are fully vectorised using NumPy.
"""

import numpy as np


# ── Distance Functions ────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean (L2) distance between two vectors.

    Formula: sqrt(sum((a - b)^2))

    Args:
        a (np.ndarray): First vector, shape (n,).
        b (np.ndarray): Second vector, shape (n,).

    Returns:
        float: Euclidean distance between a and b.

    Raises:
        ValueError: If a and b have different shapes or are not 1-D.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        5.0
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"utils.euclidean_distance: expected 1-D arrays, got {a.shape} and {b.shape}."
        )
    if a.shape != b.shape:
        raise ValueError(
            f"utils.euclidean_distance: shape mismatch {a.shape} vs {b.shape}."
        )
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Manhattan (L1) distance between two vectors.

    Formula: sum(|a - b|)

    Args:
        a (np.ndarray): First vector, shape (n,).
        b (np.ndarray): Second vector, shape (n,).

    Returns:
        float: Manhattan distance between a and b.

    Raises:
        ValueError: If a and b have different shapes or are not 1-D.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> manhattan_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        7.0
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"utils.manhattan_distance: expected 1-D arrays, got {a.shape} and {b.shape}."
        )
    if a.shape != b.shape:
        raise ValueError(
            f"utils.manhattan_distance: shape mismatch {a.shape} vs {b.shape}."
        )
    return float(np.sum(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Formula: dot(a, b) / (||a|| * ||b||)
    Returns 0.0 if either vector is the zero vector.

    Args:
        a (np.ndarray): First vector, shape (n,).
        b (np.ndarray): Second vector, shape (n,).

    Returns:
        float: Cosine similarity in [-1, 1].

    Raises:
        ValueError: If a and b have different shapes or are not 1-D.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        1.0
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"utils.cosine_similarity: expected 1-D arrays, got {a.shape} and {b.shape}."
        )
    if a.shape != b.shape:
        raise ValueError(
            f"utils.cosine_similarity: shape mismatch {a.shape} vs {b.shape}."
        )
    norm_a = np.sqrt(np.dot(a, a))
    norm_b = np.sqrt(np.dot(b, b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_distances(
    X: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute an (n x n) pairwise distance matrix for a set of row vectors.

    Args:
        X (np.ndarray): Input matrix, shape (n_samples, n_features).
        metric (str): One of 'euclidean', 'manhattan', 'cosine'. Default 'euclidean'.

    Returns:
        np.ndarray: Symmetric distance matrix, shape (n_samples, n_samples).

    Raises:
        ValueError: If X is not 2-D or metric is unsupported.

    Complexity:
        Time: O(n^2 * m)  Space: O(n^2)

    Examples:
        >>> X = np.array([[0.0, 0.0], [3.0, 4.0]])
        >>> pairwise_distances(X, metric='euclidean')
        array([[0., 5.], [5., 0.]])
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"utils.pairwise_distances: expected 2-D array, got shape {X.shape}."
        )
    valid = {"euclidean", "manhattan", "cosine"}
    if metric not in valid:
        raise ValueError(
            f"utils.pairwise_distances: metric must be one of {valid}, got '{metric}'."
        )

    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)

    if metric == "euclidean":
        # Vectorised: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq = np.sum(X ** 2, axis=1)
        dist = np.sqrt(
            np.maximum(sq[:, np.newaxis] + sq[np.newaxis, :] - 2.0 * (X @ X.T), 0.0)
        )
        # matmul vs np.sum use different summation orders, so the diagonal can
        # leave tiny non-zero residuals. A point is exactly distance 0 from itself.
        np.fill_diagonal(dist, 0.0)
    elif metric == "manhattan":
        # Vectorised via broadcasting (n, 1, m) - (1, n, m) → (n, n, m)
        dist = np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
    elif metric == "cosine":
        norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = X / norms
        dist = 1.0 - (X_norm @ X_norm.T)
        np.fill_diagonal(dist, 0.0)

    return dist


# ── Activation Functions ──────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function element-wise.

    Formula: 1 / (1 + exp(-x))
    Numerically stable: clips input to avoid overflow.

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        np.ndarray: Values in (0, 1), same shape as x.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> sigmoid(np.array([0.0]))
        array([0.5])
    """
    x = np.asarray(x, dtype=float)
    # Clip to avoid overflow in exp for very large/small values
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def relu(x: np.ndarray) -> np.ndarray:
    """Apply the ReLU activation function element-wise.

    Formula: max(0, x)

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        np.ndarray: Non-negative values, same shape as x.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> relu(np.array([-1.0, 0.0, 2.0]))
        array([0., 0., 2.])
    """
    return np.maximum(0.0, np.asarray(x, dtype=float))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply numerically stable softmax along the specified axis.

    Uses max-shifting to prevent overflow:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        x (np.ndarray): Input array of any shape.
        axis (int): Axis along which softmax is computed. Default -1 (last).

    Returns:
        np.ndarray: Probability distribution summing to 1 along axis,
            same shape as x.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09003057, 0.24472847, 0.66524096])
    """
    x = np.asarray(x, dtype=float)
    # Subtract max for numerical stability (max-shift trick)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def tanh(x: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic tangent activation function element-wise.

    Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        np.ndarray: Values in (-1, 1), same shape as x.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> tanh(np.array([0.0]))
        array([0.])
    """
    return np.tanh(np.asarray(x, dtype=float))


# ── Numerical Stability ───────────────────────────────────────────────────────

def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log(sum(exp(x))) in a numerically stable way.

    Uses the max-shift trick:
    logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    This avoids overflow when x contains large values and underflow
    when x contains very negative values.

    Args:
        x (np.ndarray): Input array of any shape.
        axis (int): Axis along which to compute. Default -1 (last axis).

    Returns:
        np.ndarray: Result with the specified axis reduced away.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> logsumexp(np.array([1000.0, 1001.0, 1002.0]))
        1002.4076...
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max.squeeze(axis=axis) + np.log(
        np.sum(np.exp(x - x_max), axis=axis)
    )
    return result


# ── Batching ──────────────────────────────────────────────────────────────────

def make_batches(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> list:
    """Split an array into a list of mini-batches along axis 0.

    The last batch may be smaller than batch_size if n is not divisible.

    Args:
        X (np.ndarray): Input array, shape (n_samples, ...).
        batch_size (int): Number of samples per batch. Must be >= 1.
        shuffle (bool): If True, shuffle rows before splitting. Default False.
        seed (int | None): Random seed for reproducibility when shuffle=True.

    Returns:
        list of np.ndarray: List of batches, each with shape (batch_size, ...).

    Raises:
        ValueError: If X is empty or batch_size < 1.

    Complexity:
        Time: O(n)  Space: O(n)

    Examples:
        >>> X = np.arange(10).reshape(5, 2)
        >>> batches = make_batches(X, batch_size=2)
        >>> len(batches)
        3
    """
    X = np.asarray(X)
    if X.shape[0] == 0:
        raise ValueError("utils.make_batches: input array is empty.")
    if batch_size < 1:
        raise ValueError(
            f"utils.make_batches: batch_size must be >= 1, got {batch_size}."
        )

    indices = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        X = X[indices]

    return [X[i : i + batch_size] for i in range(0, X.shape[0], batch_size)]
