import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy: fraction of correct predictions.

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of integers).
        y_pred (np.ndarray): Predicted labels (1D array of integers).

    Returns:
        float: Accuracy score between 0 and 1.

    Raises:
        ValueError: If inputs are empty or have mismatched shapes.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> accuracy(y_true, y_pred)
        0.5
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("metrics.accuracy: inputs must be non-empty with matching shapes.")
    
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision: fraction of positive predictions that are correct.

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of integers).
        y_pred (np.ndarray): Predicted labels (1D array of integers).

    Returns:
        float: Precision score between 0 and 1.

    Raises:
        ValueError: If inputs are empty, mismatched shapes, or no positive predictions.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 1])
        >>> precision(y_true, y_pred)
        0.5
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("metrics.precision: inputs must be non-empty with matching shapes.")
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    if predicted_positives == 0:
        return 0.0
    
    return float(true_positives / predicted_positives)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall: fraction of actual positives correctly identified.

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of integers).
        y_pred (np.ndarray): Predicted labels (1D array of integers).

    Returns:
        float: Recall score between 0 and 1.

    Raises:
        ValueError: If inputs are empty or have mismatched shapes.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 1])
        >>> recall(y_true, y_pred)
        0.5
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("metrics.recall: inputs must be non-empty with matching shapes.")
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    return float(true_positives / actual_positives)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score: harmonic mean of precision and recall.

    F1 = 2 × (precision × recall) / (precision + recall)

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of integers).
        y_pred (np.ndarray): Predicted labels (1D array of integers).

    Returns:
        float: F1 score between 0 and 1.

    Raises:
        ValueError: If inputs are empty or have mismatched shapes.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 1])
        >>> f1(y_true, y_pred)
        0.5
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    return float(2 * prec * rec / (prec + rec))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for binary classification.

                 Predicted
               0       1
    Actual 0 | TN | FP |
            1 | FN | TP |

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of 0s and 1s).
        y_pred (np.ndarray): Predicted labels (1D array of 0s and 1s).

    Returns:
        np.ndarray: 2x2 confusion matrix [[TN, FP], [FN, TP]]

    Raises:
        ValueError: If inputs are empty, mismatched shapes, or contain invalid labels.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 1])
        >>> confusion_matrix(y_true, y_pred)
        array([[1, 1],
               [1, 1]])
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("metrics.confusion_matrix: inputs must be non-empty with matching shapes.")
    
    # Validate binary labels
    if not (np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1]))):
        raise ValueError("metrics.confusion_matrix: labels must be binary (0 or 1).")
    
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # TN
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # FP
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # FN
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # TP
    
    return cm


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error for regression.

    MSE = (1/n) × Σ(pred_i - true_i)²

    Args:
        y_true (np.ndarray): Ground truth continuous values (1D array).
        y_pred (np.ndarray): Predicted continuous values (1D array).

    Returns:
        float: Mean squared error (lower is better).

    Raises:
        ValueError: If inputs are empty or have mismatched shapes.

    Complexity:
        Time: O(n)  Space: O(1)

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> mse(y_true, y_pred)
        0.375
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("metrics.mse: inputs must be non-empty with matching shapes.")
    
    return float(np.mean((y_true - y_pred) ** 2))