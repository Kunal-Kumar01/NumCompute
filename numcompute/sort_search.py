"""Sort and Search Module

Implements sorting algorithms, top-k selection, quickselect,
and binary search using vectorised NumPy operations.
All functions operate on 1-D arrays as per module specification.
"""

import numpy as np


def argsort(arr: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Return indices that would sort the array.

    Args:
        arr (np.ndarray): Input 1-D array, shape (n,).
        ascending (bool): If True sort smallest to largest. Default True.

    Returns:
        np.ndarray: Indices that sort arr, shape (n,).

    Raises:
        ValueError: If arr is not 1-D.

    Complexity:
        Time: O(n log n)  Space: O(n)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"sort_search functions operate on 1-D arrays only, got shape {arr.shape}."
        )
    indices = np.argsort(arr, kind="stable")
    return indices if ascending else indices[::-1]


def sort(arr: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Return a sorted copy of the array.

    Args:
        arr (np.ndarray): Input 1-D array, shape (n,).
        ascending (bool): Sort direction. Default True.

    Returns:
        np.ndarray: Sorted copy of arr, shape (n,).

    Raises:
        ValueError: If arr is not 1-D.

    Complexity:
        Time: O(n log n)  Space: O(n)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"sort_search functions operate on 1-D arrays only, got shape {arr.shape}."
        )
    sorted_arr = np.sort(arr, kind="stable")
    return sorted_arr if ascending else sorted_arr[::-1]


def top_k(
    arr: np.ndarray,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
):
    """Return the k largest (or smallest) values, optionally with indices.

    Uses np.argpartition for O(n) average partitioning, then sorts
    only the k candidates — faster than a full sort for large arrays.

    Args:
        arr (np.ndarray): Input 1-D array, shape (n,).
        k (int): Number of elements to return. Must satisfy 1 <= k <= n.
        largest (bool): If True (default), return the k largest in
            descending order. If False, return the k smallest in ascending order.
        return_indices (bool): If True (default), return a (values, indices)
            tuple. If False, return only values.

    Returns:
        np.ndarray or tuple:
            If return_indices is True: (values, indices) where values has
                shape (k,) and indices contains their positions in arr.
            If return_indices is False: values of shape (k,).

    Raises:
        ValueError: If arr is not 1-D, empty, or k is out of range.

    Complexity:
        Time: O(n + k log k)  Space: O(k)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"sort_search functions operate on 1-D arrays only, got shape {arr.shape}."
        )
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Array must not be empty.")
    if not (1 <= k <= n):
        raise ValueError(f"k={k} is out of valid range [1, {n}].")

    if largest:
        partition_indices = np.argpartition(arr, -k)[-k:]
        order = np.argsort(arr[partition_indices])[::-1]
    else:
        partition_indices = np.argpartition(arr, k - 1)[:k]
        order = np.argsort(arr[partition_indices])
    selected = partition_indices[order]
    values = arr[selected]

    if return_indices:
        return values, selected
    return values


def quickselect(arr: np.ndarray, k: int) -> float:
    """Return the k-th smallest value (0-indexed) without a full sort.

    Uses np.partition internally for O(n) average performance.

    Args:
        arr (np.ndarray): Input 1-D array, shape (n,).
        k (int): 0-based rank of the desired order statistic (0 = minimum).

    Returns:
        float: The k-th smallest element.

    Raises:
        ValueError: If arr is not 1-D, empty, or k is out of range.

    Complexity:
        Time: O(n) average  Space: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"sort_search functions operate on 1-D arrays only, got shape {arr.shape}."
        )
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Array must not be empty.")
    if not (0 <= k < n):
        raise ValueError(f"k={k} is out of valid range [0, {n - 1}].")

    return float(np.partition(arr, k)[k])


def binary_search(arr: np.ndarray, target: float) -> tuple:
    """Search for target in a sorted array.

    Returns the position where target should be inserted to keep the array
    sorted, plus a flag indicating whether target is already present.

    Args:
        arr (np.ndarray): Sorted 1-D array in ascending order, shape (n,).
        target (float): Value to find.

    Returns:
        tuple:
            insertion_index (int): Index where target should be inserted.
                If target is present, this is the index of (one of) its
                occurrences. If absent, this is where it would go to keep
                the array sorted; can equal len(arr) if target is larger
                than every element.
            exists (bool): True if target is already in arr, False otherwise.

    Raises:
        ValueError: If arr is not 1-D.

    Complexity:
        Time: O(log n)  Space: O(1)

    Examples:
        >>> binary_search(np.array([1.0, 3.0, 5.0]), 3.0)
        (1, True)
        >>> binary_search(np.array([1.0, 3.0, 5.0]), 4.0)
        (2, False)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"sort_search functions operate on 1-D arrays only, got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        return 0, False

    idx = int(np.searchsorted(arr, target))
    exists = idx < arr.shape[0] and arr[idx] == target
    return idx, bool(exists)


def multi_key_sort(
    X: np.ndarray,
    keys: list,
    ascending: bool = True,
) -> np.ndarray:
    """Sort rows of a 2-D array by multiple key columns (priority order).

    Earlier entries in `keys` are higher priority. For example, with
    `keys=[2, 0]` rows are first sorted by column 2, and ties are broken
    by column 0.

    Args:
        X (np.ndarray): Input 2-D array, shape (n_rows, n_cols). Must be numeric.
        keys (list of int): Column indices to sort by, most significant first.
        ascending (bool): If True (default), sort all keys in ascending order.
            If False, sort all keys in descending order.

    Returns:
        np.ndarray: A new array with rows reordered, shape (n_rows, n_cols).

    Raises:
        ValueError: If X is not 2-D, keys is empty, or any key is out of range.

    Complexity:
        Time: O(n log n) per key  Space: O(n)

    Examples:
        >>> X = np.array([[2, 1], [1, 3], [2, 0]])
        >>> multi_key_sort(X, keys=[0, 1])
        array([[1, 3], [2, 0], [2, 1]])
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"sort_search.multi_key_sort: expected 2-D array, got shape {X.shape}."
        )
    if not keys:
        raise ValueError("sort_search.multi_key_sort: keys must not be empty.")
    n_cols = X.shape[1]
    for k in keys:
        if not (0 <= k < n_cols):
            raise ValueError(
                f"sort_search.multi_key_sort: key {k} out of range [0, {n_cols - 1}]."
            )

    # np.lexsort uses the LAST key as primary, so reverse our priority order.
    # For descending, negate the columns (assumes numeric input).
    sign = 1.0 if ascending else -1.0
    columns = [sign * X[:, k] for k in reversed(keys)]
    indices = np.lexsort(columns)
    return X[indices]