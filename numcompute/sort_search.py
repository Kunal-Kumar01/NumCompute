"""Sort and Search Module

Implements sorting algorithms, top-k selection, quickselect,
and binary search using vectorised NumPy operations.
"""

import numpy as np


def argsort(arr: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Return indices that would sort the array.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        Input 1-D array.
    ascending : bool
        If True, sort smallest to largest. Default True.

    Returns
    -------
    np.ndarray, shape (n,)
        Indices that sort arr.

    Raises
    ------
    ValueError
        If arr is not 1-D.

    Complexity
    ----------
    Time: O(n log n)  Space: O(n)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    indices = np.argsort(arr, kind="stable")
    return indices if ascending else indices[::-1]


def sort(arr: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Return a sorted copy of the array.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        Input 1-D array.
    ascending : bool
        Sort direction. Default True.

    Returns
    -------
    np.ndarray, shape (n,)
        Sorted array (copy, original unchanged).

    Raises
    ------
    ValueError
        If arr is not 1-D.

    Complexity
    ----------
    Time: O(n log n)  Space: O(n)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    sorted_arr = np.sort(arr, kind="stable")
    return sorted_arr if ascending else sorted_arr[::-1]


def top_k(arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the k largest values and their indices.

    Uses np.argpartition for O(n) average complexity — faster
    than a full sort for large arrays.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        Input 1-D array.
    k : int
        Number of top elements to return. Must satisfy 1 <= k <= n.

    Returns
    -------
    values : np.ndarray, shape (k,)
        The k largest values, sorted descending.
    indices : np.ndarray, shape (k,)
        Their positions in the original array.

    Raises
    ------
    ValueError
        If arr is not 1-D or k is out of range.

    Complexity
    ----------
    Time: O(n + k log k)  Space: O(k)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Array must not be empty.")
    if not (1 <= k <= n):
        raise ValueError(f"k={k} is out of valid range [1, {n}].")

    # argpartition guarantees the k largest are in the last k slots
    partition_indices = np.argpartition(arr, -k)[-k:]
    # sort those k elements descending
    order = np.argsort(arr[partition_indices])[::-1]
    top_indices = partition_indices[order]
    return arr[top_indices], top_indices


def quickselect(arr: np.ndarray, k: int) -> float:
    """Return the k-th smallest value (0-indexed) without full sort.

    Uses NumPy's partition for an efficient O(n) average solution.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        Input 1-D array.
    k : int
        0-based index of the order statistic (0 = minimum).

    Returns
    -------
    float
        The k-th smallest element.

    Raises
    ------
    ValueError
        If arr is not 1-D, empty, or k is out of range.

    Complexity
    ----------
    Time: O(n) average  Space: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Array must not be empty.")
    if not (0 <= k < n):
        raise ValueError(f"k={k} is out of valid range [0, {n - 1}].")

    partitioned = np.partition(arr, k)
    return float(partitioned[k])


def binary_search(arr: np.ndarray, target: float) -> int:
    """Search for target in a sorted array, return its index or -1.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        Sorted 1-D array (ascending).
    target : float
        Value to search for.

    Returns
    -------
    int
        Index of target in arr, or -1 if not found.

    Raises
    ------
    ValueError
        If arr is not 1-D.

    Complexity
    ----------
    Time: O(log n)  Space: O(1)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        return -1

    idx = np.searchsorted(arr, target)
    if idx < arr.shape[0] and arr[idx] == target:
        return int(idx)
    return -1