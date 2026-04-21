"""Unit tests for sort_search module."""

import numpy as np
import pytest
from numcompute.sort_search import argsort, sort, top_k, quickselect, binary_search


# ── argsort ──────────────────────────────────────────────────────────────────

def test_argsort_ascending():
    arr = np.array([3, 1, 2])
    assert list(argsort(arr)) == [1, 2, 0]

def test_argsort_descending():
    arr = np.array([3, 1, 2])
    assert list(argsort(arr, ascending=False)) == [0, 2, 1]

def test_argsort_all_equal():
    arr = np.array([5, 5, 5])
    result = argsort(arr)
    assert len(result) == 3

def test_argsort_raises_on_2d():
    with pytest.raises(ValueError):
        argsort(np.array([[1, 2], [3, 4]]))


# ── sort ─────────────────────────────────────────────────────────────────────

def test_sort_ascending():
    arr = np.array([3, 1, 4, 1, 5])
    np.testing.assert_array_equal(sort(arr), np.array([1, 1, 3, 4, 5]))

def test_sort_descending():
    arr = np.array([3, 1, 4])
    np.testing.assert_array_equal(sort(arr, ascending=False), np.array([4, 3, 1]))

def test_sort_does_not_modify_original():
    arr = np.array([3, 1, 2])
    _ = sort(arr)
    np.testing.assert_array_equal(arr, np.array([3, 1, 2]))

def test_sort_single_element():
    np.testing.assert_array_equal(sort(np.array([42])), np.array([42]))


# ── top_k ────────────────────────────────────────────────────────────────────

def test_top_k_basic():
    arr = np.array([1, 5, 3, 9, 2])
    values, indices = top_k(arr, 2)
    assert set(values) == {9, 5}

def test_top_k_returns_sorted_descending():
    arr = np.array([4, 7, 1, 9, 3])
    values, _ = top_k(arr, 3)
    assert list(values) == sorted(values, reverse=True)

def test_top_k_k_equals_n():
    arr = np.array([3, 1, 2])
    values, indices = top_k(arr, 3)
    assert len(values) == 3

def test_top_k_invalid_k_raises():
    with pytest.raises(ValueError):
        top_k(np.array([1, 2, 3]), k=0)

def test_top_k_k_too_large_raises():
    with pytest.raises(ValueError):
        top_k(np.array([1, 2, 3]), k=10)

def test_top_k_empty_raises():
    with pytest.raises(ValueError):
        top_k(np.array([]), k=1)


# ── quickselect ───────────────────────────────────────────────────────────────

def test_quickselect_minimum():
    arr = np.array([4, 2, 7, 1, 9])
    assert quickselect(arr, 0) == 1.0

def test_quickselect_maximum():
    arr = np.array([4, 2, 7, 1, 9])
    assert quickselect(arr, 4) == 9.0

def test_quickselect_median():
    arr = np.array([3, 1, 4, 1, 5])
    assert quickselect(arr, 2) == 3.0

def test_quickselect_duplicates():
    arr = np.array([2, 2, 2, 2])
    assert quickselect(arr, 1) == 2.0

def test_quickselect_invalid_k_raises():
    with pytest.raises(ValueError):
        quickselect(np.array([1, 2, 3]), k=5)

def test_quickselect_empty_raises():
    with pytest.raises(ValueError):
        quickselect(np.array([]), k=0)


# ── binary_search ─────────────────────────────────────────────────────────────

def test_binary_search_found():
    arr = np.array([1, 3, 5, 7, 9])
    assert binary_search(arr, 5) == 2

def test_binary_search_not_found():
    arr = np.array([1, 3, 5, 7, 9])
    assert binary_search(arr, 4) == -1

def test_binary_search_first_element():
    arr = np.array([1, 3, 5])
    assert binary_search(arr, 1) == 0

def test_binary_search_last_element():
    arr = np.array([1, 3, 5])
    assert binary_search(arr, 5) == 2

def test_binary_search_empty_array():
    assert binary_search(np.array([]), 5) == -1