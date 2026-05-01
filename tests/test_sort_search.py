"""Unit tests for sort_search module."""

import numpy as np
import pytest
from numcompute.sort_search import (
    argsort,
    binary_search,
    multi_key_sort,
    quickselect,
    sort,
    top_k,
)


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

def test_binary_search_found_returns_index_and_true():
    arr = np.array([1, 3, 5, 7, 9])
    idx, exists = binary_search(arr, 5)
    assert idx == 2 and exists is True

def test_binary_search_not_found_returns_insertion_index_and_false():
    arr = np.array([1, 3, 5, 7, 9])
    idx, exists = binary_search(arr, 4)
    assert idx == 2 and exists is False

def test_binary_search_first_element():
    arr = np.array([1, 3, 5])
    idx, exists = binary_search(arr, 1)
    assert idx == 0 and exists is True

def test_binary_search_last_element():
    arr = np.array([1, 3, 5])
    idx, exists = binary_search(arr, 5)
    assert idx == 2 and exists is True

def test_binary_search_target_larger_than_all_returns_len():
    arr = np.array([1, 3, 5])
    idx, exists = binary_search(arr, 100)
    assert idx == 3 and exists is False

def test_binary_search_empty_array():
    idx, exists = binary_search(np.array([]), 5)
    assert idx == 0 and exists is False


# ── edge cases: strides, extremes ────────────────────────────────────────────

def test_sort_handles_non_contiguous_input():
    # A reverse-step slice is non-contiguous. The function must not assume
    # contiguous memory.
    arr = np.arange(10)[::2]  # [0, 2, 4, 6, 8] — view with non-unit stride
    assert arr.strides[0] != arr.itemsize
    np.testing.assert_array_equal(sort(arr), np.array([0, 2, 4, 6, 8]))

def test_argsort_handles_non_contiguous_input():
    arr = np.arange(10)[::-2]  # reverse-step view
    result = argsort(arr)
    assert result.shape == arr.shape
    # Sorted by index order should give monotonic values
    np.testing.assert_array_equal(arr[result], np.sort(arr))

def test_top_k_extreme_k_one():
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    values, _ = top_k(arr, 1)
    assert len(values) == 1 and values[0] == 9

def test_top_k_with_all_equal_values():
    arr = np.array([7, 7, 7, 7, 7])
    values, indices = top_k(arr, 3)
    assert len(values) == 3
    np.testing.assert_array_equal(values, [7, 7, 7])

def test_quickselect_with_all_equal_values():
    arr = np.array([5, 5, 5, 5, 5])
    for k in range(arr.size):
        assert quickselect(arr, k) == 5.0

def test_binary_search_handles_non_contiguous_sorted_input():
    arr = np.arange(20)[::2]  # [0, 2, 4, ..., 18] — sorted, non-contiguous
    idx, exists = binary_search(arr, 10)
    assert idx == 5 and exists is True
    idx, exists = binary_search(arr, 11)
    assert idx == 6 and exists is False


# ── top_k options ────────────────────────────────────────────────────────────

def test_top_k_largest_false_returns_smallest_ascending():
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    values, _ = top_k(arr, 3, largest=False)
    np.testing.assert_array_equal(values, [1, 1, 2])

def test_top_k_return_indices_false_returns_only_values():
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    values = top_k(arr, 2, return_indices=False)
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, [9, 6])

def test_top_k_indices_match_values_in_original_array():
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    values, indices = top_k(arr, 4)
    np.testing.assert_array_equal(arr[indices], values)


# ── multi_key_sort ───────────────────────────────────────────────────────────

def test_multi_key_sort_two_columns_priority():
    X = np.array([[2, 1], [1, 3], [2, 0], [1, 1]])
    result = multi_key_sort(X, keys=[0, 1])
    expected = np.array([[1, 1], [1, 3], [2, 0], [2, 1]])
    np.testing.assert_array_equal(result, expected)

def test_multi_key_sort_descending():
    X = np.array([[1, 5], [3, 2], [2, 8]])
    result = multi_key_sort(X, keys=[0], ascending=False)
    expected = np.array([[3, 2], [2, 8], [1, 5]])
    np.testing.assert_array_equal(result, expected)

def test_multi_key_sort_single_key_matches_sort_by_column():
    X = np.array([[1, 9], [3, 2], [2, 8]])
    result = multi_key_sort(X, keys=[1])
    expected = X[np.argsort(X[:, 1])]
    np.testing.assert_array_equal(result, expected)

def test_multi_key_sort_does_not_modify_input():
    X = np.array([[2, 1], [1, 3]])
    X_before = X.copy()
    _ = multi_key_sort(X, keys=[0])
    np.testing.assert_array_equal(X, X_before)

def test_multi_key_sort_rejects_1d():
    with pytest.raises(ValueError, match="expected 2-D"):
        multi_key_sort(np.array([1, 2, 3]), keys=[0])

def test_multi_key_sort_rejects_empty_keys():
    with pytest.raises(ValueError, match="keys must not be empty"):
        multi_key_sort(np.array([[1, 2]]), keys=[])

def test_multi_key_sort_rejects_out_of_range_key():
    with pytest.raises(ValueError, match="out of range"):
        multi_key_sort(np.array([[1, 2]]), keys=[5])