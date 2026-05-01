"""Ranking helpers for 1D NumPy arrays.

The functions here focus on the ranking edge cases that usually bite first:
ties, NaNs, sort direction, and percentile conversion.
"""

from __future__ import annotations

import numpy as np

# Re-export percentile from stats so the spec's "percentiles in rank.py"
# wording works as an import path.
from .stats import percentile  # noqa: F401


def rank(
    values: np.ndarray,
    method: str = "average",
    ascending: bool = True,
    na_option: str = "keep",
    start: int = 1,
) -> np.ndarray:
    """Rank a 1D array with configurable tie and NaN behavior.

    Tie methods:
        - average: every tied value gets the midpoint of the tie block.
        - min: every tied value gets the first rank in the tie block.
        - max: every tied value gets the last rank in the tie block.
        - dense: tied values share rank, and the next distinct value moves by 1.
        - ordinal: each item gets a unique rank in stable sorted order.

    Args:
        values: Input values as a 1D array-like object.
        method: Tie strategy. One of average, min, max, dense, ordinal.
        ascending: If True, smaller values receive smaller ranks.
        na_option: NaN strategy. One of keep, top, bottom.
        start: Starting rank index. Must be >= 1.

    Returns:
        Rank array with shape (n,). The dtype is float so it can represent
        averaged ties and optional NaN placeholders.

    Raises:
        ValueError: If values is not 1D, method/na_option is invalid,
            or start is less than 1.

    Complexity:
        Time: O(n log n) because ranking needs sorting.
        Space: O(n).
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"rank.rank: expected 1-D input, got shape {arr.shape}.")
    if start < 1:
        raise ValueError("rank.rank: start must be >= 1.")

    valid_methods = {"average", "min", "max", "dense", "ordinal"}
    if method not in valid_methods:
        raise ValueError(
            f"rank.rank: method must be one of {valid_methods}, got '{method}'."
        )

    valid_na_options = {"keep", "top", "bottom"}
    if na_option not in valid_na_options:
        raise ValueError(
            f"rank.rank: na_option must be one of {valid_na_options}, got '{na_option}'."
        )

    if arr.size == 0:
        return np.asarray([], dtype=float)

    nan_mask = np.isnan(arr)
    valid_mask = ~nan_mask
    valid_values = arr[valid_mask]
    n_valid = valid_values.size

    ranks = np.full(arr.shape, np.nan, dtype=float)
    if n_valid > 0:
        sort_key = valid_values if ascending else -valid_values
        order = np.argsort(sort_key, kind="stable")
        sorted_values = valid_values[order]

        if method == "ordinal":
            sorted_ranks = np.arange(start, start + n_valid, dtype=float)
        elif method == "dense":
            group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
            sorted_ranks = np.cumsum(group_start).astype(float) + (start - 1)
        else:
            sorted_ranks = np.empty(n_valid, dtype=float)
            group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
            starts = np.flatnonzero(group_start)
            ends = np.r_[starts[1:], n_valid] - 1

            # This loop keeps tie-rank assignment clear and easy to debug.
            for s_idx, e_idx in zip(starts, ends):
                min_rank = float(start + s_idx)
                max_rank = float(start + e_idx)
                if method == "min":
                    group_rank = min_rank
                elif method == "max":
                    group_rank = max_rank
                else:  # method == "average"
                    group_rank = (min_rank + max_rank) / 2.0
                sorted_ranks[s_idx : e_idx + 1] = group_rank

        valid_ranks = np.empty(n_valid, dtype=float)
        valid_ranks[order] = sorted_ranks
        ranks[valid_mask] = valid_ranks

    n_nan = int(np.sum(nan_mask))
    if n_nan > 0 and na_option != "keep":
        if na_option == "top":
            ranks[valid_mask] = ranks[valid_mask] + n_nan
            ranks[nan_mask] = np.arange(start, start + n_nan, dtype=float)
        else:  # na_option == "bottom"
            nan_start = start + n_valid
            ranks[nan_mask] = np.arange(nan_start, nan_start + n_nan, dtype=float)

    return ranks


def percentile_ranks(
    values: np.ndarray,
    method: str = "average",
    ascending: bool = True,
    na_option: str = "keep",
) -> np.ndarray:
    """Convert ranks of a 1D array into percentiles in [0, 100].

    Internally this uses min-max scaling on the computed ranks:
    percentile = (rank - min_rank) / (max_rank - min_rank) * 100

    Args:
        values: Input values as a 1D array-like object.
        method: Tie strategy used by rank.
        ascending: If True, smaller values receive smaller percentiles.
        na_option: NaN strategy used by rank.

    Returns:
        Percentile array in [0, 100] with shape (n,). NaN values stay NaN
        when na_option is keep.

    Raises:
        ValueError: Propagated from rank for invalid inputs.

    Complexity:
        Time: O(n log n)
        Space: O(n)
    """
    ranked = rank(
        values=values,
        method=method,
        ascending=ascending,
        na_option=na_option,
        start=1,
    )

    if ranked.size == 0:
        return ranked

    valid_mask = ~np.isnan(ranked)
    if not np.any(valid_mask):
        return ranked

    valid = ranked[valid_mask]
    lo = float(np.min(valid))
    hi = float(np.max(valid))

    percentiles = np.full(ranked.shape, np.nan, dtype=float)
    if np.isclose(hi, lo):
        percentiles[valid_mask] = 100.0
        return percentiles

    percentiles[valid_mask] = (valid - lo) / (hi - lo) * 100.0
    return percentiles
