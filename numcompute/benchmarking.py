"""Benchmarking Module

Micro-benchmark harness for comparing vectorised NumPy implementations
against Python loop equivalents. Produces timing tables and speedup ratios.
"""

import time
import numpy as np


def timer(func, *args, repeats: int = 5, **kwargs) -> dict:
    """Time a function call over multiple repeats and return statistics.

    Args:
        func (callable): Function to benchmark.
        *args: Positional arguments to pass to func.
        repeats (int): Number of times to run func. Default 5.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        dict: Timing statistics with keys:
            - mean_s (float): Mean elapsed time in seconds.
            - min_s (float): Minimum elapsed time in seconds.
            - max_s (float): Maximum elapsed time in seconds.
            - std_s (float): Standard deviation of elapsed times.
            - repeats (int): Number of repetitions run.

    Raises:
        ValueError: If repeats < 1.

    Complexity:
        Time: O(repeats * cost_of_func)  Space: O(repeats)

    Examples:
        >>> result = timer(np.sum, np.arange(1_000_000), repeats=10)
        >>> result['mean_s'] < 0.01
        True
    """
    if repeats < 1:
        raise ValueError(
            f"benchmarking.timer: repeats must be >= 1, got {repeats}."
        )

    times = np.empty(repeats, dtype=float)
    for i in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times[i] = time.perf_counter() - t0

    return {
        "mean_s": float(np.mean(times)),
        "min_s":  float(np.min(times)),
        "max_s":  float(np.max(times)),
        "std_s":  float(np.std(times)),
        "repeats": repeats,
    }


def compare(
    funcs: dict,
    *args,
    repeats: int = 5,
    **kwargs,
) -> dict:
    """Benchmark multiple functions on the same inputs and compare timings.

    Args:
        funcs (dict): Mapping of label → callable, e.g.
            {'vectorised': np_func, 'loop': py_func}.
        *args: Positional arguments passed to every function.
        repeats (int): Repeats per function. Default 5.
        **kwargs: Keyword arguments passed to every function.

    Returns:
        dict: Maps each label to its timer() result dict, plus a
            'speedup' entry showing mean time relative to the fastest.

    Raises:
        ValueError: If funcs is empty.

    Complexity:
        Time: O(k * repeats * cost_per_func)  Space: O(k)

    Examples:
        >>> arr = np.random.rand(100_000)
        >>> results = compare({'numpy': np.sum, 'loop': sum}, arr, repeats=5)
        >>> results['numpy']['mean_s'] < results['loop']['mean_s']
        True
    """
    if not funcs:
        raise ValueError("benchmarking.compare: funcs dict must not be empty.")

    results = {}
    for label, func in funcs.items():
        results[label] = timer(func, *args, repeats=repeats, **kwargs)

    # Compute speedup relative to the slowest function
    slowest = max(r["mean_s"] for r in results.values())
    for label in results:
        mean = results[label]["mean_s"]
        results[label]["speedup_vs_slowest"] = round(slowest / mean, 2) if mean > 0 else float("inf")

    return results


def print_table(results: dict, title: str = "Benchmark Results") -> None:
    """Print a formatted performance comparison table to stdout.

    Args:
        results (dict): Output from compare(), mapping label → stats dict.
        title (str): Title printed above the table. Default 'Benchmark Results'.

    Returns:
        None

    Examples:
        >>> arr = np.random.rand(10_000)
        >>> results = compare({'numpy': np.sum, 'loop': sum}, arr)
        >>> print_table(results)
    """
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    header = f"{'Function':<20} {'Mean (ms)':>10} {'Min (ms)':>10} {'Speedup':>10}"
    print(header)
    print(f"{'-'*62}")
    for label, stats in results.items():
        mean_ms = stats["mean_s"] * 1000
        min_ms  = stats["min_s"]  * 1000
        speedup = stats.get("speedup_vs_slowest", "-")
        print(f"{label:<20} {mean_ms:>10.3f} {min_ms:>10.3f} {speedup:>10}x")
    print(f"{'='*62}\n")


# ── Built-in Benchmark Suites ─────────────────────────────────────────────────

def bench_sort(n: int = 100_000, repeats: int = 5) -> dict:
    """Compare vectorised np.sort against a Python loop sort.

    Args:
        n (int): Array size. Default 100_000.
        repeats (int): Repeats per function. Default 5.

    Returns:
        dict: compare() results for 'numpy_sort' vs 'python_sort'.
    """
    arr = np.random.rand(n)
    arr_list = arr.tolist()

    def numpy_sort():
        np.sort(arr)

    def python_sort():
        sorted(arr_list)

    results = compare(
        {"numpy_sort": numpy_sort, "python_sort": python_sort},
        repeats=repeats,
    )
    print_table(results, title=f"Sort benchmark  (n={n:,})")
    return results


def bench_mean(n: int = 1_000_000, repeats: int = 5) -> dict:
    """Compare np.mean against a Python loop mean.

    Args:
        n (int): Array size. Default 1_000_000.
        repeats (int): Repeats per function. Default 5.

    Returns:
        dict: compare() results.
    """
    arr = np.random.rand(n)
    arr_list = arr.tolist()

    def numpy_mean():
        np.mean(arr)

    def python_mean():
        total = 0.0
        for v in arr_list:
            total += v
        total / len(arr_list)

    results = compare(
        {"numpy_mean": numpy_mean, "python_mean": python_mean},
        repeats=repeats,
    )
    print_table(results, title=f"Mean benchmark  (n={n:,})")
    return results


def bench_euclidean(n: int = 1_000, repeats: int = 5) -> dict:
    """Compare vectorised euclidean distance against a Python loop version.

    Args:
        n (int): Vector length. Default 1_000.
        repeats (int): Repeats per function. Default 5.

    Returns:
        dict: compare() results.
    """
    from numcompute.utils import euclidean_distance

    a = np.random.rand(n)
    b = np.random.rand(n)
    a_list = a.tolist()
    b_list = b.tolist()

    def numpy_dist():
        euclidean_distance(a, b)

    def python_dist():
        total = 0.0
        for ai, bi in zip(a_list, b_list):
            total += (ai - bi) ** 2
        total ** 0.5

    results = compare(
        {"numpy_euclidean": numpy_dist, "python_euclidean": python_dist},
        repeats=repeats,
    )
    print_table(results, title=f"Euclidean distance benchmark  (n={n:,})")
    return results


def bench_standard_scaler(n: int = 10_000, m: int = 50, repeats: int = 5) -> dict:
    """Compare vectorised StandardScaler against a Python loop version.

    Args:
        n (int): Number of samples. Default 10_000.
        m (int): Number of features. Default 50.
        repeats (int): Repeats per function. Default 5.

    Returns:
        dict: compare() results.
    """
    from numcompute.preprocessing import StandardScaler

    X = np.random.rand(n, m)

    def numpy_scale():
        StandardScaler().fit_transform(X)

    def python_scale():
        result = [[0.0] * m for _ in range(n)]
        col_means = [sum(X[r][c] for r in range(n)) / n for c in range(m)]
        col_stds  = [
            (sum((X[r][c] - col_means[c]) ** 2 for r in range(n)) / n) ** 0.5
            for c in range(m)
        ]
        for r in range(n):
            for c in range(m):
                s = col_stds[c] if col_stds[c] != 0 else 1.0
                result[r][c] = (X[r][c] - col_means[c]) / s

    results = compare(
        {"numpy_scaler": numpy_scale, "python_scaler": python_scale},
        repeats=repeats,
    )
    print_table(results, title=f"StandardScaler benchmark  (n={n:,}, m={m})")
    return results


def run_all(repeats: int = 5) -> dict:
    """Run all built-in benchmark suites and return combined results.

    Args:
        repeats (int): Repeats per benchmark. Default 5.

    Returns:
        dict: Mapping of suite name → compare() results.
    """
    print("\nNumCompute Benchmark Suite")
    print("Python version: " + __import__("sys").version.split()[0])
    print("NumPy version:  " + np.__version__)
    print()

    all_results = {
        "sort":             bench_sort(repeats=repeats),
        "mean":             bench_mean(repeats=repeats),
        "euclidean":        bench_euclidean(repeats=repeats),
        "standard_scaler":  bench_standard_scaler(repeats=repeats),
    }
    return all_results
