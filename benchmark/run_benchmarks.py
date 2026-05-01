"""Reproducible benchmark runner for NumCompute.

Runs the built-in benchmark suites in `numcompute.benchmarking` plus a few extra
loop-vs-vectorised comparisons (top-k, softmax, rank, percentile_ranks) and
writes a markdown table of results to `benchmark/results.md`.

The numpy global RNG is seeded before each suite so the input arrays are
identical run-to-run, which means timings are reproducible up to OS scheduler
noise.

Usage:
    python benchmark/run_benchmarks.py
"""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path

# Always import the in-tree numcompute, not whatever pip happened to install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from numcompute.benchmarking import compare, print_table, timer
from numcompute import (
    percentile_ranks,
    rank as nc_rank,
    softmax,
    top_k,
)
from numcompute.preprocessing import StandardScaler
from numcompute.utils import euclidean_distance


SEED = 0


def _env_info() -> dict[str, str]:
    return {
        "python":   sys.version.split()[0],
        "numpy":    np.__version__,
        "platform": platform.platform(),
        "machine":  platform.machine(),
        "processor": platform.processor() or "unknown",
    }


# ── benchmark suites ─────────────────────────────────────────────────────────

def bench_sort(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    arr = np.random.rand(n)
    arr_list = arr.tolist()
    return compare(
        {"vectorised (np.sort)": lambda: np.sort(arr),
         "python loop (sorted)": lambda: sorted(arr_list)},
        repeats=repeats,
    )


def bench_mean(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    arr = np.random.rand(n)
    arr_list = arr.tolist()

    def python_mean() -> None:
        total = 0.0
        for v in arr_list:
            total += v
        total / len(arr_list)

    return compare(
        {"vectorised (np.mean)": lambda: np.mean(arr),
         "python loop":          python_mean},
        repeats=repeats,
    )


def bench_euclidean(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    a = np.random.rand(n)
    b = np.random.rand(n)
    a_list, b_list = a.tolist(), b.tolist()

    def python_dist() -> None:
        total = 0.0
        for ai, bi in zip(a_list, b_list):
            total += (ai - bi) ** 2
        math.sqrt(total)

    return compare(
        {"vectorised (numcompute)": lambda: euclidean_distance(a, b),
         "python loop":             python_dist},
        repeats=repeats,
    )


def bench_standard_scaler(n: int, m: int, repeats: int) -> dict:
    np.random.seed(SEED)
    X = np.random.rand(n, m)

    def python_scale() -> None:
        col_means = [sum(X[r][c] for r in range(n)) / n for c in range(m)]
        col_stds  = [
            (sum((X[r][c] - col_means[c]) ** 2 for r in range(n)) / n) ** 0.5
            for c in range(m)
        ]
        out = [[0.0] * m for _ in range(n)]
        for r in range(n):
            for c in range(m):
                s = col_stds[c] if col_stds[c] != 0 else 1.0
                out[r][c] = (X[r][c] - col_means[c]) / s

    return compare(
        {"vectorised (StandardScaler)": lambda: StandardScaler().fit_transform(X),
         "python loop":                  python_scale},
        repeats=repeats,
    )


def bench_top_k(n: int, k: int, repeats: int) -> dict:
    np.random.seed(SEED)
    arr = np.random.rand(n)
    arr_list = arr.tolist()

    def python_top_k() -> None:
        # Naive: sort everything, take last k. Simple and honest baseline.
        sorted(arr_list)[-k:]

    return compare(
        {"vectorised (numcompute)": lambda: top_k(arr, k),
         "python loop":             python_top_k},
        repeats=repeats,
    )


def bench_softmax(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    x = np.random.rand(n)
    x_list = x.tolist()

    def python_softmax() -> None:
        m = max(x_list)
        exps = [math.exp(v - m) for v in x_list]
        total = sum(exps)
        [e / total for e in exps]

    return compare(
        {"vectorised (numcompute)": lambda: softmax(x),
         "python loop":             python_softmax},
        repeats=repeats,
    )


def bench_rank(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    arr = np.random.randint(0, 1000, size=n).astype(float)
    arr_list = arr.tolist()

    def python_rank_average() -> None:
        # Rank with average tie handling, naive O(n^2).
        ranks = [0.0] * n
        for i in range(n):
            less = 0
            equal = 0
            for j in range(n):
                if arr_list[j] < arr_list[i]:
                    less += 1
                elif arr_list[j] == arr_list[i]:
                    equal += 1
            ranks[i] = less + (equal + 1) / 2.0

    return compare(
        {"vectorised (numcompute.rank)": lambda: nc_rank(arr, method="average"),
         "python loop (O(n^2))":          python_rank_average},
        repeats=repeats,
    )


def bench_percentile_ranks(n: int, repeats: int) -> dict:
    np.random.seed(SEED)
    arr = np.random.rand(n)
    return compare(
        {"vectorised (percentile_ranks)": lambda: percentile_ranks(arr)},
        repeats=repeats,
    )


# ── orchestration ────────────────────────────────────────────────────────────

SUITES = [
    ("Sort",            lambda r: bench_sort(n=100_000, repeats=r),                      "n=100,000"),
    ("Mean",            lambda r: bench_mean(n=1_000_000, repeats=r),                    "n=1,000,000"),
    ("Euclidean",       lambda r: bench_euclidean(n=10_000, repeats=r),                  "n=10,000"),
    ("StandardScaler",  lambda r: bench_standard_scaler(n=2_000, m=50, repeats=r),       "n=2,000, m=50"),
    ("Top-k",           lambda r: bench_top_k(n=100_000, k=10, repeats=r),               "n=100,000, k=10"),
    ("Softmax",         lambda r: bench_softmax(n=100_000, repeats=r),                   "n=100,000"),
    ("Rank (avg ties)", lambda r: bench_rank(n=2_000, repeats=r),                        "n=2,000"),
]


def _format_results_md(env: dict, ran_at: str, results: list) -> str:
    lines = [
        "# Benchmark Results",
        "",
        f"_Generated by `python benchmark/run_benchmarks.py` at {ran_at}._",
        "",
        "## Environment",
        "",
        f"- Python: {env['python']}",
        f"- NumPy: {env['numpy']}",
        f"- Platform: {env['platform']}",
        f"- Machine: {env['machine']}",
        f"- Processor: {env['processor']}",
        f"- Seed: {SEED}",
        "",
        "## Vectorised vs. Python loop",
        "",
        "| Suite | Inputs | Vectorised mean (ms) | Loop mean (ms) | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, inputs, suite_results in results:
        labels = list(suite_results.keys())
        if len(labels) < 2:
            continue
        vec_label, loop_label = labels[0], labels[1]
        vec_ms  = suite_results[vec_label]["mean_s"]  * 1000.0
        loop_ms = suite_results[loop_label]["mean_s"] * 1000.0
        speedup = loop_ms / vec_ms if vec_ms > 0 else float("inf")
        lines.append(
            f"| {name} | {inputs} | {vec_ms:.3f} | {loop_ms:.3f} | {speedup:.1f}x |"
        )
    lines.append("")
    lines.append(
        "Speedup is the loop's mean time divided by the vectorised mean time. "
        "Times are wall-clock measured with `time.perf_counter` over the configured "
        "number of repeats."
    )
    lines.append("")
    return "\n".join(lines)


def main(repeats: int = 5) -> None:
    env = _env_info()
    print("\nNumCompute reproducible benchmark run")
    print("=" * 62)
    for k, v in env.items():
        print(f"  {k:<10}: {v}")
    print(f"  seed      : {SEED}")
    print(f"  repeats   : {repeats}")
    print("=" * 62)

    results = []
    for name, runner, inputs in SUITES:
        suite_results = runner(repeats)
        print_table(suite_results, title=f"{name}  ({inputs})")
        results.append((name, inputs, suite_results))

    ran_at = time.strftime("%Y-%m-%d %H:%M:%S")
    md = _format_results_md(env, ran_at, results)
    out_path = Path(__file__).resolve().parent / "results.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWrote results table to: {out_path}")


if __name__ == "__main__":
    main()
