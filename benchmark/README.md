# Benchmarks

Reproducible loop-vs-vectorised performance comparisons for the modules in
`numcompute/`.

## Run

```bash
python benchmark/run_benchmarks.py
```

This prints a table per suite and writes a combined markdown summary to
`benchmark/results.md`.

## What is measured

Each suite runs the same operation in two implementations on identical inputs:

| Suite | Compares |
| --- | --- |
| Sort | `np.sort` vs. Python `sorted` |
| Mean | `np.mean` vs. Python loop accumulator |
| Euclidean | `numcompute.utils.euclidean_distance` vs. Python loop |
| StandardScaler | `numcompute.preprocessing.StandardScaler` vs. nested-loop scaler |
| Top-k | `numcompute.top_k` vs. `sorted(...)[-k:]` |
| Softmax | `numcompute.softmax` (max-shifted) vs. Python loop |
| Rank | `numcompute.rank` (average ties) vs. naive O(n^2) loop |

Inputs are sized so the loop versions complete in a few seconds.

## Reproducibility

- The numpy global RNG is reseeded with `SEED = 0` at the start of every suite,
  so input arrays are identical run-to-run.
- Timings are wall-clock from `time.perf_counter`, mean over 5 repeats by
  default.
- Absolute times still depend on the host machine; speedup ratios are more
  portable than the millisecond numbers themselves.
- Environment metadata (Python version, NumPy version, OS, CPU) is recorded in
  `results.md` next to the table.

## Files

- [`run_benchmarks.py`](run_benchmarks.py) — runner script.
- [`results.md`](results.md) — generated summary table (regenerate by re-running).
