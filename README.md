# NumCompute

A modular scientific computing toolkit built on plain Python and NumPy. NumCompute
provides reusable, vectorised components for data I/O, preprocessing,
sort/search/ranking, descriptive statistics, evaluation metrics, finite-difference
gradients, a lightweight pipeline abstraction, and a benchmarking harness.

No external ML or DL libraries are used. Only Python and NumPy.

## Installation

```bash
git clone <repo-url>
cd NumCompute
pip install .
```

For a development install (editable, with tests runnable in place):

```bash
pip install -e .
pip install pytest
```

## Quickstart

```python
import numpy as np
from numcompute import (
    load_data,
    Imputer, StandardScaler, MinMaxScaler, OneHotEncoder,
    Compose, FeatureUnion,
    rank, percentile_ranks,
    top_k, quickselect, binary_search,
    mean, std, percentile, histogram, WelfordStats,
    accuracy, precision, recall, f1, mse, confusion_matrix,
    finite_diff_gradient, finite_diff_jacobian,
    softmax, logsumexp, pairwise_distances, make_batches,
)

# Load a CSV with possibly missing values (becomes NaN).
X = load_data("data.csv", skip_header=1)

# Build a preprocessing pipeline.
pipe = Compose([
    ("impute", Imputer(strategy="mean")),
    ("scale",  StandardScaler()),
])
X_scaled = pipe.fit_transform(X)

# Rank with tie handling and convert to percentiles.
scores = np.array([10.0, 20.0, 20.0, 30.0])
print(rank(scores, method="average"))      # [1.  2.5 2.5 4. ]
print(percentile_ranks(scores))            # [0.  50. 50. 100.]

# Top-k and quickselect.
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
values, indices = top_k(arr, 3)            # 3 largest
median = quickselect(arr, k=arr.size // 2) # k-th smallest

# Numerically stable softmax / logsumexp.
print(softmax(np.array([1000.0, 1001.0, 1002.0])))
print(logsumexp(np.array([1000.0, 1000.0])))

# Finite-difference gradient of f(x) = x0^2 + x1^2 at (1, 2).
grad = finite_diff_gradient(lambda x: x[0]**2 + x[1]**2, np.array([1.0, 2.0]))
```

## API overview

| Module | Purpose | Key exports |
| --- | --- | --- |
| `io` | CSV reading with NaN handling | `load_data` |
| `preprocessing` | Scalers, imputation, one-hot encoding | `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder` |
| `sort_search` | Sorting, top-k, quickselect, binary search | `sort`, `argsort`, `top_k`, `quickselect`, `binary_search` |
| `rank` | Ranking with ties, percentile ranks | `rank`, `percentile_ranks` |
| `stats` | Descriptive stats, streaming Welford, histogram, percentiles | `mean`, `median`, `std`, `minimum`, `maximum`, `percentile`, `histogram`, `WelfordStats` |
| `metrics` | Classification/regression metrics | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse` |
| `optim` | Finite-difference gradient and Jacobian, line search | `finite_diff_gradient`, `finite_diff_jacobian`, `line_search` |
| `pipeline` | Transformer / Estimator API, composition | `Transformer`, `Estimator`, `Compose`, `FeatureUnion` |
| `utils` | Distances, activations, logsumexp, batching | `euclidean_distance`, `manhattan_distance`, `cosine_similarity`, `pairwise_distances`, `sigmoid`, `relu`, `tanh`, `softmax`, `logsumexp`, `make_batches` |
| `benchmarking` | Micro-benchmark harness | `timer`, `compare`, `print_table`, `run_all` |

All functions document parameter shapes, return shapes, exceptions raised, and
time/space complexity in their docstrings.

## Design notes

- **Vectorisation.** Core computations avoid Python loops where possible. Loops
  are only used where they aid clarity for low-volume inner work (e.g. tie-block
  rank assignment, where `n_unique` is typically small).
- **Numerical stability.** `softmax` and `logsumexp` use the max-shift trick;
  `pairwise_distances` clamps the squared form to non-negative values and
  forces an exact-zero diagonal; descriptive stats and the imputer ignore NaN.
- **API consistency.** Functions raise `ValueError` for shape or value problems
  and `RuntimeError` when a fitted estimator is used before `.fit(...)`.
  Error messages are namespaced (e.g. `stats.mean: ...`).

## Running the tests

```bash
pip install pytest
python -m pytest tests/
```

The suite covers the rubric's required edge cases: empty arrays, all-equal
values, duplicates and ties, extreme `k`, NaNs, and non-contiguous strides.

## Demo

`demo/quickstart.ipynb` walks through an end-to-end flow: read CSV with missing
values, preprocess, rank, compute statistics, estimate gradients, and benchmark
vectorised vs. Python-loop implementations.

## Benchmarks

See [`benchmark/`](benchmark/) for reproducible loop-vs-vectorised comparisons
and a performance summary.

## Repository layout

```
NumCompute/
├── numcompute/        # the package
├── tests/             # unit tests
├── demo/              # quickstart notebook
├── benchmark/         # loop vs. vectorised benchmarks
├── README.md
└── pyproject.toml
```

## Contributing

1. Clone the repo and `cd` into the project directory.
2. Install in editable mode: `pip install -e .`
3. Add or edit modules under `numcompute/`.
4. Add tests under `tests/` using the file-name pattern `test_<module>.py`.
5. Run the suite with `python -m pytest tests/`.
6. Update `.gitignore` if needed before committing.
