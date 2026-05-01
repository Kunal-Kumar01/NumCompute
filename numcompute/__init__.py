"""NumCompute package public API."""

__version__ = "0.1.0"

from .io import load_data
from .metrics import accuracy, auc, confusion_matrix, f1, mse, precision, recall, roc_curve
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from .rank import percentile_ranks, rank
from .sort_search import binary_search, argsort, multi_key_sort, quickselect, sort, top_k
from .stats import WelfordStats, histogram, maximum, mean, median, minimum, percentile, std
from .utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    pairwise_distances,
    sigmoid,
    relu,
    softmax,
    tanh,
    logsumexp,
    make_batches,
)
from .optim import (
    finite_diff_gradient,
    finite_diff_jacobian,
    grad,
    jacobian,
    line_search,
)
from .pipeline import Transformer, Estimator, Compose, FeatureUnion, Pipeline
from .benchmarking import timer, compare, print_table, run_all

__all__ = [
    "__version__",
    # io
    "load_data",
    # metrics
    "accuracy", "auc", "confusion_matrix", "f1", "mse", "precision", "recall", "roc_curve",
    # preprocessing
    "Imputer", "MinMaxScaler", "OneHotEncoder", "StandardScaler",
    # rank
    "percentile_ranks", "rank",
    # sort_search
    "binary_search", "argsort", "multi_key_sort", "quickselect", "sort", "top_k",
    # stats
    "WelfordStats", "histogram", "maximum", "mean", "median",
    "minimum", "percentile", "std",
    # utils
    "euclidean_distance", "manhattan_distance", "cosine_similarity",
    "pairwise_distances", "sigmoid", "relu", "softmax", "tanh",
    "logsumexp", "make_batches",
    # optim
    "finite_diff_gradient", "finite_diff_jacobian", "grad", "jacobian", "line_search",
    # pipeline
    "Transformer", "Estimator", "Compose", "FeatureUnion", "Pipeline",
    # benchmarking
    "timer", "compare", "print_table", "run_all",
]