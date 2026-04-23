"""NumCompute package public API."""

__version__ = "0.1.0"

from .io import load_data
from .metrics import accuracy, confusion_matrix, f1, mse, precision, recall
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from .sort_search import binary_search, argsort, quickselect, sort, top_k
from .stats import WelfordStats, histogram, maximum, mean, median, minimum, percentile, std

__all__ = [
	"__version__",
	"load_data",
	"accuracy",
	"confusion_matrix",
	"f1",
	"mse",
	"precision",
	"recall",
	"Imputer",
	"MinMaxScaler",
	"OneHotEncoder",
	"StandardScaler",
	"binary_search",
	"argsort",
	"quickselect",
	"sort",
	"top_k",
	"WelfordStats",
	"histogram",
	"maximum",
	"mean",
	"median",
	"minimum",
	"percentile",
	"std",
]
