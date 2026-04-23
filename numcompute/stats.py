import numpy as np

class WelfordStats:
    """Streaming mean, variance, std, min, and max using Welford's algorithm.
    
    Processes values one at a time or in batches without storing the
    full dataset, making it suitable for large files processed in chunks.
    NaN values are silently ignored.

    Attributes:

        n (int): Number of non-NaN values seen so far.
        mean (float): Running mean estimate.
        M2 (float): Running sum of squared deviations from the running mean.
        min (float): Current minimum value.
        max (float): Current maximum value.

    Complexity:
        Time Complexity per update: O(1)   
        Space Complexity per Update: O(1)
    """

    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0
        self.min  = float('inf')
        self.max  = float('-inf')

    def update(self, x: float) -> None:
        """Updates the existing statistics with a new value.

        Args: 
            x(float): The next observed value. NaN values are silently skipped.
        """
        if np.isnan(x):
            return 
        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (x - self.mean)
        self.min   = x if x < self.min else self.min
        self.max   = x if x > self.max else self.max

    def update_batch(self, arr: np.ndarray) -> None:
        """Updates the existing statistics with new chunk of values.

        Args: 
            arr (np.ndarray): Array of values to incorporate. NaN values are skipped.
        """
        # The loop here is intentional — Welford's algorithm is sequential
        # by nature, so vectorisation is not possible for this step.
        for x in np.asarray(arr, dtype=float).ravel():
            self.update(float(x))

    @property
    def variance(self) -> float:
        """Population variance (divided by n).

        Returns:
            Udpated value of variance.
        """
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation.

        Returns:
            Updated value of standard deviation.
        """
        return float(np.sqrt(self.variance))

    def summary(self) -> dict:
        """Return a dictionary of the current running statistics.

        Returns:
            Updated statistics including Mean, Standard Deviation, Minimum, and Maximum.
        """
        return {
            "n":    self.n,
            "mean": round(self.mean, 4),
            "std":  round(self.std,  4),
            "min":  self.min,
            "max":  self.max,
        }

def mean(X: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Compute the arithmetic mean, ignoring NaN values.

    Args:
        X (np.ndarray): Input array of any shape.
        axis (int | None): Axis along which the mean is computed. Defaults to None.

    Returns:
        np.ndarray or float: Mean value(s). Returns a scalar when axis=None,
        otherwise an array with the chosen axis removed.

    Raises:
        ValueError: If X is empty.

    Complexity:
        Time: O(n)  Space: O(1) for scalar, O(m) for axis-wise result.

    Examples:
        >>> X = np.array([[1.0, 2.0], [3.0, np.nan]])
        >>> mean(X, axis=0)   # column means → array([2., 2.])
        >>> mean(X, axis=1)   # row means    → array([1.5, 3.])
        >>> mean(X)           # grand mean   → 2.0
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("stats.mean: input array is empty.")
    return np.nanmean(X, axis=axis)


def median(X: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Compute the median, ignoring NaN values.

    The median is the middle value of sorted data. For an even number
    of values it is the average of the two middle values. It is more
    robust to outliers than the mean.

    Args:
        X (np.ndarray): Input array of any shape.
        axis (int | None): Axis along which the median is computed. Defaults to None.

    Returns:
        np.ndarray or float: Median value(s). Returns a scalar when axis=None,
        otherwise an array with the chosen axis removed.

    Raises:
        ValueError: If X is empty.

    Complexity:
        Time: O(n log n)  Space: O(n) — median requires an internal sort.

    Examples:
        >>> X = np.array([[3.0, 1.0], [4.0, np.nan]])
        >>> median(X, axis=0)   # column medians → array([3.5, 1.])
        >>> median(X)           # grand median   → 3.0
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("stats.median: input array is empty.")
    return np.nanmedian(X, axis=axis)


def std(X: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Compute the population standard deviation, ignoring NaN values.

    Standard deviation measures how spread out values are around the mean.
    A small std means values are tightly clustered; a large std means
    they are widely spread. Uses population std (divides by n).

    Args:
        X (np.ndarray): Input array of any shape.
        axis (int | None): Axis along which std is computed. Defaults to None.

    Returns:
        np.ndarray or float: Standard deviation value(s). Returns a scalar
        when axis=None, otherwise an array with the chosen axis removed.

    Raises:
        ValueError: If X is empty.

    Complexity:
        Time: O(n)  Space: O(1) for scalar, O(m) for axis-wise result.

    Examples:
        >>> X = np.array([[2.0, 4.0], [4.0, 6.0]])
        >>> std(X, axis=0)   # column stds → array([1., 1.])
        >>> std(X, axis=1)   # row stds    → array([1., 1.])
        >>> std(X)           # grand std   → 1.4142...
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("stats.std: input array is empty.")
    return np.nanstd(X, axis=axis)


def minimum(X: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Return the minimum value, ignoring NaN values.

    Args:
        X (np.ndarray): Input array of any shape.
        axis (int | None): Axis along which the minimum is found. Defaults to None.

    Returns:
        np.ndarray or float: Minimum value(s). Returns a scalar when axis=None,
        otherwise an array with the chosen axis removed.

    Raises:
        ValueError: If X is empty.

    Complexity:
        Time: O(n)  Space: O(1) for scalar, O(m) for axis-wise result.

    Examples:
        >>> X = np.array([[3.0, np.nan], [1.0, 5.0]])
        >>> minimum(X, axis=0)   # column mins → array([1., 5.])
        >>> minimum(X)           # grand min   → 1.0
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("stats.minimum: input array is empty.")
    return np.nanmin(X, axis=axis)


def maximum(X: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Return the maximum value, ignoring NaN values.

    Args:
        X (np.ndarray): Input array of any shape.
        axis (int | None): Axis along which the maximum is found. Defaults to None. 

    Returns:
        np.ndarray or float: Maximum value(s). Returns a scalar when axis=None,
        otherwise an array with the chosen axis removed.

    Raises:
        ValueError: If X is empty.

    Complexity:
        Time: O(n)  Space: O(1) for scalar, O(m) for axis-wise result.

    Examples:
        >>> X = np.array([[3.0, np.nan], [1.0, 5.0]])
        >>> maximum(X, axis=0)   # column maxes → array([3., 5.])
        >>> maximum(X)           # grand max    → 5.0
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("stats.maximum: input array is empty.")
    return np.nanmax(X, axis=axis)

def histogram(
    X: np.ndarray,
    bins: int | np.ndarray = 10,
    range_: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 1D histogram of the input data, ignoring NaN values.

    The input is flattened before computation, so this function treats all
    values in the array as one collection of samples. This is consistent with
    NumPy's default histogram behavior.

    Args:
        X (np.ndarray): Input array of any shape.
        bins (int | np.ndarray): Number of equal-width bins or explicit bin edges.
            If an integer is provided, the bins are computed over the data range
            or over the given `range`.
        range_ (tuple[float, float] | None): Lower and upper range of the bins.
            Used only when `bins` is an integer. If None, the range is inferred
            from the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple `(counts, bin_edges)` where:
            - `counts` is an array of length `len(bin_edges) - 1`
            - `bin_edges` defines the bin boundaries

    Raises:
        ValueError: If the input is empty after removing NaN values, or if
            `bins` is invalid, or if `range` is invalid.

    Complexity:
        Time: O(n + b), where n is the number of input values and b is the
        number of bins.
        Space: O(n + b)

    Examples:
        >>> X = np.array([1.0, 2.0, 2.0, 3.0, np.nan, 4.0])
        >>> counts, edges = histogram(X, bins=3)
        >>> counts
        array([...])
        >>> edges
        array([...])
    """
    X = np.asarray(X, dtype=float)

    if X.size == 0:
        raise ValueError("stats.histogram: input array is empty.")

    X = X.ravel()
    X = X[~np.isnan(X)]

    if X.size == 0:
        raise ValueError("stats.histogram: input contains only NaN values.")

    counts, bin_edges = np.histogram(X, bins=bins, range=range_)
    return counts, bin_edges

def percentile(
    X: np.ndarray,
    q: float | list[float],
    axis: int | None = None,
    interpolation: str = "linear"
) -> np.ndarray | float:
    """Compute the q-th percentile(s) of the data, ignoring NaN values.

    Args:
        X (np.ndarray): Input array of any shape.
        q (float | list[float]): Percentile(s) to compute, between 0 and 100.
        axis (int | None): Axis along which percentiles are computed.
            Defaults to None (flatten and compute a single percentile).
        interpolation (str): Method to interpolate when q does not land on
            a data point. Options: 'linear', 'lower', 'higher', 'midpoint',
            'nearest'. Defaults to 'linear'.

    Returns:
        np.ndarray or float: Percentile value(s). Returns a scalar when
        axis=None and q is a scalar, otherwise returns an array with the
        chosen axis removed.

    Raises:
        ValueError: If X is empty, q is outside [0, 100], or interpolation
            method is unsupported.

    Complexity:
        Time: O(n log n)  Space: O(n) — requires sorting internally.

    Examples:
        >>> X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]])
        >>> percentile(X, 50)               # grand median        → 3.0
        >>> percentile(X, 50, axis=0)       # column medians      → array([2.5, 3.5, 3.0])
        >>> percentile(X, [25, 75])         # two percentiles     → array([1.75, 4.25])
        >>> percentile(X, [25, 75], axis=1) # row quartiles       → 2D array
    """
    X = np.asarray(X, dtype=float)

    if X.size == 0:
        raise ValueError("stats.percentile: input array is empty.")

    valid_interpolations = {"linear", "lower", "higher", "midpoint", "nearest"}
    if interpolation not in valid_interpolations:
        raise ValueError(
            f"stats.percentile: interpolation must be one of "
            f"{valid_interpolations}, got '{interpolation}'."
        )

    scalar_input = isinstance(q, (int, float))

    if scalar_input:
        q = [float(q)]
    else:
        q = [float(qi) for qi in q]

    if not all(0 <= qi <= 100 for qi in q):
        raise ValueError(
            "stats.percentile: all values in q must be between 0 and 100."
        )

    result = np.nanpercentile(X, q, axis=axis, method=interpolation)
    if scalar_input and axis is None:
        return float(result[0])

    return result