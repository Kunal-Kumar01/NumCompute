"""Preprocessing Module

Implements StandardScaler, MinMaxScaler, Imputer, and OneHotEncoder
using vectorised NumPy operations.
"""

import numpy as np


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Formula: z = (x - mean) / std

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during fit.
    std_ : np.ndarray, shape (n_features,)
        Per-feature standard deviation computed during fit.

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Complexity
        ----------
        Time: O(n * m)  Space: O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise X using fitted mean and std.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Standardised data.

        Raises
        ------
        RuntimeError
            If transform is called before fit.

        Complexity
        ----------
        Time: O(n * m)  Space: O(n * m)
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardisation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return np.asarray(X, dtype=float) * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to a fixed range [feature_min, feature_max].

    Formula: x_scaled = (x - min) / (max - min)

    Attributes
    ----------
    min_ : np.ndarray, shape (n_features,)
    max_ : np.ndarray, shape (n_features,)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute per-feature min and max.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Complexity
        ----------
        Time: O(n * m)  Space: O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to [0, 1] range.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If transform is called before fit.

        Complexity
        ----------
        Time: O(n * m)  Space: O(n * m)
        """
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        # Constant features stay at 0
        scale[scale == 0] = 1.0
        return (X - self.min_) / scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the min-max scaling."""
        if self.min_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        return np.asarray(X, dtype=float) * scale + self.min_


class Imputer:
    """Fill missing (NaN) values using a chosen strategy.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'mode'. Default 'mean'.

    Attributes
    ----------
    fill_values_ : np.ndarray, shape (n_features,)
        The value used to fill each feature column.
    """

    def __init__(self, strategy: str = "mean"):
        valid = {"mean", "median", "mode"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{strategy}'.")
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X: np.ndarray) -> "Imputer":
        """Learn fill values from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Complexity
        ----------
        Time: O(n * m)  Space: O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")

        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        elif self.strategy == "mode":
            # Mode: most frequent value per column (ignoring NaN)
            fill = np.zeros(X.shape[1])
            for col in range(X.shape[1]):
                col_data = X[:, col]
                col_data = col_data[~np.isnan(col_data)]
                if col_data.size == 0:
                    fill[col] = 0.0
                else:
                    values, counts = np.unique(col_data, return_counts=True)
                    fill[col] = values[np.argmax(counts)]
            self.fill_values_ = fill
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaNs with the fitted fill values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Array with no NaN values.

        Raises
        ------
        RuntimeError
            If transform is called before fit.

        Complexity
        ----------
        Time: O(n * m)  Space: O(n * m)
        """
        if self.fill_values_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float).copy()
        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col])
            X[nan_mask, col] = self.fill_values_[col]
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode integer categorical columns as one-hot binary arrays.

    Parameters
    ----------
    dtype : np.dtype
        Output dtype. Default np.float64.

    Attributes
    ----------
    categories_ : list of np.ndarray
        Unique categories found per feature column during fit.
    """

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.categories_ = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Learn unique categories per column.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Complexity
        ----------
        Time: O(n * m)  Space: O(m * c) where c = max unique categories
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.categories_ = [np.unique(X[:, col]) for col in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert categorical columns to one-hot encoding.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, sum of unique categories per feature)
            One-hot encoded array.

        Raises
        ------
        RuntimeError
            If transform is called before fit.

        Complexity
        ----------
        Time: O(n * m * c)  Space: O(n * m * c)
        """
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X)
        parts = []
        for col, cats in enumerate(self.categories_):
            # Vectorised comparison: (n_samples, n_categories)
            one_hot = (X[:, col:col+1] == cats[np.newaxis, :]).astype(self.dtype)
            parts.append(one_hot)
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)