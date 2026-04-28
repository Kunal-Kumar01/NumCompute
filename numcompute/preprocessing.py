"""Preprocessing Module

Implements StandardScaler, MinMaxScaler, Imputer, and OneHotEncoder
using vectorised NumPy operations.
"""

import numpy as np


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Formula: z = (x - mean) / std

    Attributes:
        mean_ (np.ndarray): Per-feature mean, shape (n_features,).
        std_ (np.ndarray): Per-feature std, shape (n_features,).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Raises:
            ValueError: If X is not 2-D.

        Complexity:
            Time: O(n * m)  Space: O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise X using fitted mean and std.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Standardised array, shape (n_samples, n_features).

        Raises:
            RuntimeError: If called before fit().

        Complexity:
            Time: O(n * m)  Space: O(n * m)
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Standardised array, shape (n_samples, n_features).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardisation.

        Args:
            X (np.ndarray): Scaled data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Original scale data, shape (n_samples, n_features).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return np.asarray(X, dtype=float) * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to the [0, 1] range.

    Formula: x_scaled = (x - min) / (max - min)

    Attributes:
        min_ (np.ndarray): Per-feature minimum, shape (n_features,).
        max_ (np.ndarray): Per-feature maximum, shape (n_features,).
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute per-feature min and max.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Raises:
            ValueError: If X is not 2-D.

        Complexity:
            Time: O(n * m)  Space: O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to [0, 1] using fitted min and max.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Scaled array, shape (n_samples, n_features).

        Raises:
            RuntimeError: If called before fit().

        Complexity:
            Time: O(n * m)  Space: O(n * m)
        """
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        return (X - self.min_) / scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Scaled array, shape (n_samples, n_features).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the min-max scaling.

        Args:
            X (np.ndarray): Scaled data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Original scale data, shape (n_samples, n_features).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.min_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        return np.asarray(X, dtype=float) * scale + self.min_


class Imputer:
    """Fill missing (NaN) values using a chosen strategy.

    Args:
        strategy (str): One of 'mean', 'median', 'mode'. Default 'mean'.

    Attributes:
        fill_values_ (np.ndarray): Fill value per feature, shape (n_features,).
    """

    def __init__(self, strategy: str = "mean"):
        valid = {"mean", "median", "mode"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{strategy}'.")
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X: np.ndarray) -> "Imputer":
        """Learn fill values from training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Raises:
            ValueError: If X is not 2-D.

        Complexity:
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
            # Build a masked array to ignore NaNs, then find the most frequent
            # value per column using bincount on ranked indices — fully vectorised
            # per column via np.apply_along_axis.
            def _col_mode(col):
                col = col[~np.isnan(col)]
                if col.size == 0:
                    return 0.0
                values, counts = np.unique(col, return_counts=True)
                return values[np.argmax(counts)]
            self.fill_values_ = np.apply_along_axis(_col_mode, 0, X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaNs with the fitted fill values.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Array with NaNs replaced, shape (n_samples, n_features).

        Raises:
            RuntimeError: If called before fit().

        Complexity:
            Time: O(n * m)  Space: O(n * m)
        """
        if self.fill_values_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float).copy()
        # Build boolean mask (n_samples, n_features) then use np.where
        # to broadcast fill values — no Python loops.
        nan_mask = np.isnan(X)
        fill_matrix = np.where(nan_mask, self.fill_values_[np.newaxis, :], X)
        return fill_matrix

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Array with NaNs replaced, shape (n_samples, n_features).
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categorical columns as one-hot binary arrays.

    Args:
        dtype (np.dtype): Output dtype. Default np.float64.

    Attributes:
        categories_ (list of np.ndarray): Unique categories per feature column.
    """

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.categories_ = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Learn unique categories per column.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Raises:
            ValueError: If X is not 2-D.

        Complexity:
            Time: O(n * m)  Space: O(m * c) where c = max unique categories.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        self.categories_ = [np.unique(X[:, col]) for col in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert categorical columns to one-hot encoding.

        Each column is expanded into len(categories) binary columns.
        Uses fully vectorised broadcasting — no Python loops over samples.

        Args:
            X (np.ndarray): Data to encode, shape (n_samples, n_features).

        Returns:
            np.ndarray: One-hot array, shape (n_samples, sum of category counts).

        Raises:
            RuntimeError: If called before fit().

        Complexity:
            Time: O(n * m * c)  Space: O(n * m * c)
        """
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X)
        # Stack all columns: X[:, col] is (n,1), cats is (1, c) → broadcast
        # gives (n, c) boolean matrix per column — vectorised, no sample loops.
        parts = [
            (X[:, col, np.newaxis] == cats[np.newaxis, :]).astype(self.dtype)
            for col, cats in enumerate(self.categories_)
        ]
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: One-hot array, shape (n_samples, sum of category counts).
        """
        return self.fit(X).transform(X)