"""Pipeline Module

Provides Transformer and Estimator base classes, a Compose pipeline,
and a FeatureUnion for parallel transformations. Follows the same
fit/transform/predict API pattern used throughout NumCompute.
"""

import numpy as np


# ── Base Classes ──────────────────────────────────────────────────────────────

class Transformer:
    """Abstract base class for all data transformers.

    A transformer modifies input data without producing a label prediction.
    Subclasses must implement fit() and transform().

    All transformers support fit_transform() which calls fit then transform.
    """

    def fit(self, X: np.ndarray) -> "Transformer":
        """Learn parameters from training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement fit()."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to X.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()."
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return the transformed result in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X).transform(X)


class Estimator:
    """Abstract base class for all estimators (models that make predictions).

    An estimator learns a mapping from features to outputs.
    Subclasses must implement fit() and predict().

    Args:
        None

    Complexity:
        Depends on subclass implementation.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator":
        """Train the estimator on labelled data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y (np.ndarray): Target labels or values, shape (n_samples,).

        Returns:
            self

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement fit()."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions, shape (n_samples,).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement predict()."
        )


# ── Compose ───────────────────────────────────────────────────────────────────

class Compose:
    """Chain a sequence of transformers and an optional final estimator.

    Each step is either a Transformer (has fit/transform) or, for the last
    step only, an Estimator (has fit/predict). This mirrors the design of
    sklearn's Pipeline.

    Args:
        steps (list of tuple): List of (name, object) pairs where each object
            is a Transformer, except optionally the last which can be an Estimator.

    Raises:
        ValueError: If steps is empty or any intermediate step lacks transform().

    Examples:
        >>> from numcompute.preprocessing import StandardScaler, Imputer
        >>> pipe = Compose([('imputer', Imputer()), ('scaler', StandardScaler())])
        >>> X_clean = pipe.fit_transform(X)
    """

    def __init__(self, steps: list):
        if not steps:
            raise ValueError("pipeline.Compose: steps must not be empty.")
        for name, obj in steps:
            if not isinstance(name, str):
                raise ValueError(
                    f"pipeline.Compose: step names must be strings, got {type(name)}."
                )
        self.steps = steps
        self._fitted = False

    @property
    def _transformers(self):
        """All steps except the last."""
        return self.steps[:-1]

    @property
    def _final_step(self):
        """The last step (name, object)."""
        return self.steps[-1]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Compose":
        """Fit all steps in sequence.

        For all but the last step, calls fit_transform to pass data through.
        For the last step, calls fit(X) or fit(X, y) depending on whether
        it is a Transformer or Estimator.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray | None): Target labels, required if the last step
                is an Estimator. Default None.

        Returns:
            self

        Raises:
            ValueError: If an intermediate step is missing transform().
        """
        X_current = np.asarray(X, dtype=float)

        for name, step in self._transformers:
            if not hasattr(step, "transform"):
                raise ValueError(
                    f"pipeline.Compose: intermediate step '{name}' must have transform()."
                )
            X_current = step.fit_transform(X_current)

        name, last = self._final_step
        if hasattr(last, "predict"):
            if y is None:
                raise ValueError(
                    f"pipeline.Compose: final estimator '{name}' requires y in fit()."
                )
            last.fit(X_current, y)
        else:
            last.fit(X_current)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformer steps to X (no final estimator step).

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data after all transformer steps.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If the last step is an Estimator (use predict instead).
        """
        if not self._fitted:
            raise RuntimeError("pipeline.Compose: call fit() before transform().")
        _, last = self._final_step
        if hasattr(last, "predict"):
            raise ValueError(
                "pipeline.Compose: last step is an Estimator — use predict() instead."
            )
        X_current = np.asarray(X, dtype=float)
        for _, step in self.steps:
            X_current = step.transform(X_current)
        return X_current

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit all steps and transform X in one call.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X through all transformer steps then predict with the final estimator.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions from the final estimator, shape (n_samples,).

        Raises:
            RuntimeError: If called before fit().
            ValueError: If the last step is not an Estimator.
        """
        if not self._fitted:
            raise RuntimeError("pipeline.Compose: call fit() before predict().")
        _, last = self._final_step
        if not hasattr(last, "predict"):
            raise ValueError(
                "pipeline.Compose: last step has no predict() — use transform() instead."
            )
        X_current = np.asarray(X, dtype=float)
        for _, step in self._transformers:
            X_current = step.transform(X_current)
        return last.predict(X_current)

    def __repr__(self) -> str:
        step_str = ", ".join(f"('{n}', {type(s).__name__})" for n, s in self.steps)
        return f"Compose([{step_str}])"


# ── FeatureUnion ──────────────────────────────────────────────────────────────

class FeatureUnion:
    """Apply multiple transformers in parallel and concatenate their outputs.

    Each transformer receives the same input X. Their transformed outputs
    are concatenated column-wise to form a single feature matrix.

    Args:
        transformer_list (list of tuple): List of (name, Transformer) pairs.

    Raises:
        ValueError: If transformer_list is empty or any step lacks transform().

    Examples:
        >>> from numcompute.preprocessing import StandardScaler, MinMaxScaler
        >>> fu = FeatureUnion([('std', StandardScaler()), ('mm', MinMaxScaler())])
        >>> X_combined = fu.fit_transform(X)
        >>> X_combined.shape[1] == X.shape[1] * 2
        True
    """

    def __init__(self, transformer_list: list):
        if not transformer_list:
            raise ValueError("pipeline.FeatureUnion: transformer_list must not be empty.")
        for name, t in transformer_list:
            if not hasattr(t, "transform"):
                raise ValueError(
                    f"pipeline.FeatureUnion: '{name}' must have a transform() method."
                )
        self.transformer_list = transformer_list
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureUnion":
        """Fit all transformers on X independently.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            self

        Complexity:
            Time: O(k * n * m) where k = number of transformers  Space: O(k * m)
        """
        X = np.asarray(X, dtype=float)
        for _, transformer in self.transformer_list:
            transformer.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X with all transformers and concatenate outputs column-wise.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Concatenated features,
                shape (n_samples, sum of output features per transformer).

        Raises:
            RuntimeError: If called before fit().

        Complexity:
            Time: O(k * n * m)  Space: O(k * n * m)
        """
        if not self._fitted:
            raise RuntimeError("pipeline.FeatureUnion: call fit() before transform().")
        X = np.asarray(X, dtype=float)
        parts = [t.transform(X) for _, t in self.transformer_list]
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit all transformers and return concatenated output in one step.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Concatenated features.
        """
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        step_str = ", ".join(f"('{n}', {type(t).__name__})" for n, t in self.transformer_list)
        return f"FeatureUnion([{step_str}])"
