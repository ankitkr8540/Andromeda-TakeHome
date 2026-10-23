"""Training-fitted input scaling for scalar function approximation."""

from typing import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from function_approximation.data.generation import Dataset


class DataPreprocessor:
    """Standardize inputs or preserve their coordinates; targets stay unscaled.

    Persist this object with the model and reuse it for every evaluation region.
    Only fit() or fit_transform_dataset() update training statistics.
    """

    def __init__(self, scaling: Literal["standard", "none"] = "standard") -> None:
        if scaling not in ("standard", "none"):
            raise ValueError(f"Unsupported scaling method: {scaling}")
        self.scaler = StandardScaler() if scaling == "standard" else None
        self.is_fitted = False

    @staticmethod
    def _prepare_array(values: np.ndarray) -> np.ndarray:
        """Accept a scalar, vector, or scalar column without flattening features."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim > 2 or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError("Expected scalar samples with shape (N,) or (N, 1).")
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Samples must be nonempty and finite.")
        return values.reshape(-1, 1)

    def fit(self, x_train: np.ndarray) -> "DataPreprocessor":
        """Fit input statistics on training observations only."""
        x_train = self._prepare_array(x_train)
        if self.scaler is not None:
            self.scaler.fit(x_train)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transform and return a float32 column."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        x = self._prepare_array(x)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x.astype(np.float32)

    def transform_dataset(self, dataset: Dataset) -> Dataset:
        """Transform any supplied splits without fitting, including extrapolation."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        processed = {}
        for split, (x, y) in dataset.items():
            x, y = self.transform(x), self._prepare_array(y)
            if x.shape != y.shape:
                raise ValueError(
                    f"Input and target shapes must match in split '{split}'."
                )
            processed[split] = (x, y.astype(np.float32))
        return processed

    def fit_transform_dataset(self, dataset: Dataset) -> Dataset:
        """Fit on train inputs, then transform all supplied splits."""
        self.fit(dataset["train"][0])
        return self.transform_dataset(dataset)

    @property
    def mean(self) -> float | None:
        """Fitted training mean, or None when scaling is disabled/unfitted."""
        if not self.is_fitted or self.scaler is None:
            return None
        return float(self.scaler.mean_[0])

    @property
    def std(self) -> float | None:
        """Fitted training scale, or None when scaling is disabled/unfitted."""
        if not self.is_fitted or self.scaler is None:
            return None
        return float(self.scaler.scale_[0])
