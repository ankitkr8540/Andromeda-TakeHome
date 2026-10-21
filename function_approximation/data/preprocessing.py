
"""Data preprocessing for function approximation."""

from typing import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from function_approximation.data.generation import Dataset


class DataPreprocessor:
    """Leakage-safe preprocessing for synthetic regression data."""

    def __init__(
        self,
        scaling: Literal["standard", "none"] = "standard",
    ) -> None:
        if scaling not in ("standard", "none"):
            raise ValueError(f"Unsupported scaling method: {scaling}")

        self.scaling = scaling
        self.scaler = StandardScaler() if scaling == "standard" else None
        self.is_fitted = False

    def fit(self, x_train: np.ndarray) -> "DataPreprocessor":
        """Fit input scaling using training data only."""
        x_train = self._prepare_array(x_train)

        if self.scaler is not None:
            self.scaler.fit(x_train)

        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform inputs using fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")

        x = self._prepare_array(x)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Convert scaled inputs back to original coordinates."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")

        x = self._prepare_array(x)

        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)

        return x

    def fit_transform_dataset(self, dataset: Dataset) -> Dataset:
        """Fit on training inputs and transform every split."""
        self.fit(dataset["train"][0])

        processed = {}

        for split, (x, y) in dataset.items():
            y = self._prepare_array(y)

            processed[split] = (
                self.transform(x),
                y.astype(np.float32),
            )

        return processed

    @property
    def mean(self) -> float | None:
        """Return the fitted training mean."""
        if not self.is_fitted or self.scaler is None:
            return None

        return float(self.scaler.mean_[0])

    @property
    def std(self) -> float | None:
        """Return the fitted input scaling factor."""
        if not self.is_fitted or self.scaler is None:
            return None

        return float(self.scaler.scale_[0])

    @staticmethod
    def _prepare_array(x: np.ndarray) -> np.ndarray:
        """Validate and reshape a single-feature array."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)

        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("Input must be nonempty and finite.")

        return x
