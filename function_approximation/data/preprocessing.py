
"""Data preprocessing for function approximation."""

from typing import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from function_approximation.data.generation import Dataset


class DataPreprocessor:
    """Fit an input transform on training observations and reuse it unchanged.

    Args:
        scaling: ``"standard"`` for training-mean/unit-scale inputs or
            ``"none"`` to preserve original coordinates.

    Raises:
        ValueError: The scaling mode is unsupported.

    Notes:
        Targets are never standardized. Persist this object alongside a model.
    """

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
        """Learn input statistics from a nonempty, finite training array.

        Args:
            x_train: Scalar training inputs, flattened to a column.

        Returns:
            This fitted preprocessor, including when scaling is disabled.

        Raises:
            ValueError: Inputs are empty or nonfinite.
        """
        x_train = self._prepare_array(x_train)

        if self.scaler is not None:
            self.scaler.fit(x_train)

        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted input transform without updating its statistics.

        Args:
            x: Nonempty finite scalar inputs, flattened to a column.

        Returns:
            Float32 array of shape ``(N, 1)`` for model input.

        Raises:
            RuntimeError: The preprocessor has not been fitted.
            ValueError: Inputs are empty or nonfinite before conversion.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")

        x = self._prepare_array(x)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Convert a column of model-input coordinates to original x units.

        Args:
            x: Nonempty finite model inputs, flattened to ``(N, 1)``.

        Returns:
            Float64 inputs in original coordinates. With scaling disabled,
            this validates and reshapes the supplied values.

        Raises:
            RuntimeError: The preprocessor has not been fitted.
            ValueError: Inputs are empty or nonfinite.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")

        x = self._prepare_array(x)

        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)

        return x

    def fit_transform_dataset(self, dataset: Dataset) -> Dataset:
        """Fit using ``dataset['train'][0]`` and transform each supplied split.

        Args:
            dataset: Mapping from split names to ``(x, y)`` arrays.

        Returns:
            A new mapping containing float32 input/target columns. Only x is
            scaled; targets retain their original units. Validation and test
            inputs do not contribute to fitted statistics.

        Raises:
            KeyError: The training split is missing.
            ValueError: An input or target array is empty or nonfinite.
        """
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
