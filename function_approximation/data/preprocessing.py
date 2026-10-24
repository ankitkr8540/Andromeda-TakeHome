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
        """Choose input scaling without fitting any statistics.

        Args:
            scaling: ``standard`` fits training mean/scale; ``none`` preserves
                coordinates. Neither mode rescales target values.

        Raises:
            ValueError: The scaling method is unsupported."""
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

    @staticmethod
    def _as_float32(values: np.ndarray) -> np.ndarray:
        """Reject finite float64 values that overflow the model's dtype."""
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError("Samples must remain finite after conversion to float32.")
        return converted

    def fit(self, x_train: np.ndarray) -> "DataPreprocessor":
        """Fit the input transform using training observations only.

        Args:
            x_train: Nonempty finite scalar, vector, or column of training inputs.

        Returns:
            This preprocessor, allowing chained calls. Sets ``is_fitted`` to True;
            the ``none`` mode also requires this explicit fitting step.

        Raises:
            ValueError: Inputs are empty, nonfinite, or contain multiple features."""
        x_train = self._prepare_array(x_train)
        if self.scaler is not None:
            self.scaler.fit(x_train)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform original-coordinate inputs using the fitted training statistics.

        Args:
            x: Finite scalar, vector, or column of inputs from any evaluation region.

        Returns:
            Float32 column of shape ``(N, 1)``. Fitted statistics remain unchanged.

        Raises:
            RuntimeError: The preprocessor has not been fitted.
            ValueError: Inputs are empty, nonfinite, invalidly shaped, or overflow
                float32 after transformation."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        x = self._prepare_array(x)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return self._as_float32(x)

    def transform_dataset(self, dataset: Dataset) -> Dataset:
        """Transform supplied splits without refitting, including extrapolation bands.

        Args:
            dataset: Mapping of split names to matching ``(x, y)`` arrays.

        Returns:
            New split mapping with float32 input and target columns. Inputs use
            the fitted transform; targets are only reshaped and converted in dtype.

        Raises:
            RuntimeError: The preprocessor has not been fitted.
            ValueError: Arrays have incompatible shapes or invalid/overflowing values."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        processed = {}
        for split, (x, y) in dataset.items():
            x, y = self.transform(x), self._prepare_array(y)
            if x.shape != y.shape:
                raise ValueError(
                    f"Input and target shapes must match in split '{split}'."
                )
            processed[split] = (x, self._as_float32(y))
        return processed

    def fit_transform_dataset(self, dataset: Dataset) -> Dataset:
        """Fit on training inputs, transform splits, and verify model-input independence.

        Args:
            dataset: Split mapping with a required ``train`` entry and optional
                validation, test, or evaluation-only entries.

        Returns:
            New mapping of float32 input/target columns using training statistics.

        Raises:
            KeyError: The training split is missing.
            ValueError: Data are invalid, or train/validation/test inputs contain
                duplicates within or across splits after float32 preprocessing.

        Notes:
            Validation/test observations never affect fitted scaling statistics.
            Evaluation-only bands are transformed but excluded from split-overlap checks."""
        self.fit(dataset["train"][0])
        processed = self.transform_dataset(dataset)
        # Enforce independence in the exact coordinates consumed by the model.
        splits = [name for name in ("train", "val", "test") if name in processed]
        for index, name in enumerate(splits):
            x = processed[name][0]
            if np.unique(x).size != x.size:
                raise ValueError(
                    f"Split '{name}' contains duplicate inputs after float32 preprocessing."
                )
            for other in splits[:index]:
                if np.intersect1d(x, processed[other][0]).size:
                    raise ValueError(
                        f"Splits '{name}' and '{other}' overlap after float32 preprocessing."
                    )
        return processed

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
