"""Periodic benchmark labels and reproducible PyTorch datasets/batches."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from function_approximation.data.generation import Dataset


def rich_periodic(x: np.ndarray) -> np.ndarray:
    """Evaluate the bounded, multi-harmonic reference function.

    Args:
        x: NumPy input coordinates in original units; scalar or array-shaped.

    Returns:
        Target values with the same shape as ``x``. The six angular frequencies
        are 1, 2, 3, 5, 7, and 11, giving a fundamental period of 2π.

    Notes:
        Used to generate labels and reference curves only. Trained models and
        saved-model inference must not call this function."""
    return (
        np.sin(x)
        + 0.55 * np.cos(2 * x)
        + 0.35 * np.sin(3 * x + 0.4)
        + 0.20 * np.cos(5 * x - 0.7)
        + 0.12 * np.sin(7 * x)
        + 0.08 * np.cos(11 * x + 0.2)
    )


class FunctionDataset(TorchDataset):
    """Float32 input/target columns; preprocessing and splitting happen upstream."""

    def __init__(
        self, x: ArrayLike | torch.Tensor, y: ArrayLike | torch.Tensor
    ) -> None:
        """Validate and store matching input/target columns for PyTorch.

        Args:
            x: Nonempty finite vector or column of scalar inputs.
            y: Matching nonempty finite vector or column of target values.

        Raises:
            ValueError: Inputs have multiple features, mismatched sample counts,
                or values that are empty or nonfinite after float32 conversion.

        Notes:
            Arrays become float32 tensors of shape ``(N, 1)``. No splitting or
            scaling occurs here. Compatible arrays/tensors may share storage."""
        self.x = self._column(x)
        self.y = self._column(y)
        if self.x.shape != self.y.shape:
            raise ValueError("Input and target shapes must match.")

    @staticmethod
    def _column(values: ArrayLike | torch.Tensor) -> torch.Tensor:
        """Convert a finite scalar vector/column to a float32 tensor of shape (N, 1)."""
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim not in (1, 2) or (tensor.ndim == 2 and tensor.shape[1] != 1):
            raise ValueError("Expected scalar samples with shape (N,) or (N, 1).")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError("Dataset samples must be nonempty and finite in float32.")
        return tensor.reshape(-1, 1)

    def __len__(self) -> int:
        """Return the number of input-target pairs available to a DataLoader."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and target tensors for an index; each has shape (1,)."""
        return self.x[index], self.y[index]


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create reproducible training, validation, and interpolation-test loaders.

    Args:
        dataset: Mapping with ``train``, ``val``, and ``test`` entries, each
            containing matching ``(x, y)`` arrays or tensors.
        batch_size: Positive integer number of observations per batch.
        seed: Seed for the training loader's local shuffle generator.

    Returns:
        Mapping of the three split names to PyTorch DataLoaders. Only training
        is shuffled; evaluation order and partial final batches are preserved.

    Raises:
        KeyError: A required split is missing.
        ValueError: Batch size or dataset shapes/values are invalid.

    Notes:
        Extra entries, including extrapolation bands, are excluded. Creating
        loaders does not fit preprocessing or modify the global random seed."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    shuffle_generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            FunctionDataset(*dataset[split]),
            batch_size=int(batch_size),
            shuffle=split == "train",
            generator=shuffle_generator if split == "train" else None,
            num_workers=0,
        )
        for split in ("train", "val", "test")
    }
