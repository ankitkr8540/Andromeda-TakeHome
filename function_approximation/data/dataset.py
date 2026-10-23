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
    """Multi-harmonic periodic benchmark."""
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
        self.x = self._column(x)
        self.y = self._column(y)
        if self.x.shape != self.y.shape:
            raise ValueError("Input and target shapes must match.")

    @staticmethod
    def _column(values: ArrayLike | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim not in (1, 2) or (tensor.ndim == 2 and tensor.shape[1] != 1):
            raise ValueError("Expected scalar samples with shape (N,) or (N, 1).")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError("Dataset samples must be nonempty and finite in float32.")
        return tensor.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Load train/val/test splits, shuffling training only with a local seed.

    Extra splits, including extrapolation bands, are deliberately excluded.
    Evaluation order and partial final batches are preserved. Missing required
    splits raise KeyError; invalid samples or batch size raise ValueError.
    """
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
