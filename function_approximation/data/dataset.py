
"""PyTorch dataset and dataloader utilities."""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from function_approximation.data.generation import Dataset


class FunctionDataset(TorchDataset):
    """Store scalar inputs and targets as float32 column tensors.

    Args:
        x: Array-like input observations, flattened to ``(N, 1)``.
        y: Array-like targets with the same flattened sample count.

    Raises:
        ValueError: Input/target counts differ or either tensor is nonfinite
            after conversion to float32.

    Notes:
        No scaling or splitting happens here. Fit preprocessing on training
        data before constructing datasets. Indexing returns an input/target
        pair, each of shape ``(1,)`` for a scalar index.
    """

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

        if self.x.shape != self.y.shape:
            raise ValueError("Input and target shapes must match.")

        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y).all():
            raise ValueError("Dataset contains non-finite values.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Wrap already-preprocessed splits in reproducible PyTorch loaders.

    Args:
        dataset: Mapping with ``train``, ``val``, and ``test`` keys, each
            containing an ``(x, y)`` pair of scalar observation arrays.
        batch_size: Positive maximum number of observations per batch.
        seed: Seed for the training loader's independent shuffle generator.

    Returns:
        A mapping of split names to loaders yielding float32 tensors of shape
        ``(batch_size, 1)``. Only training is shuffled; final batches can be smaller.

    Raises:
        ValueError: Batch size or dataset values are invalid.
        KeyError: A required split is absent.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    loaders = {}

    for split in ("train", "val", "test"):
        x, y = dataset[split]

        generator = torch.Generator().manual_seed(seed)

        loaders[split] = DataLoader(
            FunctionDataset(x, y),
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            num_workers=0,
        )

    return loaders
