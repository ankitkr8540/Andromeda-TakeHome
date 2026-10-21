
"""PyTorch dataset and dataloader utilities."""

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from function_approximation.data.generation import Dataset


class FunctionDataset(TorchDataset):
    """Dataset for one-dimensional function approximation."""

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
    """Create reproducible train, validation, and test loaders."""

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
