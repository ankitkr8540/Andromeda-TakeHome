
"""Baseline MLP for one-dimensional function approximation."""

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Configurable multilayer perceptron for scalar regression."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if min(input_dim, hidden_dim, output_dim, num_hidden_layers) <= 0:
            raise ValueError("Model dimensions and layer count must be positive.")

        layers = []
        in_features = input_dim

        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.Tanh(),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict function values for input tensor of shape (N, 1)."""
        return self.network(x)
