
"""Baseline MLP for one-dimensional function approximation."""

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Configurable Linear/Tanh hidden layers with a linear regression output.

    Args:
        input_dim: Number of input features; the experiment pipeline uses one.
        hidden_dim: Width of every hidden layer.
        num_hidden_layers: Number of Linear/Tanh hidden-layer pairs.
        output_dim: Number of outputs; the pipeline uses one scalar target.

    Raises:
        ValueError: A dimension or hidden-layer count is nonpositive.

    Notes:
        Tanh supplies the nonlinearity. The output has no activation, allowing
        regression values outside a bounded interval. Inputs must use the same
        preprocessing at training and inference time.
    """

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
        """Map a batch of inputs to regression values.

        Args:
            x: Tensor of shape ``(N, input_dim)`` on the model's device.

        Returns:
            Tensor of shape ``(N, output_dim)``. The shared Trainer requires
            the target tensor to have exactly this shape.
        """
        return self.network(x)
