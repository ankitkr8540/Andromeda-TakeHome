"""Baseline MLP for one-dimensional function approximation."""

import torch
from torch import nn

from function_approximation.models.snake_mlp import Snake


class MLPRegressor(nn.Module):
    """Configurable hidden widths and activations with a linear regression output.

    hidden_dims sets each hidden width, e.g. [64, 256, 128]. The default
    is [64, 64]. Use the same fitted preprocessing for training and inference.
    """

    model_type = "mlp"

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        output_dim: int = 1,
        activation: str = "tanh",
    ) -> None:
        """Build linear hidden layers with the requested nonlinear activation.

        Args:
            input_dim: Positive number of features per input observation.
            hidden_dims: Nonempty positive hidden widths; defaults to ``[64, 64]``.
            output_dim: Positive number of outputs per observation.
            activation: ``tanh``, ``relu``, ``silu``, or trainable per-neuron ``snake``.

        Raises:
            ValueError: Dimensions or activation name are invalid.

        Notes:
            The final layer is linear. Inputs must already use the experiment's
            preprocessing; no scaling or target evaluation happens in this model."""
        super().__init__()

        hidden_dims = [64, 64] if hidden_dims is None else hidden_dims
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError(
                "hidden_dims must be a nonempty list of positive integers."
            )
        if any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in (input_dim, output_dim, *hidden_dims)
        ):
            raise ValueError("Model dimensions must be positive integers.")

        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "snake": Snake,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of {list(activations)}.")

        layers = []
        in_features = input_dim

        for width in hidden_dims:
            nonlinearity = (
                Snake(width) if activation == "snake" else activations[activation]()
            )
            layers.extend([nn.Linear(in_features, width), nonlinearity])
            in_features = width

        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the MLP on preprocessed input tensors.

        Args:
            x: Tensor shaped ``(N, input_dim)`` on the model's device.

        Returns:
            Tensor shaped ``(N, output_dim)`` with a linear output activation."""
        return self.network(x)
