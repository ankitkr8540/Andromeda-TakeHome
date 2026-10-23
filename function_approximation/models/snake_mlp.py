"""Standalone scalar regressor with trainable Snake hidden activations."""

import math

import torch
from torch import nn


class Snake(nn.Module):
    """z + sin²(a*z)/a, with one positive trainable a per neuron."""

    def __init__(self, n_features: int, a0: float = 1.0) -> None:
        """Initialize one positive learnable activation scale per hidden neuron.

        Args:
            n_features: Positive number of neuron channels in the input tensor.
            a0: Initial positive scale, greater than ``1e-6``.

        Raises:
            ValueError: Channel count or initial scale is invalid.

        Notes:
            Optimizes ``raw_alpha`` and derives ``a = softplus(raw_alpha) + 1e-6``.
            The inverse-softplus initialization makes the initial scale equal ``a0``."""
        super().__init__()
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 1
        ):
            raise ValueError("n_features must be a positive integer.")
        if isinstance(a0, bool) or not math.isfinite(a0) or a0 <= 1e-6:
            raise ValueError("a0 must be finite and greater than 1e-6.")
        value = float(a0) - 1e-6
        # Stable inverse softplus, including large initial values.
        raw = value + math.log(-math.expm1(-value))
        self.raw_alpha = nn.Parameter(torch.full((n_features,), raw))

    @property
    def alpha(self) -> torch.Tensor:
        """Return the differentiable positive per-neuron scale softplus(raw_alpha) + 1e-6."""
        return torch.nn.functional.softplus(self.raw_alpha) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Snake: x + sin²(a*x)/a, preserving the tensor shape.

        Args:
            x: Hidden activations whose last dimension equals ``n_features``.

        Returns:
            Tensor with the same shape and a learnable oscillatory component."""
        alpha = self.alpha
        return x + torch.sin(alpha * x).square() / alpha


class SnakeMLP(nn.Module):
    """Scalar MLP with Snake hidden layers and a linear output.

    Inputs are divided by x_scale, which defaults to 1 (raw coordinates).
    The experiment preprocessor should use scaling=none for raw-input runs.
    """

    model_type = "snake"

    def __init__(
        self,
        dims: list[int] | tuple[int, ...] = (64, 256, 128),
        a0: float = 1.0,
        x_scale: float = 1.0,
    ) -> None:
        """Build a scalar regressor with Snake hidden layers and a linear output.

        Args:
            dims: Nonempty positive hidden widths, defaulting to ``(64, 256, 128)``.
            a0: Initial positive activation scale for every hidden neuron.
            x_scale: Finite positive input divisor; 1.0 retains raw coordinates.

        Raises:
            ValueError: Hidden widths, input divisor, or activation scale are invalid.

        Notes:
            Each hidden neuron learns its own positive scale. Raw-input experiments
            also require ``preprocessing.scaling: none`` outside the model."""
        super().__init__()
        if (
            not isinstance(dims, (list, tuple))
            or not dims
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width < 1
                for width in dims
            )
        ):
            raise ValueError("dims must be a nonempty sequence of positive integers.")
        if isinstance(x_scale, bool) or not math.isfinite(x_scale) or x_scale <= 0:
            raise ValueError("x_scale must be finite and positive.")
        layers = []
        in_dim = 1
        for width in dims:
            layers.extend([nn.Linear(in_dim, width), Snake(width, a0)])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.x_scale = float(x_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar output after dividing inputs by the configured x_scale.

        Args:
            x: Input tensor of shape ``(N, 1)`` on the model's device.

        Returns:
            Prediction tensor of shape ``(N, 1)``; no target function is evaluated."""
        return self.net(x / self.x_scale)
