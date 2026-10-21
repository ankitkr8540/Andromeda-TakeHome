"""Fourier regression with learned exponential amplitude envelopes."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from function_approximation.models.fourier import FourierRegressor


class DampedFourierRegressor(FourierRegressor):
    """Learn damped sinusoidal components from scalar observations.

    The model predicts ``c + Σ exp(-λₖ (x - x_ref)) [aₖ sin(ωₖx) +
    bₖ cos(ωₖx)]``. Frequencies, coefficients, and nonnegative damping rates
    are trainable. ``x_ref`` is fitted from training inputs only and stored as
    a checkpoint buffer.

    Args:
        num_frequencies: Number of independent damped components.
        frequency_min: Positive lower bound for frequency initialization.
        frequency_max: Upper bound for frequency initialization.
        initial_decay: Positive, generic starting decay rate. It is not a
            target-specific rate and remains trainable.
    """

    model_type = "damped_fourier"

    def __init__(
        self,
        num_frequencies: int = 8,
        frequency_min: float = 0.1,
        frequency_max: float = 5.0,
        initial_decay: float = 0.02,
    ) -> None:
        super().__init__(num_frequencies, frequency_min, frequency_max)
        if (
            isinstance(initial_decay, bool)
            or not isinstance(initial_decay, (int, float))
            or not math.isfinite(initial_decay)
            or initial_decay <= 0
        ):
            raise ValueError("initial_decay must be a finite positive number.")
        raw_decay = math.log(math.expm1(initial_decay))
        self.raw_decay_rates = nn.Parameter(
            torch.full((num_frequencies,), raw_decay)
        )
        self.register_buffer("x_reference", torch.zeros(1))
        self.initial_decay = float(initial_decay)

    @property
    def decay_rates(self) -> torch.Tensor:
        """Return nonnegative learned damping rates."""
        return F.softplus(self.raw_decay_rates)

    def initialize_from_data(
        self, x: np.ndarray, y: np.ndarray, *, grid_size: int = 4096
    ) -> dict:
        """Initialize frequencies from training data and anchor the envelope.

        The parent spectral procedure estimates a data-driven stationary
        starting spectrum. This method then sets ``x_reference`` to the minimum
        training coordinate so that decay is numerically stable during fitting
        and later checkpointed inference.
        """
        report = super().initialize_from_data(x, y, grid_size=grid_size)
        x_values = np.asarray(x, dtype=np.float64).ravel()
        with torch.no_grad():
            self.x_reference.copy_(self.x_reference.new_tensor([x_values.min()]))
        report["method"] = "spectral_damped"
        report["x_reference"] = float(self.x_reference.item())
        report["spectrum"] = self.spectrum()
        return report

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate learned damped components without accessing labels."""
        if x.ndim != 2 or x.shape[1] != 1 or len(x) == 0:
            raise ValueError(
                "Damped Fourier regression expects nonempty inputs of shape (N, 1)."
            )
        if not torch.isfinite(x).all():
            raise ValueError("Damped Fourier inputs must be finite.")
        angles = x * self.frequencies
        envelope = torch.exp(-(x - self.x_reference) * self.decay_rates)
        return (
            envelope
            * (
                torch.sin(angles) * self.sine_weights
                + torch.cos(angles) * self.cosine_weights
            )
        ).sum(dim=1, keepdim=True) + self.bias

    def spectrum(self) -> dict:
        """Return learned frequencies, amplitudes, damping rates, and reference."""
        components = (
            torch.stack(
                (
                    self.frequencies,
                    self.sine_weights,
                    self.cosine_weights,
                    torch.hypot(self.sine_weights, self.cosine_weights),
                    self.decay_rates,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        fields = (
            "frequency",
            "sine_weight",
            "cosine_weight",
            "amplitude",
            "decay_rate",
        )
        return {
            "bias": float(self.bias.detach().cpu().item()),
            "x_reference": float(self.x_reference.detach().cpu().item()),
            "components": [
                dict(zip(fields, values, strict=True))
                for values in sorted(components, key=lambda row: row[0])
            ],
        }
