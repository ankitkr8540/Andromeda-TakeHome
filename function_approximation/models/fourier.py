"""Learnable Fourier regression for scalar observations."""

import math

import numpy as np
import torch
from torch import nn


class FourierRegressor(nn.Module):
    """Learn c + sum(a_k sin(w_k x) + b_k cos(w_k x)).

    Frequencies are independent trainable parameters. The configurable
    range controls initialization and spectral search; training can move outside it.
    Frequencies use radians per model-input unit, so input scaling changes
    their interpretation. No target frequencies or harmonic ratios are given.

    Args:
        num_frequencies: Number of independent components; eight use 25 parameters.
        frequency_min: Positive lower bound for initialization and spectral search.
        frequency_max: Upper bound, strictly greater than ``frequency_min``.

    Raises:
        ValueError: A component count or frequency bound is invalid.

    Notes:
        Construction initializes parameters randomly. Call
        :meth:`initialize_from_data` explicitly for spectral initialization.
        Forward prediction never estimates frequencies or consumes labels.
        Frequencies may move outside the initialization interval during training.
    """

    model_type = "fourier"

    def __init__(
        self,
        num_frequencies: int = 8,
        frequency_min: float = 0.1,
        frequency_max: float = 5.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_frequencies, bool)
            or not isinstance(num_frequencies, int)
            or num_frequencies <= 0
        ):
            raise ValueError("num_frequencies must be a positive integer.")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in (frequency_min, frequency_max)
            )
            or frequency_min >= frequency_max
        ):
            raise ValueError("Require finite 0 < frequency_min < frequency_max.")

        self.frequencies = nn.Parameter(
            torch.empty(num_frequencies).uniform_(frequency_min, frequency_max)
        )
        self.sine_weights = nn.Parameter(0.1 * torch.randn(num_frequencies))
        self.cosine_weights = nn.Parameter(0.1 * torch.randn(num_frequencies))
        self.bias = nn.Parameter(torch.zeros(1))
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max

    def initialize_from_data(
        self, x: np.ndarray, y: np.ndarray, *, grid_size: int = 4096
    ) -> dict:
        """Fit starting frequencies and coefficients using training data only.

        Inputs must use the same coordinates as forward(). Find Lomb–Scargle
        peaks in successive residuals, refitting linear coefficients after
        each component. Call before training; all parameters stay trainable.

        Args:
            x: Training inputs, shape ``(N,)`` or ``(N, 1)``. Apply the fitted
                input preprocessor first if the model uses standardized inputs.
            y: Corresponding training targets, in either scalar sample shape.
            grid_size: Number of angular-frequency candidates in the configured
                interval. Must be an integer at least ``max(32, 2*K + 1)``,
                where ``K`` is the model's component count.

        Returns:
            A JSON-compatible report containing sample count, search settings,
            active component count, least-squares rank, training MSE, and the
            initial spectrum. Least-squares MSE is computed in float64 before
            copying coefficients to the model's dtype.

        Raises:
            ValueError: Inputs are nonfinite, shapes differ, the input span is
                zero, fewer than ``2*K + 1`` samples are provided, or the grid
                size is invalid.
            FloatingPointError: Spectral power or fitted coefficients are nonfinite.
            RuntimeError: The frequency grid has no remaining eligible candidates.

        Notes:
            This is part of fitting the model and must receive training data
            only. It updates existing parameters in place without replacing
            them or consuming PyTorch random numbers. Constant targets use
            only the bias; unused components retain random frequencies and
            receive zero initial amplitude. Finite search resolution, noise,
            and the search interval limit which frequencies can be found.
        """
        from scipy.optimize import minimize_scalar
        from scipy.signal import lombscargle

        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        for values in (x, y):
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
                raise ValueError(
                    "Initialization requires scalar samples of shape (N,) or (N, 1)."
                )
        x, y = x.ravel(), y.ravel()
        count = self.frequencies.numel()
        if x.shape != y.shape or x.size < 2 * count + 1:
            raise ValueError(
                "Initialization needs matching x/y and at least 2 * num_frequencies + 1 samples."
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) <= 0:
            raise ValueError(
                "Initialization samples must be finite with a nonzero input span."
            )
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or grid_size < max(32, 2 * count + 1)
        ):
            raise ValueError(
                "grid_size must be an integer >= max(32, 2 * num_frequencies + 1)."
            )

        grid = np.linspace(self.frequency_min, self.frequency_max, grid_size)
        separation = min(
            2 * np.pi / np.ptp(x),
            (self.frequency_max - self.frequency_min) / (2 * count),
        )
        frequencies = []
        residual = y - y.mean()
        coefficients = np.array([y.mean()])
        rank = 1
        available = np.ones(grid_size, dtype=bool)
        initial_energy = float(residual @ residual)

        def design(values):
            angles = x[:, None] * np.asarray(values)
            return np.column_stack((np.ones(len(x)), np.sin(angles), np.cos(angles)))

        for _ in range(count):
            if np.ptp(y) == 0 or float(residual @ residual) <= 1e-12 * max(
                initial_energy, np.finfo(float).tiny
            ):
                break
            power = lombscargle(x, residual, grid, floating_mean=True)
            if not np.isfinite(power).all():
                raise FloatingPointError(
                    "Nonfinite spectral power; inspect input span and frequency bounds."
                )
            if not available.any():
                raise RuntimeError("Frequency grid exhausted; increase grid_size.")
            index = int(np.argmax(np.where(available, power, -np.inf)))

            def residual_error(frequency, residual=residual):
                matrix = design([frequency])
                weights = np.linalg.lstsq(matrix, residual, rcond=None)[0]
                error = residual - matrix @ weights
                return float(error @ error)

            optimum = minimize_scalar(
                residual_error,
                bounds=(grid[max(index - 1, 0)], grid[min(index + 1, grid_size - 1)]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            frequency = min((float(grid[index]), float(optimum.x)), key=residual_error)
            frequencies.append(frequency)
            available &= np.abs(grid - frequency) >= separation
            matrix = design(frequencies)
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
            residual = y - matrix @ coefficients

        active = len(frequencies)
        weights = self.bias.new_tensor(coefficients)
        if not torch.isfinite(weights).all():
            raise FloatingPointError(
                "Least-squares coefficients are not finite in the model dtype."
            )
        with torch.no_grad():
            # Unused components retain random frequencies and zero amplitude.
            self.sine_weights.zero_()
            self.cosine_weights.zero_()
            self.bias.copy_(weights[:1])
            if active:
                self.frequencies[:active].copy_(
                    self.frequencies.new_tensor(frequencies)
                )
                self.sine_weights[:active].copy_(weights[1 : 1 + active])
                self.cosine_weights[:active].copy_(weights[1 + active :])
        return {
            "method": "spectral",
            "training_samples": int(x.size),
            "grid_size": grid_size,
            "frequency_range": [self.frequency_min, self.frequency_max],
            "active_components": active,
            "least_squares_rank": int(rank),
            "least_squares_train_mse": float(np.mean(residual**2)),
            "spectrum": self.spectrum(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate learned sine/cosine components without fitting anything.

        Args:
            x: Finite tensor of shape ``(N, 1)`` in model-input coordinates.

        Returns:
            Tensor of shape ``(N, 1)`` on the model's device. Gradients flow
            through inputs, frequencies, sine/cosine weights, and bias.

        Raises:
            ValueError: Inputs are empty, nonfinite, or have the wrong shape.
        """
        if x.ndim != 2 or x.shape[1] != 1 or len(x) == 0:
            raise ValueError(
                "Fourier regression expects nonempty inputs of shape (N, 1)."
            )
        if not torch.isfinite(x).all():
            raise ValueError("Fourier inputs must be finite.")
        angles = x * self.frequencies
        return (
            torch.sin(angles) * self.sine_weights
            + torch.cos(angles) * self.cosine_weights
        ).sum(dim=1, keepdim=True) + self.bias

    def spectrum(self) -> dict:
        """Return a detached, JSON-compatible description of learned components.

        Returns:
            A bias and a list of components sorted by signed angular frequency.
            Each component contains ``frequency``, ``sine_weight``,
            ``cosine_weight``, and ``amplitude = hypot(sine_weight, cosine_weight)``.
            Frequencies are radians per model-input unit, not necessarily per
            original x unit. Sorting does not modify model parameters.
        """
        components = (
            torch.stack(
                (
                    self.frequencies,
                    self.sine_weights,
                    self.cosine_weights,
                    torch.hypot(self.sine_weights, self.cosine_weights),
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        fields = ("frequency", "sine_weight", "cosine_weight", "amplitude")
        return {
            "bias": float(self.bias.detach().cpu().item()),
            "components": [
                dict(zip(fields, values, strict=True))
                for values in sorted(components, key=lambda row: row[0])
            ],
        }
