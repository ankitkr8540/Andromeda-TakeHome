"""Learnable Fourier regression for scalar observations."""

import math

import numpy as np
import torch
from torch import nn


class FourierRegressor(nn.Module):
    """Learn bias + sum(a_k sin(w_k x) + b_k cos(w_k x)).

    Frequencies, amplitudes, and bias are trainable. Frequency bounds control
    random initialization and spectral search, not subsequent optimization.
    Frequencies use radians per model-input unit; no target frequencies are given.
    Random initialization can be uniform or log-uniform within those bounds.
    """

    model_type = "fourier"

    def __init__(
        self,
        num_frequencies: int = 8,
        frequency_min: float = 0.1,
        frequency_max: float = 5.0,
        frequency_initialization: str = "uniform",
    ) -> None:
        """Allocate trainable frequencies, sine/cosine coefficients, and a bias.

        Args:
            num_frequencies: Positive component count K; total parameters are 3K+1.
            frequency_min: Finite positive lower initialization/search frequency.
            frequency_max: Finite upper frequency, greater than ``frequency_min``.
            frequency_initialization: ``uniform`` or ``log_uniform`` random draws.

        Raises:
            ValueError: Count, bounds, or initialization distribution is invalid.

        Notes:
            Frequencies use radians per model-input unit. Bounds apply to random
            initialization and spectral search, not subsequent optimizer updates."""
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

        if frequency_initialization not in ("uniform", "log_uniform"):
            raise ValueError("frequency_initialization must be uniform or log_uniform.")
        self.frequencies = nn.Parameter(
            torch.empty(num_frequencies).uniform_(frequency_min, frequency_max)
        )
        if frequency_initialization == "log_uniform":
            with torch.no_grad():
                # Transform the same random draws, preserving coefficient RNG state.
                self.frequencies.sub_(frequency_min).div_(frequency_max - frequency_min)
                self.frequencies.mul_(math.log(frequency_max / frequency_min)).add_(
                    math.log(frequency_min)
                ).exp_()
        self.sine_weights = nn.Parameter(0.1 * torch.randn(num_frequencies))
        self.cosine_weights = nn.Parameter(0.1 * torch.randn(num_frequencies))
        self.bias = nn.Parameter(torch.zeros(1))
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max

    def initialize_from_data(
        self, x: np.ndarray, y: np.ndarray, *, grid_size: int = 4096
    ) -> dict:
        """Initialize frequencies and coefficients using training samples only.

        Search Lomb–Scargle peaks in successive residuals, refine each frequency,
        and refit coefficients by least squares. Inputs must use model coordinates
        and have shape (N,) or (N, 1), with N >= 2*K+1. grid_size must be at least
        max(32, 2*K+1). Constant targets use bias only.

        Parameters are updated in place, remain trainable, and consume no PyTorch
        random numbers. Return a JSON-compatible initialization report. Finite
        search resolution, bounds, and noise limit frequency recovery.
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
        target_mean = y.mean()
        residual = y - target_mean
        coefficients = np.array([target_mean])
        rank = 1
        available = np.ones(grid_size, dtype=bool)
        initial_energy = float(residual @ residual)

        def design(values):
            """Build bias, sine, and cosine columns at candidate frequencies on training inputs."""
            angles = x[:, None] * np.asarray(values)
            return np.column_stack((np.ones(len(x)), np.sin(angles), np.cos(angles)))

        for _ in range(count):
            if float(residual @ residual) <= 1e-12 * max(
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
                """Return least-squares residual energy for one candidate angular frequency."""
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
        """Evaluate learned components on finite (N, 1) inputs; return (N, 1).

        Prediction never estimates frequencies or consumes labels.
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
        """Return bias and detached components sorted by signed frequency.

        Each component reports frequency, sine/cosine weights, and amplitude.
        Frequencies are in model-input units; sorting does not change parameters.
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
