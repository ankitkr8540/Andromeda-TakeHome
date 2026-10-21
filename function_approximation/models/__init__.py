"""Construct supported experiment models with backward-compatible defaults."""

from torch import nn

from function_approximation.models.damped_fourier import DampedFourierRegressor
from function_approximation.models.fourier import FourierRegressor
from function_approximation.models.mlp import MLPRegressor


def build_model(model_type: str, architecture: dict) -> nn.Module:
    """Construct an MLP, Fourier, or damped-Fourier model.

    Args:
        model_type: ``"mlp"``, ``"fourier"``, or ``"damped_fourier"``.
        architecture: Constructor keyword arguments, excluding the YAML ``type`` key.

    Returns:
        A new model with randomly initialized parameters. Saved inference loads
        its state dictionary afterward; spectral initialization is a separate
        training-only operation.

    Raises:
        ValueError: The model family or its settings are unsupported.
    """
    if model_type == "mlp":
        return MLPRegressor(**architecture)
    if model_type == "fourier":
        return FourierRegressor(**architecture)
    if model_type == "damped_fourier":
        return DampedFourierRegressor(**architecture)
    raise ValueError(f"Unsupported model: {model_type}")
