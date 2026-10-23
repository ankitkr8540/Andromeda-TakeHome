"""Public model API; configuration names remain mlp, snake, and fourier."""

from torch import nn

from function_approximation.models.fourier_regressor import FourierRegressor
from function_approximation.models.mlp_regressor import MLPRegressor
from function_approximation.models.snake_mlp import Snake, SnakeMLP

__all__ = ["FourierRegressor", "MLPRegressor", "Snake", "SnakeMLP", "build_model"]


def build_model(model_type: str, architecture: dict) -> nn.Module:
    """Construct an unfitted MLP, Snake MLP, or Fourier regressor.

    Args:
        model_type: One of ``mlp``, ``snake``, or ``fourier``.
        architecture: Constructor keyword arguments without the YAML ``type`` key.

    Returns:
        PyTorch module with randomly initialized parameters. Inference callers
        subsequently load saved weights; spectral fitting is a training step.

    Raises:
        ValueError: Model type or architecture settings are invalid, including
            mixing legacy MLP width/depth keys with ``hidden_dims``.
        TypeError: Architecture contains unsupported constructor arguments."""
    if model_type == "mlp":
        architecture = dict(architecture)
        # Reconstruct checkpoints saved before per-layer widths were introduced.
        legacy_keys = {"hidden_dim", "num_hidden_layers"} & architecture.keys()
        if legacy_keys:
            if "hidden_dims" in architecture:
                raise ValueError("Use hidden_dims without legacy width/depth settings.")
            width = architecture.pop("hidden_dim", 64)
            depth = architecture.pop("num_hidden_layers", 2)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (width, depth)
            ):
                raise ValueError(
                    "Legacy hidden_dim and num_hidden_layers must be positive integers."
                )
            architecture["hidden_dims"] = [width] * depth
        return MLPRegressor(**architecture)
    if model_type == "snake":
        return SnakeMLP(**architecture)
    if model_type == "fourier":
        return FourierRegressor(**architecture)
    raise ValueError(f"Unsupported model: {model_type}")
