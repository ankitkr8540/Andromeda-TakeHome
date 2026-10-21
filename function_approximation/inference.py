"""Predict from saved artifacts without evaluating the target function."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.models import build_model


def predict_values(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    x: np.ndarray,
) -> np.ndarray:
    """Predict from original x coordinates using a fitted input preprocessor.

    Args:
        model: A fitted scalar regression module with at least one parameter.
        preprocessor: Training-fitted transform associated with these weights.
        x: Nonempty finite inputs of shape ``(N,)`` or ``(N, 1)``; a scalar
            is also accepted as one observation.

    Returns:
        Flat NumPy array of ``N`` predictions in original target units.

    Raises:
        ValueError: Input or prediction shapes are invalid, or transformed
            inputs cannot be represented as finite float32 values.
        RuntimeError: The preprocessor has not been fitted.
        FloatingPointError: Model predictions are nonfinite.

    Notes:
        Gradients are disabled. The caller's model training/evaluation mode is
        restored, even if prediction fails. No labels or target function are used.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
        raise ValueError("Expected scalar inputs with shape (N,) or (N, 1).")
    x_processed = preprocessor.transform(x.reshape(-1, 1))
    if not np.isfinite(x_processed).all():
        raise ValueError("Preprocessed inputs must be finite in float32.")
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_processed).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            predictions = model(x_tensor)
            if predictions.shape != x_tensor.shape:
                raise ValueError("Model must return one scalar prediction per input.")
            if not torch.isfinite(predictions).all():
                raise FloatingPointError("Model produced nonfinite predictions.")
            return predictions.detach().cpu().numpy().ravel()
    finally:
        model.train(was_training)


class Predictor:
    """Pair learned parameters with their training-fitted input transform.

    Args:
        model: Fitted regression module; it is placed in evaluation mode.
        preprocessor: Fitted input transform corresponding to the model.
    """

    def __init__(self, model: nn.Module, preprocessor: DataPreprocessor) -> None:
        self.model = model.eval()
        self.preprocessor = preprocessor

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return a flat prediction array for unscaled scalar inputs.

        Args:
            x: Original-coordinate inputs accepted by :func:`predict_values`.

        Returns:
            One prediction per observation, without fitting or consuming labels.
        """
        return predict_values(self.model, self.preprocessor, x)


def load_predictor(run_dir: str | Path, device: str = "cpu") -> Predictor:
    """Reconstruct best_model.pt and preprocessor.joblib from one run.

    Only load artifacts produced by a trusted experiment: joblib files use
    Python pickle serialization. No labels or target function are needed.

    Args:
        run_dir: Timestamped run directory containing both required artifacts.
        device: PyTorch device for reconstructed parameters; CPU by default.

    Returns:
        A Predictor ready to evaluate original-coordinate inputs.

    Raises:
        FileNotFoundError: A required checkpoint or preprocessor is missing.
        ValueError: Parameters are nonfinite, the model family is unsupported,
            or the loaded preprocessor is invalid or unfitted.
        RuntimeError: Checkpoint tensor shapes do not match the architecture.
    """
    run_dir = Path(run_dir)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model = build_model(
        checkpoint.get("model_type", "mlp"), checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite model parameters.")
    preprocessor = joblib.load(run_dir / "preprocessor.joblib")
    if not isinstance(preprocessor, DataPreprocessor) or not preprocessor.is_fitted:
        raise ValueError("Expected a saved, fitted DataPreprocessor.")
    return Predictor(model, preprocessor)


def main() -> None:
    """Parse inference CLI arguments and print x/predictions as JSON."""
    parser = argparse.ArgumentParser(
        description="Predict using a saved best-validation model."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--x",
        required=True,
        nargs="+",
        type=float,
        help="Original, unscaled input values.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictions = load_predictor(args.run_dir, device=args.device).predict(
        np.array(args.x)
    )
    print(
        json.dumps({"x": args.x, "predictions": predictions.tolist()}, allow_nan=False)
    )


if __name__ == "__main__":
    main()
