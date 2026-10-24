"""Load an output run and predict without training or evaluating the target.

Python usage::

    predictor = load_predictor("outputs/<model>/<target>/<run>")
    predictions = predictor.predict([-10.0, 0.0, 100.0])

Terminal usage::

    python -m function_approximation.inference --run-dir outputs/<run> --x -10 0 100

Grouped runs keep best_model.pt and preprocessor.joblib under model/. Legacy flat runs also work.
"""

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
    """Predict from original-coordinate inputs without labels or target evaluations.

    Args:
        model: Fitted PyTorch model returning one scalar per input.
        preprocessor: Fitted input transform saved with the model.
        x: Finite scalar, vector, or column of original input coordinates.

    Returns:
        Flat NumPy prediction array of shape ``(N,)`` in target units.

    Raises:
        RuntimeError: Preprocessing is unfitted.
        ValueError: Input values/shapes are invalid or model output shape differs.
        FloatingPointError: Model predictions contain nonfinite values.

    Notes:
        Disables gradients, uses the model's device, and restores its original
        training/evaluation mode even if prediction fails."""
    x_processed = preprocessor.transform(x)
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
            return predictions.cpu().numpy().ravel()
    finally:
        model.train(was_training)


class Predictor:
    """Keep a loaded model and its fitted preprocessing together for prediction."""

    def __init__(self, model: nn.Module, preprocessor: DataPreprocessor) -> None:
        """Attach a model and fitted preprocessor; put the model into evaluation mode."""
        self.model = model.eval()
        self.preprocessor = preprocessor

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict one value per original-coordinate input using saved preprocessing.

        Args:
            x: Finite scalar, vector, or column; may lie outside the training window.

        Returns:
            Flat NumPy prediction array. No labels or reference function are required.

        Notes:
            Delegates validation and errors to ``predict_values``. Extrapolation
            is allowed, but its accuracy is not guaranteed."""
        return predict_values(self.model, self.preprocessor, x)


def load_predictor(run_dir: str | Path, device: str = "cpu") -> Predictor:
    """Reconstruct a fitted predictor from trusted experiment artifacts.

    Args:
        run_dir: Run directory containing ``model/best_model.pt`` and
            ``model/preprocessor.joblib``; legacy flat runs also work.
        device: PyTorch device on which to load and evaluate the model.

    Returns:
        Predictor containing the selected weights and fitted input transform.

    Raises:
        FileNotFoundError: A required artifact is missing.
        ValueError: Parameters are nonfinite or preprocessing is invalid/unfitted.
        RuntimeError: Saved weights do not match the reconstructed architecture.

    Notes:
        Loading does not generate labels or train the model. Joblib uses pickle:
        only load artifacts from trusted sources. Other deserialization errors
        propagate from PyTorch/joblib."""
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "model" / "best_model.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "best_model.pt"
    preprocessor_path = run_dir / "model" / "preprocessor.joblib"
    if not preprocessor_path.is_file():
        preprocessor_path = run_dir / "preprocessor.joblib"
    for path in (checkpoint_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing inference artifact: {path}. "
                "Choose the run root containing model/best_model.pt and "
                "model/preprocessor.joblib (or their legacy flat equivalents)."
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not all(
        isinstance(checkpoint.get(key), dict)
        for key in ("model_config", "model_state_dict")
    ):
        raise ValueError(
            "Expected a checkpoint mapping with model_config and model_state_dict mappings."
        )
    model = build_model(
        checkpoint.get("model_type", "mlp"), checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite model parameters.")
    preprocessor = joblib.load(preprocessor_path)
    if not isinstance(preprocessor, DataPreprocessor) or not preprocessor.is_fitted:
        raise ValueError("Expected a saved, fitted DataPreprocessor.")
    return Predictor(model, preprocessor)


def main() -> None:
    """Parse inference CLI arguments and print x/predictions as JSON."""
    parser = argparse.ArgumentParser(
        description="Predict using a saved best-validation model."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Output run root containing model/ artifacts; legacy flat runs also work.",
    )
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
