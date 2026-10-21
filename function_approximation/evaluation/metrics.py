"""Regression evaluation metrics for function approximation."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate MSE, RMSE, MAE, and R-squared on supplied observations.

    Args:
        y_true: Finite observed targets, flattened before comparison.
        y_pred: Finite predictions with the same flattened sample count.

    Returns:
        Dictionary of Python floats under ``mse``, ``rmse``, ``mae``, and ``r2``.
        Callers supply targets in original units; this function does not undo scaling.

    Raises:
        ValueError: Shapes differ after flattening, fewer than two samples are
            supplied, or either array contains nonfinite values.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("Target and prediction shapes must match.")

    if y_true.size < 2:
        raise ValueError("At least two samples are required.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite.")

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_baseline_metrics(dataset) -> dict:
    """Evaluate a training-target-mean reference on all provided splits.

    Args:
        dataset: Split mapping of ``(x, y)`` arrays with a ``train`` entry.

    Returns:
        ``prediction`` contains the scalar mean of training targets; ``metrics``
        contains each split's scores for that fixed constant prediction.

    Notes:
        Validation and test labels only evaluate the fixed reference. This
        diagnostic is distinct from the trainable MLP baseline model.
    """
    prediction = float(np.mean(dataset["train"][1]))
    return {
        "prediction": prediction,
        "metrics": {
            split: compute_metrics(y, np.full(np.asarray(y).shape, prediction))
            for split, (_, y) in dataset.items()
        },
    }
