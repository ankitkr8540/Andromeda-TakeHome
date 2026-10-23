"""Regression scores and the training-only mean reference."""

import numpy as np
import pytest

from function_approximation.evaluation.metrics import (
    compute_baseline_metrics,
    compute_metrics,
)


def test_metrics_match_known_errors_and_original_target_units():
    scores = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert scores == pytest.approx(
        {"mse": 1 / 3, "rmse": np.sqrt(1 / 3), "mae": 1 / 3, "r2": 0.5}
    )
    assert all(isinstance(value, float) for value in scores.values())


def test_perfect_predictions_accept_columns_and_vectors():
    y = np.array([1.0, 2.0, 3.0])
    assert compute_metrics(y[:, None], y) == {
        "mse": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 1.0,
    }


@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([], []),
        ([1.0], [1.0]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([np.nan, 2.0], [1.0, 2.0]),
        ([1.0, 2.0], [np.inf, 2.0]),
    ],
)
def test_invalid_observations_are_rejected(truth, prediction):
    with pytest.raises(ValueError):
        compute_metrics(truth, prediction)


def test_mean_baseline_is_fitted_only_on_training_targets():
    data = {
        "train": (np.array([0.0, 1.0]), np.array([1.0, 3.0])),
        "val": (np.array([2.0, 3.0]), np.array([100.0, 200.0])),
        "extrap_far": (np.array([100.0, 101.0]), np.array([-100.0, -200.0])),
    }
    result = compute_baseline_metrics(data)
    assert result["prediction"] == 2.0
    assert set(result["metrics"]) == set(data)
    for split, (_, y) in data.items():
        assert result["metrics"][split] == compute_metrics(y, np.full_like(y, 2.0))
    assert data["val"][1].tolist() == [100.0, 200.0]
