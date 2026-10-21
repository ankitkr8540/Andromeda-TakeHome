"""Predict only from saved model/preprocessor artifacts."""

from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import torch

from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.inference import load_predictor, predict_values
from function_approximation.models.mlp import MLPRegressor


@pytest.fixture
def saved_artifacts(tmp_path):
    torch.manual_seed(42)
    architecture = {"hidden_dim": 4, "num_hidden_layers": 1}
    model = MLPRegressor(**architecture)
    preprocessor = DataPreprocessor().fit(np.array([-3.0, -1.0, 0.0, 2.0]))
    torch.save(
        {
            "model_config": architecture,
            "model_state_dict": model.state_dict(),
            "epoch": 1,
        },
        tmp_path / "best_model.pt",
    )
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    return tmp_path, model, preprocessor


def test_saved_predictions_roundtrip_without_target_function(
    saved_artifacts, monkeypatch
):
    path, model, preprocessor = saved_artifacts
    # The held-out values need no labels or synthetic data generation.
    x = np.array([-5.0, 0.25, 4.0])
    expected = predict_values(model, preprocessor, x)
    forbidden = Mock(
        side_effect=AssertionError("Ground truth must never run during inference")
    )
    monkeypatch.setattr(DataGenerator, "_target_function", forbidden)
    monkeypatch.setattr(DataGenerator, "generate", forbidden)
    predictor = load_predictor(path)
    assert not predictor.model.training
    np.testing.assert_array_equal(predictor.predict(x), expected)
    np.testing.assert_array_equal(predictor.predict(x.reshape(-1, 1)), expected)
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "x", [[], [float("nan")], [float("inf")], [[1.0, 2.0]], [[[1.0]]]]
)
def test_invalid_inference_inputs_rejected(saved_artifacts, x):
    path, _, _ = saved_artifacts
    with pytest.raises(ValueError):
        load_predictor(path).predict(np.array(x))


def test_unfitted_preprocessor_rejected(saved_artifacts):
    path, _, _ = saved_artifacts
    joblib.dump(DataPreprocessor(), path / "preprocessor.joblib")
    with pytest.raises(ValueError, match="fitted"):
        load_predictor(path)


def test_nonfinite_checkpoint_rejected(saved_artifacts):
    path, _, _ = saved_artifacts
    checkpoint = torch.load(path / "best_model.pt", weights_only=True)
    next(iter(checkpoint["model_state_dict"].values())).fill_(float("nan"))
    torch.save(checkpoint, path / "best_model.pt")
    with pytest.raises(ValueError, match="nonfinite"):
        load_predictor(path)
