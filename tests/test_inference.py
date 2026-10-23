"""Predict only from saved model/preprocessor artifacts."""

import json
import subprocess
import sys
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import torch

from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.inference import load_predictor, predict_values
from function_approximation.models import build_model


@pytest.fixture
def saved_artifacts(tmp_path):
    torch.manual_seed(42)
    architecture = {"hidden_dim": 4, "num_hidden_layers": 1}
    model = build_model("mlp", architecture)
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
    monkeypatch.setattr("function_approximation.data.dataset.rich_periodic", forbidden)
    monkeypatch.setattr(
        "function_approximation.data.generation.rich_periodic", forbidden
    )
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


@pytest.mark.parametrize(
    "kind,architecture",
    [
        ("mlp", {"hidden_dims": [4, 8]}),
        ("snake", {"dims": [4, 8]}),
        ("fourier", {"num_frequencies": 4}),
    ],
)
@pytest.mark.parametrize("scaling", ["none", "standard"])
def test_output_directory_loads_each_model_with_saved_preprocessing(
    tmp_path, kind, architecture, scaling
):
    torch.manual_seed(42)
    model = build_model(kind, architecture)
    preprocessor = DataPreprocessor(scaling).fit(np.array([-30.0, 0.0, 30.0]))
    torch.save(
        {
            "model_type": kind,
            "model_config": architecture,
            "model_state_dict": model.state_dict(),
        },
        tmp_path / "best_model.pt",
    )
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    x = np.array([-40.0, 0.25, 180.0])
    expected = predict_values(model, preprocessor, x)
    predictor = load_predictor(tmp_path)
    np.testing.assert_array_equal(predictor.predict(x), expected)
    np.testing.assert_array_equal(
        predictor.predict(0.25), predict_values(model, preprocessor, np.array(0.25))
    )


@pytest.mark.parametrize("name", ["best_model.pt", "preprocessor.joblib"])
def test_missing_artifact_explains_required_output_directory(saved_artifacts, name):
    path, _, _ = saved_artifacts
    (path / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        load_predictor(path)


@pytest.mark.parametrize("checkpoint", [[], {}, {"model_config": []}])
def test_malformed_checkpoint_has_clear_error(saved_artifacts, checkpoint):
    path, _, _ = saved_artifacts
    torch.save(checkpoint, path / "best_model.pt")
    with pytest.raises(ValueError, match="checkpoint mapping"):
        load_predictor(path)


def test_cli_loads_output_and_returns_predictions_as_json(saved_artifacts):
    path, model, preprocessor = saved_artifacts
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "function_approximation.inference",
            "--run-dir",
            str(path),
            "--x",
            "-40",
            "0.25",
            "180",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["x"] == [-40.0, 0.25, 180.0]
    np.testing.assert_array_equal(
        payload["predictions"],
        predict_values(model, preprocessor, np.array(payload["x"])),
    )
