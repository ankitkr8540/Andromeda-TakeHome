"""Canonical model exports and saved-checkpoint reconstruction."""

from importlib import import_module

import joblib
import numpy as np
import pytest
import torch

from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.inference import load_predictor
from function_approximation.models import (
    FourierRegressor,
    MLPRegressor,
    Snake,
    SnakeMLP,
    build_model,
)


@pytest.mark.parametrize(
    "module_name,name,canonical",
    [
        ("mlp_regressor", "MLPRegressor", MLPRegressor),
        ("snake_mlp", "Snake", Snake),
        ("fourier_regressor", "FourierRegressor", FourierRegressor),
        ("snake_mlp", "SnakeMLP", SnakeMLP),
        ("", "build_model", build_model),
    ],
)
def test_public_exports_reference_canonical_modules(module_name, name, canonical):
    module = import_module(
        "function_approximation.models" + (f".{module_name}" if module_name else "")
    )
    assert getattr(module, name) is canonical
    assert canonical.__module__ == module.__name__


def test_factory_preserves_legacy_architecture_and_random_initialization():
    legacy = {"hidden_dim": 4, "num_hidden_layers": 2}
    torch.manual_seed(42)
    old_architecture = build_model("mlp", legacy)
    torch.manual_seed(42)
    new_architecture = build_model("mlp", {"hidden_dims": [4, 4]})
    assert legacy == {"hidden_dim": 4, "num_hidden_layers": 2}
    assert old_architecture.state_dict().keys() == new_architecture.state_dict().keys()
    for name, parameter in old_architecture.state_dict().items():
        torch.testing.assert_close(
            parameter, new_architecture.state_dict()[name], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "kind,architecture",
    [
        ("mlp", {"hidden_dims": [4, 8]}),
        ("mlp", {"hidden_dim": 4, "num_hidden_layers": 2}),
        ("snake", {"dims": [4, 8], "a0": 1.0, "x_scale": 1.0}),
        ("fourier", {"num_frequencies": 4}),
    ],
)
def test_model_modules_roundtrip_through_saved_state_dict(tmp_path, kind, architecture):
    torch.manual_seed(42)
    model = build_model(kind, architecture)
    x = np.array([-30.0, 0.0, 40.0, 180.0], dtype=np.float32)
    preprocessor = DataPreprocessor("none").fit(x)
    with torch.no_grad():
        expected = model(torch.from_numpy(x.reshape(-1, 1))).numpy().ravel()
    torch.save(
        {
            "model_type": kind,
            "model_config": architecture,
            "model_state_dict": model.state_dict(),
        },
        tmp_path / "best_model.pt",
    )
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    np.testing.assert_array_equal(load_predictor(tmp_path).predict(x), expected)
