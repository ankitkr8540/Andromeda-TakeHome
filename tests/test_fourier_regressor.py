"""Learnable Fourier values, gradients, training, and saved inference."""

import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import numpy as np
import pytest
import torch
import yaml

from function_approximation.data.dataset import create_dataloaders
from function_approximation.data.generation import DataGenerator
from function_approximation.inference import load_predictor
from function_approximation.models import build_model
from function_approximation.models.fourier_regressor import FourierRegressor
from function_approximation.training.config import load_experiment_config
from function_approximation.training.trainer import Trainer
from scripts.run_experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_fourier_values_and_analytic_gradients():
    model = FourierRegressor(num_frequencies=2).double()
    with torch.no_grad():
        model.frequencies.copy_(torch.tensor([0.7, 1.6]))
        model.sine_weights.copy_(torch.tensor([0.3, -0.4]))
        model.cosine_weights.copy_(torch.tensor([0.2, 0.5]))
        model.bias.fill_(0.1)
    x = torch.tensor([[-1.0], [0.0], [0.4]], dtype=torch.float64, requires_grad=True)
    angles = x.detach() * model.frequencies.detach()
    sine, cosine = model.sine_weights.detach(), model.cosine_weights.detach()
    expected = (angles.sin() * sine + angles.cos() * cosine).sum(1, keepdim=True) + 0.1
    actual = model(x)
    assert actual.shape == (3, 1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(model.sine_weights.grad, angles.sin().sum(0))
    torch.testing.assert_close(model.cosine_weights.grad, angles.cos().sum(0))
    torch.testing.assert_close(
        model.frequencies.grad,
        (x.detach() * (sine * angles.cos() - cosine * angles.sin())).sum(0),
    )
    torch.testing.assert_close(
        model.bias.grad, torch.tensor([3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        x.grad,
        (
            model.frequencies.detach() * (sine * angles.cos() - cosine * angles.sin())
        ).sum(1, keepdim=True),
    )


def test_initialization_is_seeded_and_frequencies_are_independent():
    torch.manual_seed(42)
    first = FourierRegressor()
    torch.manual_seed(42)
    second = FourierRegressor()
    assert sum(parameter.numel() for parameter in first.parameters()) == 25
    assert all(
        torch.equal(a, b)
        for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert torch.all((first.frequencies >= 0.1) & (first.frequencies <= 5.0))
    assert first.frequencies.requires_grad


@pytest.mark.parametrize(
    "settings",
    [
        {"num_frequencies": 0},
        {"num_frequencies": True},
        {"num_frequencies": 2.5},
        {"frequency_min": 0},
        {"frequency_min": True},
        {"frequency_min": float("nan")},
        {"frequency_max": float("inf")},
        {"frequency_min": 5.0, "frequency_max": 5.0},
        {"frequency_min": 6.0, "frequency_max": 5.0},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        FourierRegressor(**settings)


@pytest.mark.parametrize(
    "x",
    [
        torch.empty(0, 1),
        torch.ones(3),
        torch.ones(2, 2),
        torch.tensor([[float("nan")]]),
    ],
)
def test_invalid_inputs(x):
    with pytest.raises(ValueError):
        FourierRegressor()(x)


def test_existing_trainer_learns_frequency_and_preserves_checkpoint_type(tmp_path):
    torch.manual_seed(42)
    x = np.linspace(-3.0, 3.0, 64, dtype=np.float32).reshape(-1, 1)
    y = 0.6 * np.sin(0.73 * x) + 0.2 * np.cos(0.73 * x) - 0.1
    data = {split: (x, y) for split in ("train", "val", "test")}
    loaders = create_dataloaders(data, batch_size=64)
    architecture = {"num_frequencies": 1, "frequency_min": 0.4, "frequency_max": 1.2}
    model = FourierRegressor(**architecture)
    initial_frequencies = model.frequencies.detach().clone()
    trainer = Trainer(model, learning_rate=0.02, max_epochs=300, patience=300)
    initial_loss = trainer.evaluate(loaders["train"])
    trainer.fit(loaders["train"], loaders["val"])
    assert trainer.evaluate(loaders["train"]) < initial_loss * 0.01
    assert not torch.equal(model.frequencies.detach(), initial_frequencies)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for save in (trainer.save_checkpoint,):
        save(tmp_path / "model.pt", architecture)
        checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
        assert checkpoint["model_type"] == "fourier"
        reloaded = build_model(checkpoint["model_type"], checkpoint["model_config"])
        reloaded.load_state_dict(checkpoint["model_state_dict"])


@pytest.mark.parametrize(
    "name",
    [
        "experiments/fourier_rich_periodic.yaml",
    ],
)
@pytest.mark.parametrize("scaling", ["none", "standard"])
def test_full_pipeline_saved_inference_and_spectrum(
    tmp_path, monkeypatch, name, scaling
):
    monkeypatch.chdir(tmp_path)
    config = load_experiment_config(ROOT / "configs" / name)
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["data"].update(sample_size=40)
    config["preprocessing"]["scaling"] = scaling
    config["training"].update(max_epochs=3, patience=3, batch_size=8)
    config["tracking"].update(
        enabled=name == "experiments/fourier_rich_periodic.yaml",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    initialization_calls = []
    original_initializer = FourierRegressor.initialize_from_data

    def record_initialization(model, x, y):
        initialization_calls.append((x.copy(), y.copy()))
        return original_initializer(model, x, y)

    monkeypatch.setattr(FourierRegressor, "initialize_from_data", record_initialization)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    output = run_experiment(path)
    spectrum = json.loads((output / "spectrum.json").read_text())
    predictions = np.load(output / "predictions.npz")
    initialization = json.loads((output / "initialization.json").read_text())
    assert initialization["method"] == "spectral"
    assert initialization["training_samples"] == 28
    assert len(initialization_calls) == 1
    forbidden = Mock(
        side_effect=AssertionError("Inference must not use labels or target function")
    )
    monkeypatch.setattr(DataGenerator, "generate", forbidden)
    monkeypatch.setattr("function_approximation.data.dataset.rich_periodic", forbidden)
    monkeypatch.setattr(
        "function_approximation.data.generation.rich_periodic", forbidden
    )
    monkeypatch.setattr(FourierRegressor, "initialize_from_data", forbidden)
    predictor = load_predictor(output)
    np.testing.assert_array_equal(
        initialization_calls[0][0],
        predictor.preprocessor.transform(predictions["train_x"]),
    )
    np.testing.assert_array_equal(
        initialization_calls[0][1],
        predictions["train_y"].reshape(-1, 1).astype(np.float32),
    )
    np.testing.assert_allclose(
        predictor.predict(predictions["test_x"]), predictions["test_pred"], atol=1e-6
    )
    forbidden.assert_not_called()
    assert not predictor.model.training
    for checkpoint_name in ("best",):
        checkpoint = torch.load(
            output / f"{checkpoint_name}_model.pt", weights_only=True
        )
        assert checkpoint["model_type"] == "fourier"
        model = build_model(checkpoint["model_type"], checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        assert spectrum[checkpoint_name] == {
            "epoch": checkpoint["epoch"],
            **model.spectrum(),
        }
        for component in spectrum[checkpoint_name]["components"]:
            assert component["amplitude"] == pytest.approx(
                np.hypot(component["sine_weight"], component["cosine_weight"]), rel=1e-6
            )
    assert spectrum["input_transform"]["mean"] == (predictor.preprocessor.mean or 0.0)
    assert spectrum["input_transform"]["scale"] == (predictor.preprocessor.std or 1.0)
    if config["tracking"]["enabled"]:
        experiment = mlflow.get_experiment_by_name(config["experiment"]["name"])
        runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
        assert len(runs) == 1 and runs[0].info.status == "FINISHED"
        artifacts = mlflow.MlflowClient().list_artifacts(
            runs[0].info.run_id, "experiment_outputs"
        )
        assert "spectrum.json" in {Path(artifact.path).name for artifact in artifacts}
        assert "initialization.json" in {
            Path(artifact.path).name for artifact in artifacts
        }


@pytest.mark.parametrize(
    "settings",
    [
        {"num_frequencies": False},
        {"num_frequencies": 1.5},
        {"num_frequencies": 0},
        {"frequency_min": 0},
        {"frequency_max": float("inf")},
        {"frequency_min": 5.0, "frequency_max": 1.0},
        {"hidden_dim": 8},
    ],
)
def test_invalid_fourier_config_rejected_before_output(tmp_path, settings):
    config = load_experiment_config(
        ROOT / "configs/experiments/fourier_rich_periodic.yaml"
    )
    config["model"].update(settings)
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        run_experiment(path)
    assert not (tmp_path / "outputs").exists()
