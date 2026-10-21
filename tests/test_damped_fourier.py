"""Regression tests for learned damped Fourier components."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from function_approximation.data.generation import DataGenerator
from function_approximation.inference import load_predictor
from function_approximation.models import build_model
from function_approximation.models.damped_fourier import DampedFourierRegressor
from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def test_damped_fourier_matches_its_learned_equation_and_has_gradients():
    model = DampedFourierRegressor(num_frequencies=1).double()
    with torch.no_grad():
        model.frequencies.fill_(0.7)
        model.sine_weights.fill_(0.3)
        model.cosine_weights.fill_(-0.2)
        model.raw_decay_rates.fill_(_inverse_softplus(0.4))
        model.x_reference.fill_(-2.0)
        model.bias.fill_(0.1)

    x = torch.tensor([[-2.0], [-1.0], [0.5]], dtype=torch.float64)
    expected = torch.exp(-0.4 * (x + 2.0)) * (
        0.3 * torch.sin(0.7 * x) - 0.2 * torch.cos(0.7 * x)
    ) + 0.1
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert model.raw_decay_rates.grad is not None
    assert torch.isfinite(model.raw_decay_rates.grad).all()


def test_training_initializer_anchors_to_training_minimum_and_preserves_inference_state():
    x = np.linspace(-8.0, 2.0, 100)
    y = np.exp(-0.15 * (x + 8.0)) * np.sin(0.8 * x) + 0.2
    model = DampedFourierRegressor(
        num_frequencies=1, frequency_min=0.2, frequency_max=1.5
    )
    report = model.initialize_from_data(x, y)

    assert report["method"] == "spectral_damped"
    assert report["x_reference"] == pytest.approx(x.min())
    assert model.x_reference.item() == pytest.approx(x.min())
    assert all(rate >= 0 for rate in [component["decay_rate"] for component in report["spectrum"]["components"]])

    reloaded = DampedFourierRegressor(
        num_frequencies=1, frequency_min=0.2, frequency_max=1.5
    )
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        model(torch.tensor(x[:, None], dtype=torch.float32)),
        reloaded(torch.tensor(x[:, None], dtype=torch.float32)),
    )


def test_damped_config_and_saved_inference(tmp_path):
    config = load_experiment_config(ROOT / "configs/damped_fourier.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["data"].update(n_train=24, n_val=8, n_test=8)
    config["training"].update(max_epochs=2, patience=2, batch_size=8)
    config["tracking"]["enabled"] = False
    config["evaluation"].update(generate_plots=False, save_predictions=True)
    path = tmp_path / "damped.yaml"
    path.write_text(yaml.safe_dump(config))

    output = run_experiment(path)
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert checkpoint["model_type"] == "damped_fourier"
    model = build_model(checkpoint["model_type"], checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    assert isinstance(model, DampedFourierRegressor)
    predictor = load_predictor(output)
    assert predictor.model.x_reference.item() < 0


def test_named_damped_target_is_finite_and_differs_from_periodic_default():
    x = np.linspace(-10.0, 10.0, 20)
    damped = DataGenerator(target="damped_oscillation").target_fn(x)
    periodic = DataGenerator(target="rich_periodic").target_fn(x)
    assert np.isfinite(damped).all()
    assert not np.allclose(damped, periodic)
