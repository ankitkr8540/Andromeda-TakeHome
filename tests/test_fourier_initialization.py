"""Data-derived initialization, independent frequencies, and failure checks."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from function_approximation.models.fourier_regressor import FourierRegressor
from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_random_initialization_remains_available_without_fitting_data(
    tmp_path, monkeypatch
):
    config = load_experiment_config(
        ROOT / "configs/experiments/fourier_rich_periodic.yaml"
    )
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["training"].update(max_epochs=1, patience=1)
    del config["training"]["fourier_initialization"]
    config["tracking"]["enabled"] = False
    config["evaluation"]["generate_plots"] = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Random mode must not fit spectral initialization")

    monkeypatch.setattr(FourierRegressor, "initialize_from_data", forbidden)
    path = tmp_path / "random.yaml"
    path.write_text(yaml.safe_dump(config))
    output = run_experiment(path)
    import json

    initialization = json.loads((output / "initialization.json").read_text())
    assert initialization["method"] == "random"
    torch.manual_seed(config["experiment"]["seed"])
    reference = FourierRegressor(
        **{key: value for key, value in config["model"].items() if key != "type"}
    )
    assert initialization["spectrum"] == reference.spectrum()


def test_initialization_fits_independent_frequencies_without_changing_parameters():
    rng = np.random.default_rng(11)
    x = rng.uniform(-20, 20, 180)
    y = 0.8 * np.sin(0.73 * x) + 0.25 * np.cos(1.61 * x) + 0.14
    model = FourierRegressor(num_frequencies=2, frequency_min=0.2, frequency_max=2.5)
    parameters = list(model.parameters())
    rng_state = torch.get_rng_state().clone()
    report = model.initialize_from_data(x, y)
    frequencies = sorted(model.frequencies.detach().tolist())
    np.testing.assert_allclose(frequencies, [0.73, 1.61], atol=0.01)
    assert report["least_squares_train_mse"] < 1e-4
    assert report["least_squares_rank"] == 5
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert all(
        a is b and b.requires_grad
        for a, b in zip(parameters, model.parameters(), strict=True)
    )
    loss = model(torch.tensor(x[:, None], dtype=torch.float32)).square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.frequencies.grad.abs().sum() > 0


def test_initialization_does_not_assume_a_sine_phase():
    x = np.linspace(-12, 8, 120)
    y = 0.6 * np.cos(1.37 * x) - 0.4 * np.sin(1.37 * x) + 0.3
    model = FourierRegressor(
        num_frequencies=1, frequency_min=0.2, frequency_max=2.0
    ).double()
    report = model.initialize_from_data(x, y)
    assert report["least_squares_train_mse"] < 1e-12
    assert model.sine_weights.item() == pytest.approx(-0.4, abs=1e-5)
    assert model.cosine_weights.item() == pytest.approx(0.6, abs=1e-5)
    assert model.bias.item() == pytest.approx(0.3, abs=1e-5)


def test_constant_observations_use_bias_without_spurious_spectral_peaks():
    x = np.linspace(-3, 3, 40)
    model = FourierRegressor()
    report = model.initialize_from_data(x, np.full_like(x, 0.123))
    assert report["active_components"] == 0
    with torch.no_grad():
        actual = model(torch.tensor(x[:, None], dtype=torch.float32))
    torch.testing.assert_close(actual, torch.full_like(actual, 0.123))


@pytest.mark.parametrize(
    "x,y,grid_size",
    [
        (np.arange(5), np.arange(4), 4096),
        (np.arange(2), np.arange(2), 4096),
        (np.ones(8), np.arange(8), 4096),
        (np.arange(8), np.full(8, np.nan), 4096),
        (np.ones((8, 2)), np.ones(8), 4096),
        (np.arange(8), np.arange(8), True),
        (np.arange(8), np.arange(8), 4),
    ],
)
def test_bad_initialization_inputs_do_not_change_weights(x, y, grid_size):
    model = FourierRegressor(num_frequencies=1)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        model.initialize_from_data(x, y, grid_size=grid_size)
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name])


@pytest.mark.parametrize(
    "model_type,mode,sample_size",
    [
        ("fourier", "unknown", 40),
        ("fourier", True, 40),
        ("mlp", "spectral", 40),
        ("fourier", "spectral", 20),
    ],
)
def test_invalid_initialization_config_creates_no_outputs(
    tmp_path, model_type, mode, sample_size
):
    source = (
        "experiments/fourier_rich_periodic.yaml"
        if model_type == "fourier"
        else "experiments/mlp_rich_periodic.yaml"
    )
    config = load_experiment_config(ROOT / "configs" / source)
    config["training"]["fourier_initialization"] = mode
    config["data"]["sample_size"] = sample_size
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        run_experiment(path)
    assert not (tmp_path / "outputs").exists()
