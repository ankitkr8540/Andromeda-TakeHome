"""Fourier ablation configuration, zero-step fitting, and reload behavior."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from function_approximation.inference import load_predictor
from function_approximation.training.config import load_experiment_config
from function_approximation.training.trainer import Trainer
from scripts.run_ablation import PROJECT_ROOT, prepare_jobs
from scripts.run_experiments import run_experiment

DEFAULT_STUDY = (
    PROJECT_ROOT / "configs/ablations/fourier/initialization_and_capacity.yaml"
)


def test_five_agreed_fourier_variants(tmp_path):
    study, jobs = prepare_jobs(DEFAULT_STUDY, tmp_path)
    assert len(jobs) == 5
    assert study["targets"] == ["rich_periodic"]
    assert study["seed"] == 42
    configs = [load_experiment_config(j["config_path"]) for j in jobs]
    assert [c["model"]["num_frequencies"] for c in configs] == [8, 8, 8, 4, 16]
    assert all(c["training"]["max_epochs"] == 30000 for c in configs)
    assert all(c["preprocessing"]["scaling"] == "none" for c in configs)


def test_adam_can_retain_epoch_zero(tmp_path):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.zero_()
    x = torch.ones(4, 1)
    train = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=4)
    val = DataLoader(TensorDataset(x, x), batch_size=4)
    trainer = Trainer(model, max_epochs=2, learning_rate=0.1)
    trainer.fit(train, val, include_initial=True)
    assert trainer.best_epoch == 0
    assert trainer.best_val_loss == 0
    assert torch.equal(model(x), x)
    trainer.save_checkpoint(tmp_path / "best.pt", {})


def test_spectral_only_skips_adam_and_reloads(tmp_path, monkeypatch):
    _, jobs = prepare_jobs(DEFAULT_STUDY, tmp_path / "study")
    job = next(j for j in jobs if j["variant"] == "spectral_only_k8")
    config = load_experiment_config(job["config_path"])
    config["data"]["sample_size"] = 100
    config["evaluation"]["generate_plots"] = True
    path = tmp_path / "spectral.yaml"
    path.write_text(yaml.safe_dump(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("Spectral-only must not run Adam")

    monkeypatch.setattr(Trainer, "fit", forbidden)
    run = run_experiment(path)
    checkpoint = torch.load((Path(run) / "model" / "best_model.pt"), weights_only=True)
    assert checkpoint["best_epoch"] == 0
    assert checkpoint["history"] == {"train_loss": [], "val_loss": []}
    assert not (Path(run) / "plots" / "learning_curves.png").exists()
    assert (Path(run) / "plots" / "extrapolation.png").exists()
    saved = np.load(Path(run) / "data" / "predictions.npz")
    predictor = load_predictor(run)
    np.testing.assert_allclose(predictor.predict(saved["test_x"]), saved["test_pred"])
    initialization = json.loads(
        (Path(run) / "diagnostics" / "initialization.json").read_text()
    )
    assert checkpoint["model_state_dict"]["frequencies"].numel() == 8
    assert initialization["model_train_mse"] >= 0
    assert (
        json.loads((Path(run) / "diagnostics" / "timing.json").read_text())[
            "fit_seconds"
        ]
        > 0
    )


def test_log_uniform_initialization_reproducible_and_reloadable():
    from function_approximation.models import build_model

    settings = {
        "num_frequencies": 32,
        "frequency_min": 0.1,
        "frequency_max": 20.0,
        "frequency_initialization": "log_uniform",
    }
    torch.manual_seed(42)
    first = build_model("fourier", settings)
    torch.manual_seed(42)
    second = build_model("fourier", settings)
    assert torch.equal(first.frequencies, second.frequencies)
    torch.manual_seed(42)
    uniform = build_model(
        "fourier", {**settings, "frequency_initialization": "uniform"}
    )
    assert torch.equal(first.sine_weights, uniform.sine_weights)
    assert torch.equal(first.cosine_weights, uniform.cosine_weights)
    assert torch.all((first.frequencies >= 0.1) & (first.frequencies <= 20.0))
    second.load_state_dict(first.state_dict())
    x = torch.linspace(-30, 30, 101).reshape(-1, 1)
    assert torch.equal(first(x), second(x))


def test_random_pilot_configs_validate(tmp_path):
    _study, jobs = prepare_jobs(DEFAULT_STUDY.with_name("random_pilot.yaml"), tmp_path)
    assert len(jobs) == 7
    for job in jobs:
        config = load_experiment_config(job["config_path"])
        assert config["training"]["fourier_initialization"] == "random"


def test_lbfgs_fits_full_batch_regression():
    model = torch.nn.Linear(1, 1)
    x = torch.linspace(-1, 1, 40).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, 2 * x + 0.4), batch_size=40)
    trainer = Trainer(
        model, learning_rate=1.0, max_epochs=5, patience=3, optimizer="lbfgs"
    )
    trainer.fit(loader, loader, include_initial=True)
    assert trainer.evaluate(loader) < 1e-8
