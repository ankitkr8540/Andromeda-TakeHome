"""Standalone Snake model supports raw coordinates, configuration, and reload."""

import numpy as np
import torch
import yaml

from function_approximation.inference import load_predictor
from function_approximation.models import SnakeMLP, build_model
from function_approximation.models.snake_mlp import Snake
from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import PROJECT_ROOT, run_experiment


def test_standalone_architecture_scaling_and_alpha_initialization():
    model = SnakeMLP(dims=[4, 8], a0=10.0, x_scale=2.0)
    x = torch.linspace(-10, 10, 20).reshape(-1, 1)
    torch.testing.assert_close(model(x), model.net(x / 2.0))
    assert model(x).shape == (20, 1)
    for layer in model.net:
        if isinstance(layer, Snake):
            torch.testing.assert_close(layer.alpha, torch.full_like(layer.alpha, 10.0))
    model(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert isinstance(
        build_model("snake", {"dims": [4, 8], "a0": 10.0, "x_scale": 2.0}), SnakeMLP
    )


def test_standalone_raw_input_training_and_reload(tmp_path):
    config = load_experiment_config(
        PROJECT_ROOT / "configs/experiments/snake_rich_periodic.yaml"
    )
    assert config["preprocessing"]["scaling"] == "none"
    assert config["training"]["snake_learning_rate"] == 0.001
    config["model"].update(dims=[4, 8], a0=5.0)
    config["data"]["sample_size"] = 40
    config["training"]["max_epochs"] = 2
    config["evaluation"]["generate_plots"] = False
    config["experiment"]["output_dir"] = str(tmp_path / "runs")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    run = run_experiment(path)
    predictor = load_predictor(run)
    assert isinstance(predictor.model, SnakeMLP)
    samples = np.load(run / "predictions.npz")
    np.testing.assert_allclose(
        predictor.predict(samples["test_x"]), samples["test_pred"]
    )
    assert predictor.preprocessor.std is None


def test_replot_saved_run_without_training_or_changing_artifacts(tmp_path, monkeypatch):
    import hashlib

    from function_approximation.training.trainer import Trainer
    from scripts.replot_experiment import replot_run

    config = load_experiment_config(
        PROJECT_ROOT / "configs/experiments/snake_rich_periodic.yaml"
    )
    config["model"]["dims"] = [4]
    config["data"]["sample_size"] = 40
    config["training"]["max_epochs"] = 1
    config["evaluation"]["generate_plots"] = False
    config["experiment"]["output_dir"] = str(tmp_path / "runs")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    run = run_experiment(path)
    originals = {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for name in ["best_model.pt", "metrics.json", "history.json"]
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Replot must not train")

    monkeypatch.setattr(Trainer, "fit", forbidden)
    output = replot_run(run)
    for name in [
        "predictions.png",
        "residuals.png",
        "extrapolation.png",
        "learning_curves.png",
    ]:
        assert (output / name).is_file()
    for name, digest in originals.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
