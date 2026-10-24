"""Validate pipeline settings before training or output allocation."""

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_experiment_config(ROOT / "configs/experiments/mlp_rich_periodic.yaml")


@pytest.mark.parametrize(
    "section, name, value",
    [
        ("training", "learning_rate", 0),
        ("training", "learning_rate", float("nan")),
        ("training", "min_delta", float("inf")),
        ("training", "min_delta", -1),
        ("training", "batch_size", True),
        ("training", "batch_size", 1.5),
        ("training", "max_epochs", 0),
        ("training", "patience", "50"),
        ("training", "device", "invalid"),
        ("training", "learning_rat", 0.001),
        ("training", "optimizer", "sgd"),
        ("data", "sample_size", 1),
        ("data", "sample_size", 19),
        ("data", "sample_size", True),
        ("data", "sample_size", 3.5),
        ("data", "x_min", float("nan")),
        ("data", "x_max", -100),
        ("data", "sampling", "invalid"),
        ("model", "input_dim", 2),
        ("model", "output_dim", 2),
        ("model", "hidden_dims", [False]),
        ("model", "hidden_dims", [1.5]),
        ("model", "activation", "unknown"),
        ("experiment", "seed", -1),
        ("experiment", "seed", 2**32),
        ("experiment", "name", ""),
        ("tracking", "enabled", "false"),
        ("tracking", "log_figures", 0),
        ("evaluation", "save_predictions", "false"),
        ("evaluation", "metrics", ["unknown"]),
    ],
)
def test_invalid_config_has_field_error_and_creates_no_outputs(
    tmp_path, config, section, name, value
):
    output_root = tmp_path / "outputs"
    config["experiment"]["output_dir"] = str(output_root)
    config[section][name] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        run_experiment(path)
    assert not output_root.exists()


@pytest.mark.parametrize(
    "section, name",
    [
        ("experiment", "run_name"),
        ("data", "sample_size"),
        ("training", "learning_rate"),
        ("training", "max_epochs"),
        ("training", "device"),
    ],
)
def test_missing_fields_rejected(tmp_path, config, section, name):
    del config[section][name]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=name):
        load_experiment_config(path)


SUPPLIED_CONFIGS = sorted((ROOT / "configs").glob("experiments/*.yaml"))


def test_configs_follow_model_split_target_layout():
    assert len(SUPPLIED_CONFIGS) >= 3
    assert (ROOT / "configs/base.yaml").is_file()


@pytest.mark.parametrize(
    "path", SUPPLIED_CONFIGS, ids=lambda p: str(p.relative_to(ROOT / "configs"))
)
def test_supplied_experiments_match_their_location(path):
    """configs/experiments/<model>_<target>.yaml must describe its run."""
    config = load_experiment_config(path)
    model, target = path.stem.split("_", 1)
    assert config["model"]["type"] == model
    assert "split_strategy" not in config["data"]
    assert config["data"]["target"] == target
    assert config["experiment"]["output_dir"] == f"outputs/{model}/{target}"


def test_cli_runs_every_config_in_order(monkeypatch):
    import scripts.run_experiments as runner

    calls = []
    monkeypatch.setattr(runner, "run_experiment", calls.append)
    monkeypatch.setattr("sys.argv", ["run_experiments", "--config", "a.yaml", "b.yaml"])
    runner.main()
    assert calls == ["a.yaml", "b.yaml"]


def test_unavailable_cuda_rejected(tmp_path, config, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config["training"]["device"] = "cuda:0"
    path = tmp_path / "cuda.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="unavailable"):
        load_experiment_config(path)


def test_optional_architecture_defaults_remain_supported(tmp_path, config):
    config["model"] = {"type": "mlp"}
    config["tracking"] = {"enabled": False}
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_experiment_config(path) == deepcopy(config)


def test_yaml_scientific_notation_becomes_numeric(tmp_path, config):
    config["training"].update(learning_rate="1e-3", min_delta="1e-6")
    path = tmp_path / "scientific.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_experiment_config(path)
    assert loaded["training"]["learning_rate"] == 0.001
    assert loaded["training"]["min_delta"] == 0.000001


@pytest.mark.parametrize(
    "bands",
    [
        {"name": "near", "x_min": 40.0, "x_max": 50.0, "n": 10},
        [{"name": "near", "x_min": 50.0, "x_max": 40.0, "n": 10}],
        [{"name": "near", "x_min": 40.0, "x_max": 50.0, "n": 1}],
        [{"name": "near", "x_min": float("inf"), "x_max": 50.0, "n": 10}],
        [{"name": "Near-1", "x_min": 40.0, "x_max": 50.0, "n": 10}],
        [{"name": "near", "x_min": 40.0, "x_max": 50.0}],
        [{"name": "near", "x_min": 40.0, "x_max": 50.0, "n": 10}] * 2,
    ],
)
def test_invalid_extrapolation_bands_rejected(tmp_path, config, bands):
    config["data"]["extrapolation_bands"] = bands
    path = tmp_path / "bands.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="extrapolation_bands"):
        load_experiment_config(path)


def test_extrapolation_bands_are_evaluated_but_never_fitted(tmp_path, config):
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["training"].update(max_epochs=2, patience=1)
    config["tracking"]["enabled"] = False
    config["evaluation"].update(generate_plots=True, save_predictions=True)
    config["data"]["extrapolation_bands"] = [
        {"name": "near", "x_min": 40.0, "x_max": 50.0, "n": 20},
        {"name": "far", "x_min": -700.0, "x_max": -690.0, "n": 20},
    ]
    path = tmp_path / "bands.yaml"
    path.write_text(yaml.safe_dump(config))

    output = run_experiment(path)

    metrics = json.loads((Path(output) / "metrics" / "metrics.json").read_text())
    assert set(metrics) == {"train", "val", "test", "extrap_near", "extrap_far"}
    predictions = np.load(Path(output) / "data" / "predictions.npz")
    assert predictions["extrap_far_x"].min() >= -700.0
    # The training scaler is fitted on train inputs only, never on bands.
    preprocessor = joblib.load(Path(output) / "model" / "preprocessor.joblib")
    assert preprocessor.mean == pytest.approx(predictions["train_x"].mean())
    assert (Path(output) / "plots" / "extrapolation.png").is_file()


@pytest.mark.parametrize("target", ["rich_periodic", "unknown"])
def test_target_names_validated(tmp_path, config, target):
    config["data"]["target"] = target
    path = tmp_path / "target.yaml"
    path.write_text(yaml.safe_dump(config))
    if target == "unknown":
        with pytest.raises(ValueError, match="data.target"):
            load_experiment_config(path)
    else:
        assert load_experiment_config(path)["data"]["target"] == target
