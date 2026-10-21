"""Validate pipeline settings before training or output allocation."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_experiment_config(ROOT / "configs/mlp_random.yaml")


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
        ("data", "n_train", 1),
        ("data", "n_val", 1),
        ("data", "n_test", 1),
        ("data", "n_train", 3.5),
        ("data", "x_min", float("nan")),
        ("data", "x_max", -100),
        ("data", "sampling", "invalid"),
        ("model", "input_dim", 2),
        ("model", "output_dim", 2),
        ("model", "hidden_dim", False),
        ("model", "num_hidden_layers", 1.5),
        ("model", "activation", "relu"),
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
        ("data", "n_test"),
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


@pytest.mark.parametrize(
    "name", ["mlp_random.yaml", "mlp_chrono.yaml", "mlp_chronological.yaml"]
)
def test_supplied_experiments_validate(name):
    load_experiment_config(ROOT / "configs" / name)


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
