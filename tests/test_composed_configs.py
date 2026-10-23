"""Composition and sampling contracts for the periodic study."""

import numpy as np
import pytest

from function_approximation.data.generation import DataGenerator
from function_approximation.training.config import resolve_config


def test_nested_relative_includes_and_override(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts/base.yaml").write_text(
        "training: {patience: 10, learning_rate: 0.001}\ndata: {sample_size: 10000}\n"
    )
    path = tmp_path / "experiment.yaml"
    path.write_text("includes: [parts/base.yaml]\ntraining: {learning_rate: 0.0003}\n")
    assert resolve_config(path) == {
        "training": {"patience": 10, "learning_rate": 0.0003},
        "data": {"sample_size": 10000},
    }
    assert "includes" not in resolve_config(path)


def test_include_cycle_missing_file_and_invalid_include(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("includes: [config.yaml]")
    with pytest.raises(ValueError, match="Circular"):
        resolve_config(path)
    path.write_text("includes: [missing.yaml]")
    with pytest.raises(FileNotFoundError):
        resolve_config(path)
    path.write_text("includes: base.yaml")
    with pytest.raises(ValueError, match="includes"):
        resolve_config(path)


def test_interpolation_coverage_and_ratio():
    generator = DataGenerator(sample_size=100)
    first = generator.generate()
    assert tuple(len(first[n][0]) for n in ("train", "val", "test")) == (70, 10, 20)
    assert first["train"][0][0] == -10 * np.pi
    assert first["train"][0][-1] == 10 * np.pi
    for name in ("val", "test"):
        assert first[name][0].min() > first["train"][0].min()
        assert first[name][0].max() < first["train"][0].max()
    assert not np.intersect1d(first["train"][0], first["test"][0]).size


def test_grid_bands_exclude_training_boundary():
    band = {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 100}
    generator = DataGenerator(extrapolation_bands=[band])
    x, _ = generator.generate_extrapolation()["extrap_near"]
    assert x.min() > 10 * np.pi
    assert x.max() == 20 * np.pi


@pytest.mark.parametrize("model", ["mlp", "snake", "fourier"])
def test_experiment_configs_use_current_data_api(model):
    from pathlib import Path

    from function_approximation.data.dataset import create_dataloaders
    from function_approximation.data.preprocessing import DataPreprocessor
    from function_approximation.training.config import load_experiment_config

    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(
        root / f"configs/experiments/{model}_rich_periodic.yaml"
    )
    generator = DataGenerator(**config["data"])
    processed = DataPreprocessor(**config["preprocessing"]).fit_transform_dataset(
        generator.generate()
    )
    loaders = create_dataloaders(
        processed,
        batch_size=config["training"]["batch_size"],
        seed=config["experiment"]["seed"],
    )
    assert tuple(len(loaders[name].dataset) for name in ("train", "val", "test")) == (
        7000,
        1000,
        2000,
    )
    assert set(generator.generate_extrapolation()) == {"extrap_near", "extrap_far"}
