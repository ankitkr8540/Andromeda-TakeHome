"""Saved MLP, Snake, and Fourier runs share evaluation grids without training."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
import yaml

from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.models import build_model
from scripts.evaluate_extrapolation_distance import evaluate_runs, main


@pytest.fixture
def saved_runs(tmp_path):
    from function_approximation.training.config import load_experiment_config

    root = Path(__file__).resolve().parents[1]
    runs = {}
    for kind in ("mlp", "snake", "fourier"):
        config = load_experiment_config(
            root / f"configs/experiments/{kind}_rich_periodic.yaml"
        )
        config["data"]["sample_size"] = 40
        if kind == "mlp":
            config["model"]["hidden_dims"] = [4]
        elif kind == "snake":
            config["model"]["dims"] = [4]
        else:
            config["model"]["num_frequencies"] = 2
        run = tmp_path / kind
        run.mkdir()
        (run / "model").mkdir()
        (run / "metrics").mkdir()
        architecture = {k: v for k, v in config["model"].items() if k != "type"}
        model = build_model(kind, architecture)
        torch.save(
            {
                "model_type": kind,
                "model_config": architecture,
                "model_state_dict": model.state_dict(),
            },
            run / "model" / "best_model.pt",
        )
        preprocessor = DataPreprocessor(**config["preprocessing"]).fit(
            np.array([-30.0, 0.0, 30.0])
        )
        joblib.dump(preprocessor, run / "model" / "preprocessor.joblib")
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        (run / "metrics" / "baseline_metrics.json").write_text(
            json.dumps({"prediction": 0.2})
        )
        runs[kind] = run
    return runs


def test_compare_saved_models_exports_actual_per_band_metrics(
    saved_runs, tmp_path, monkeypatch
):
    from function_approximation.data.dataset import rich_periodic
    from function_approximation.inference import load_predictor
    from function_approximation.training.trainer import Trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Distance evaluation must not fit a model")

    monkeypatch.setattr(Trainer, "fit", forbidden)
    originals = {
        p: p.read_bytes()
        for run in saved_runs.values()
        for p in run.rglob("*")
        if p.is_file()
    }
    out = evaluate_runs(
        saved_runs, tmp_path / "evaluation", max_pi=30, samples_per_band=20
    )
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert len(rows) == 8  # two bands, three models plus a training-mean baseline
    assert {r["variant"] for r in rows} == {*saved_runs, "Training-mean baseline"}
    for row in rows:
        x = np.linspace(float(row["x_min"]), float(row["x_max"]), 21)[1:]
        prediction = (
            np.full(x.shape, 0.2)
            if row["variant"] == "Training-mean baseline"
            else load_predictor(saved_runs[row["variant"]]).predict(x)
        )
        expected = np.sqrt(
            np.mean((prediction.astype(np.float64) - rich_periodic(x)) ** 2)
        )
        assert float(row["rmse"]) == pytest.approx(expected)
        assert float(row["x_min"]) >= 10 * np.pi
    provenance = json.loads((out / "source.json").read_text())
    assert set(provenance["checkpoints"]) == set(saved_runs)
    assert provenance["baseline_prediction"] == 0.2
    for name in ("rmse_vs_distance.png", "rmse_vs_distance.svg"):
        assert (out / name).stat().st_size > 0
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_comparison_rejects_mismatched_training_domains(saved_runs, tmp_path):
    path = saved_runs["snake"] / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["data"]["x_max"] = 20.0
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="share target"):
        evaluate_runs(saved_runs, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_cli_rejects_duplicate_labels_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "distance",
            "--run",
            "MLP",
            "one",
            "--run",
            "MLP",
            "two",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
