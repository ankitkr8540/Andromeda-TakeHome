"""Ablation setup, actual small runs, validation selection, and safe resume."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from function_approximation.training.config import load_experiment_config
from scripts.run_ablation import PROJECT_ROOT, prepare_jobs, run_study, summarize

DEFAULT_STUDY = PROJECT_ROOT / "configs/ablations/mlp/architecture.yaml"


def test_agreed_study_and_safe_configuration_resume(tmp_path):
    study, jobs = prepare_jobs(DEFAULT_STUDY, tmp_path)
    assert len(jobs) == 8
    assert len({j["run_id"] for j in jobs}) == 8
    assert study["seed"] == 42
    for job in jobs:
        config = load_experiment_config(job["config_path"])
        assert config["model"]["hidden_dims"] not in (
            [64, 64],
            [32, 32],
            [128, 128],
            [64],
        )
        assert config["training"]["max_epochs"] == 20000
        assert config["data"]["sample_size"] == 10000
    assert prepare_jobs(DEFAULT_STUDY, tmp_path)[1] == jobs
    with pytest.raises(ValueError, match="Settings changed"):
        prepare_jobs(DEFAULT_STUDY, tmp_path, max_epochs=1)


def test_real_small_study_and_resume(tmp_path, monkeypatch):
    overrides = {
        "data": {"sample_size": 40},
        "evaluation": {"generate_plots": False, "save_predictions": True},
    }
    spec = {
        "seed": 42,
        "max_epochs": 2,
        "reference": "reference",
        "targets": ["rich_periodic"],
        "model": {"hidden_dims": [4], "activation": "tanh"},
        "variants": {
            "reference": overrides,
            "relu": {**overrides, "model": {"activation": "relu"}},
        },
    }
    spec_path = tmp_path / "study.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    output = tmp_path / "output"
    study, jobs = prepare_jobs(spec_path, output)
    run_study(study, jobs, output)
    assert json.loads((output / "progress.json").read_text())["completed"] == 2
    selected = json.loads((output / "selected_models.json").read_text())["models"]
    records = [json.loads(p.read_text()) for p in (output / "results").glob("*.json")]
    for target in study["targets"]:
        candidates = [r for r in records if r["target"] == target]
        winner = min(candidates, key=lambda r: r["metrics"]["val"]["rmse"])
        assert selected[target]["variant"] == winner["variant"]
        assert (
            Path(selected[target]["output_dir"]) / "model" / "best_model.pt"
        ).is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "comparison.png").is_file()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed jobs must not retrain")

    monkeypatch.setattr("scripts.run_ablation.run_experiment", forbidden)
    run_study(study, jobs, output)
    assert json.loads((output / "progress.json").read_text())["completed"] == 2


def test_summary_supports_cached_records_without_timing(tmp_path):
    spec = {
        "seed": 42,
        "max_epochs": 1,
        "reference": "old",
        "targets": ["rich_periodic"],
        "model": {},
        "variants": {"old": {}, "new": {}},
    }
    output = tmp_path
    (output / "results").mkdir()
    metrics = {
        region: {"mse": 0.01, "rmse": 0.1, "mae": 0.08, "r2": 0.9}
        for region in ["val", "test", "extrap_near", "extrap_far"]
    }
    jobs = []
    for variant in ["old", "new"]:
        job = {"run_id": variant, "fingerprint": variant}
        jobs.append(job)
        record = {
            **job,
            "target": "rich_periodic",
            "variant": variant,
            "seed": 42,
            "parameters": 10,
            "run_seconds": 1.0,
            "best_epoch": 1,
            "epochs_completed": 1,
            "metrics": metrics,
            "status": "complete",
            "output_dir": str(tmp_path / variant),
        }
        if variant == "new":
            record["timing"] = {"fit_seconds": 0.5}
        (output / "results" / f"{variant}.json").write_text(json.dumps(record))
    assert summarize(spec, jobs, output) == 2
    rows = list(csv.DictReader((output / "metrics.csv").open()))
    assert rows[0]["fit_seconds"] == ""
    assert rows[-1]["fit_seconds"] == "0.5"


@pytest.mark.parametrize(
    "study_name,study_file,expected",
    [("mlp", "architecture", 8), ("fourier", "initialization_and_capacity", 5)],
)
def test_common_cli_prepares_each_model_study_without_training(
    tmp_path, monkeypatch, study_name, study_file, expected
):
    from scripts import run_ablation

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not train")

    monkeypatch.setattr(run_ablation, "run_study", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ablation",
            "--study",
            str(PROJECT_ROOT / f"configs/ablations/{study_name}/{study_file}.yaml"),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    run_ablation.main()
    assert len(list((tmp_path / "configs").glob("*.yaml"))) == expected
    assert not (tmp_path / "results").exists()
