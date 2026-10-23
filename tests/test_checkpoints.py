from function_approximation.training.config import load_experiment_config

"""Regression proof for checkpoint ownership and experiment diagnostics."""

import json
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytest
import torch
import yaml

from function_approximation.data.dataset import create_dataloaders
from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.evaluation.visualization import (
    plot_learning_curves,
)
from function_approximation.models.mlp_regressor import MLPRegressor
from function_approximation.training.trainer import Trainer
from scripts.run_experiments import evaluate_splits, run_experiment

ARCHITECTURE = {"hidden_dims": [4]}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_learning_curves_display_rmse_without_changing_history(tmp_path):
    history = {"train_loss": [4.0, 1.0, 0.25], "val_loss": [1.0, 0.25, 4.0]}
    original = deepcopy(history)
    fig = plot_learning_curves(history, best_epoch=2, save_path=tmp_path / "curves.png")
    try:
        ax = fig.axes[0]
        assert ax.get_ylabel() == "RMSE"
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2.0, 1.0, 0.5])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [1.0, 0.5, 2.0])
        np.testing.assert_allclose(ax.lines[2].get_xdata(), [2, 2])
        assert history == original
    finally:
        plt.close(fig)


@pytest.mark.parametrize("patience, expected_final", [(2, 3), (10, 4)])
def test_checkpoint_selection_and_roundtrip(
    tmp_path, monkeypatch, patience, expected_final
):
    trainer = Trainer(
        MLPRegressor(**ARCHITECTURE), max_epochs=4, patience=patience, min_delta=0.1
    )
    # Controlled epoch outcomes prove selection/restoration independent of
    # optimizer convergence. Epoch 2's improvement is smaller than min_delta.
    validation_losses = [1.0, 0.95, 1.2, 1.3]
    completed = []

    def run_epoch(loader, training):
        if training:
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(1.0)
            return 2.0 / (len(completed) + 1)
        completed.append(deepcopy(trainer.model.state_dict()))
        return validation_losses[len(completed) - 1]

    monkeypatch.setattr(trainer, "_run_epoch", run_epoch)
    trainer.fit(None, None)
    assert trainer.best_epoch == 1
    assert len(trainer.history["train_loss"]) == expected_final

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, completed[0][key])
        assert value.data_ptr() != completed[0][key].data_ptr()

    trainer.save_checkpoint(tmp_path / "best.pt", ARCHITECTURE)
    active_state = deepcopy(trainer.model.state_dict())
    loaded_models = {}
    for name, epoch, expected in (("best", 1, completed[0]),):
        checkpoint = torch.load(tmp_path / f"{name}.pt", weights_only=True)
        assert checkpoint["epoch"] == epoch
        model = MLPRegressor(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected[key])
        loaded_models[name] = model

    generator = DataGenerator(sample_size=20)
    dataset = generator.generate()
    preprocessor = DataPreprocessor().fit(dataset["train"][0])
    for model in loaded_models.values():
        metrics = evaluate_splits(model, preprocessor, dataset)
        assert all(
            np.isfinite(list(values.values())).all() for values in metrics.values()
        )
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, active_state[key])


def test_saving_before_fit_rejected(tmp_path):
    trainer = Trainer(MLPRegressor(**ARCHITECTURE))
    for save in (trainer.save_checkpoint,):
        with pytest.raises(RuntimeError, match="Train the model"):
            save(tmp_path / "model.pt", ARCHITECTURE)


def test_real_training_restores_best_epoch(monkeypatch):
    torch.manual_seed(42)
    dataset = DataGenerator(sample_size=40).generate()
    preprocessor = DataPreprocessor()
    loaders = create_dataloaders(
        preprocessor.fit_transform_dataset(dataset), batch_size=8
    )
    trainer = Trainer(MLPRegressor(**ARCHITECTURE), max_epochs=3, patience=3)
    run_epoch = trainer._run_epoch
    snapshots = []

    def record_epoch(loader, training):
        loss = run_epoch(loader, training)
        if not training:
            snapshots.append(deepcopy(trainer.model.state_dict()))
        return loss

    monkeypatch.setattr(trainer, "_run_epoch", record_epoch)
    trainer.fit(loaders["train"], loaders["val"])
    assert len(snapshots) == 3
    active_state = deepcopy(trainer.model.state_dict())
    for key, value in active_state.items():
        assert torch.equal(value, snapshots[trainer.best_epoch - 1][key])

    inference_calls = []

    def check_inference(model, inputs):
        inference_calls.append(
            (model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )

    trainer.model.register_forward_pre_hook(check_inference)
    evaluate_splits(trainer.model, preprocessor, dataset)
    assert inference_calls == [(False, False, True)] * 3
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, active_state[key])


@pytest.mark.parametrize(
    "config_name",
    ["experiments/mlp_rich_periodic.yaml"],
)
def test_short_experiment_artifacts_and_metrics(tmp_path, monkeypatch, config_name):
    """Exercise real training, serialization and a single local MLflow run."""
    monkeypatch.chdir(tmp_path)
    config = load_experiment_config(PROJECT_ROOT / "configs" / config_name)
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["data"].update(sample_size=40)
    config["model"].update(ARCHITECTURE)
    config["training"].update(max_epochs=3, patience=2, batch_size=8)
    config["tracking"].update(
        enabled=True, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = run_experiment(config_path)

    expected_artifacts = {
        "best_model.pt",
        "config.yaml",
        "metrics.json",
        "history.json",
        "preprocessor.joblib",
        "learning_curves.png",
        "predictions.png",
        "residuals.png",
        "predictions.npz",
        "baseline_metrics.json",
    }
    assert expected_artifacts <= {path.name for path in output.iterdir()}
    official = json.loads((output / "metrics.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert not (output / "final_model.pt").exists()
    assert not (output / "checkpoint_comparison.json").exists()

    dataset = DataGenerator(**config["data"]).generate()
    preprocessor = joblib.load(output / "preprocessor.joblib")
    baseline = json.loads((output / "baseline_metrics.json").read_text())
    assert baseline["prediction"] == pytest.approx(float(np.mean(dataset["train"][1])))
    from function_approximation.inference import load_predictor

    predictor = load_predictor(output)
    exported = np.load(output / "predictions.npz")
    np.testing.assert_array_equal(
        predictor.predict(dataset["test"][0]),
        exported["test_pred"],
    )
    for name in ("best",):
        checkpoint = torch.load(output / f"{name}_model.pt", weights_only=True)
        model = MLPRegressor(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        assert checkpoint["epoch"] == checkpoint["best_epoch"]
        assert checkpoint["best_val_loss"] == pytest.approx(
            history["val_loss"][checkpoint["epoch"] - 1]
        )
        recomputed = evaluate_splits(model, preprocessor, dataset)
        for split, values in recomputed.items():
            assert set(values) == {"mse", "rmse", "mae", "r2"}
            assert np.isfinite(list(values.values())).all()
            assert values == pytest.approx(official[split])

    experiment = mlflow.get_experiment_by_name(config["experiment"]["name"])
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["epochs_completed"] == len(history["train_loss"])
    assert run.data.metrics["best_epoch"] == checkpoint["best_epoch"]
    assert (
        run.data.metrics["baseline_train_rmse"] == baseline["metrics"]["train"]["rmse"]
    )
    client = mlflow.MlflowClient()
    rmse_history = client.get_metric_history(run.info.run_id, "epoch_val_rmse")
    assert [metric.step for metric in rmse_history] == list(
        range(1, len(history["val_loss"]) + 1)
    )
    np.testing.assert_allclose(
        [metric.value for metric in rmse_history], np.sqrt(history["val_loss"])
    )
    for name in ("best",):
        for split in ("train", "val", "test"):
            assert run.data.metrics[f"{split}_mse"] == official[split]["mse"]
    artifacts = client.list_artifacts(run.info.run_id, "experiment_outputs")
    assert expected_artifacts <= {Path(artifact.path).name for artifact in artifacts}
