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
    plot_checkpoint_comparison,
    plot_learning_curves,
    predict_values,
)
from function_approximation.models.mlp import MLPRegressor
from function_approximation.training.trainer import Trainer
from scripts.run_experiments import evaluate_splits, run_experiment

ARCHITECTURE = {"hidden_dim": 4, "num_hidden_layers": 1}
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
    assert trainer.final_epoch == expected_final
    assert len(trainer.history["train_loss"]) == expected_final

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, completed[0][key])
        assert torch.equal(trainer.final_model_state[key], completed[-1][key])
        assert value.data_ptr() != trainer.final_model_state[key].data_ptr()

    trainer.save_checkpoint(tmp_path / "best.pt", ARCHITECTURE)
    trainer.save_final_checkpoint(tmp_path / "final.pt", ARCHITECTURE)
    active_state = deepcopy(trainer.model.state_dict())
    loaded_models = {}
    for name, epoch, expected in (
        ("best", 1, completed[0]),
        ("final", expected_final, completed[-1]),
    ):
        checkpoint = torch.load(tmp_path / f"{name}.pt", weights_only=True)
        assert checkpoint["epoch"] == epoch
        model = MLPRegressor(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected[key])
        loaded_models[name] = model

    generator = DataGenerator(n_train=12, n_val=4, n_test=4)
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
    for save in (trainer.save_checkpoint, trainer.save_final_checkpoint):
        with pytest.raises(RuntimeError, match="Train the model"):
            save(tmp_path / "model.pt", ARCHITECTURE)


def test_real_training_captures_last_completed_epoch(monkeypatch):
    torch.manual_seed(42)
    dataset = DataGenerator(n_train=24, n_val=8, n_test=8).generate()
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
    assert trainer.final_epoch == len(snapshots) == 3
    active_state = deepcopy(trainer.model.state_dict())
    for key, value in active_state.items():
        assert torch.equal(value, snapshots[trainer.best_epoch - 1][key])
        assert torch.equal(trainer.final_model_state[key], snapshots[-1][key])

    final_model = MLPRegressor(**ARCHITECTURE)
    final_model.load_state_dict(trainer.final_model_state)
    inference_calls = []

    def check_inference(model, inputs):
        inference_calls.append(
            (model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )

    final_model.register_forward_pre_hook(check_inference)
    evaluate_splits(final_model, preprocessor, dataset)
    assert inference_calls == [(False, False, True)] * 3
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, active_state[key])


@pytest.mark.parametrize("strategy", ["random", "chronological"])
def test_comparison_plot(tmp_path, strategy):
    generator = DataGenerator(n_train=12, n_val=4, n_test=4, split_strategy=strategy)
    dataset = generator.generate()
    fitted = DataPreprocessor().fit(dataset["train"][0])
    joblib.dump(fitted, tmp_path / "preprocessor.joblib")
    preprocessor = joblib.load(tmp_path / "preprocessor.joblib")
    best = MLPRegressor(**ARCHITECTURE)
    final = MLPRegressor(**ARCHITECTURE)
    before = deepcopy(best.state_dict())
    fig = plot_checkpoint_comparison(
        best,
        final,
        preprocessor,
        dataset,
        generator.target_fn,
        split_strategy=strategy,
        save_path=tmp_path / "comparison.png",
        n_points=100,
    )
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == (3 if strategy == "chronological" else 0)
        grid = ax.lines[0].get_xdata()
        x_all = np.concatenate([x for x, _ in dataset.values()])
        assert grid[0] == x_all.min()
        assert grid[-1] == x_all.max()
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), predict_values(best, preprocessor, grid)
        )
        np.testing.assert_allclose(
            ax.lines[2].get_ydata(), predict_values(final, preprocessor, grid)
        )
        assert (tmp_path / "comparison.png").is_file()
        assert best.training  # Prediction helper restores the caller's mode.
        for key, value in best.state_dict().items():
            assert torch.equal(value, before[key])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("config_name", ["mlp_random.yaml", "mlp_chronological.yaml"])
def test_short_experiment_artifacts_and_metrics(tmp_path, monkeypatch, config_name):
    """Exercise real training, serialization and a single local MLflow run."""
    monkeypatch.chdir(tmp_path)
    config = yaml.safe_load((PROJECT_ROOT / "configs" / config_name).read_text())
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["data"].update(n_train=24, n_val=8, n_test=8)
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
        "final_model.pt",
        "checkpoint_comparison.json",
        "checkpoint_comparison.png",
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
    comparison = json.loads((output / "checkpoint_comparison.json").read_text())
    official = json.loads((output / "metrics.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert official == comparison["best"]["metrics"]
    assert comparison["final"]["epoch"] == len(history["train_loss"])
    assert comparison["final"]["last_recorded_train_loss"] == history["train_loss"][-1]
    assert comparison["final"]["metrics"]["val"]["mse"] == pytest.approx(
        history["val_loss"][-1],
        rel=1e-5,
        abs=1e-7,
    )

    dataset = DataGenerator(
        **config["data"], seed=config["experiment"]["seed"]
    ).generate()
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
    for name in ("best", "final"):
        checkpoint = torch.load(output / f"{name}_model.pt", weights_only=True)
        model = MLPRegressor(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        assert checkpoint["epoch"] == comparison[name]["epoch"]
        recomputed = evaluate_splits(model, preprocessor, dataset)
        for split, values in recomputed.items():
            assert set(values) == {"mse", "rmse", "mae", "r2"}
            assert np.isfinite(list(values.values())).all()
            assert values == pytest.approx(comparison[name]["metrics"][split])

    experiment = mlflow.get_experiment_by_name(config["experiment"]["name"])
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["final_epoch"] == comparison["final"]["epoch"]
    assert run.data.metrics["best_epoch"] == comparison["best"]["epoch"]
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
    for name in ("best", "final"):
        for split in ("train", "val", "test"):
            assert (
                run.data.metrics[f"{name}_{split}_mse"]
                == comparison[name]["metrics"][split]["mse"]
            )
    artifacts = client.list_artifacts(run.info.run_id, "experiment_outputs")
    assert expected_artifacts <= {Path(artifact.path).name for artifact in artifacts}
