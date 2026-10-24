"""Run a reproducible function-approximation experiment from YAML."""

import argparse
import json
import random
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml

from function_approximation.data.dataset import create_dataloaders
from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.evaluation.metrics import (
    compute_baseline_metrics,
    compute_metrics,
)
from function_approximation.evaluation.visualization import (
    plot_extrapolation,
    plot_learning_curves,
    plot_predictions,
    plot_residuals,
)
from function_approximation.inference import predict_values
from function_approximation.models import build_model
from function_approximation.training.config import load_experiment_config
from function_approximation.training.tracking import (
    configure_mlflow,
    log_output_artifacts,
)
from function_approximation.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators in place.

    Args:
        seed: Nonnegative NumPy-compatible integer from validated experiment settings.

    Notes:
        Includes all CUDA devices when available. Data grids are deterministic;
        setting seeds does not guarantee identical results across all hardware."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_splits(model, preprocessor, dataset):
    """Score the same fitted model independently on every supplied split.

    Args:
        model: Fitted regression model; weights are not updated.
        preprocessor: Matching fitted input transform.
        dataset: Mapping of split names to original-coordinate input/target arrays.

    Returns:
        Mapping of split names to MSE, RMSE, MAE, and R² in original target units."""
    results = {}

    for split, (x, y) in dataset.items():
        predictions = predict_values(model, preprocessor, x)
        results[split] = compute_metrics(y, predictions)

    return results


def _save_json(path: Path, values: dict) -> None:
    """Write a report without serializing NaN or infinity."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2, allow_nan=False)


def _initialize_fourier(model, train_data, training_cfg, output_dir) -> None:
    """Fit optional spectral starting parameters using training samples only."""
    initialization = {"method": training_cfg.get("fourier_initialization", "random")}
    if initialization["method"] == "spectral":
        initialization = model.initialize_from_data(*train_data)
    else:
        initialization["spectrum"] = model.spectrum()
    with torch.no_grad():
        initial_predictions = model(torch.from_numpy(train_data[0]))
        initialization["model_train_mse"] = float(
            np.mean((initial_predictions.numpy() - train_data[1]) ** 2)
        )
    _save_json(
        (Path(output_dir) / "diagnostics" / "initialization.json"), initialization
    )
    print(
        f"{model.model_type} initialization: {initialization['method']} | Training RMSE: {initialization['model_train_mse'] ** 0.5:.6f}"
    )


def _save_plots(
    trainer,
    preprocessor,
    generator,
    raw_dataset,
    extrapolation_dataset,
    metrics,
    history,
    run_name,
    output_dir,
) -> None:
    """Save per-run diagnostic plots using the selected model."""
    plots = [
        plot_predictions(
            model=trainer.model,
            preprocessor=preprocessor,
            dataset=raw_dataset,
            target_fn=generator.target_fn,
            title=f"{run_name} — Predictions",
            save_path=(Path(output_dir) / "plots" / "predictions.png"),
        ),
        plot_residuals(
            model=trainer.model,
            preprocessor=preprocessor,
            dataset=raw_dataset,
            title=f"{run_name} — Residuals",
            save_path=(Path(output_dir) / "plots" / "residuals.png"),
        ),
    ]
    if history["train_loss"]:
        plots.append(
            plot_learning_curves(
                history,
                best_epoch=trainer.best_epoch or None,
                title=f"{run_name} — Learning Curves",
                save_path=(Path(output_dir) / "plots" / "learning_curves.png"),
            )
        )
    if extrapolation_dataset:
        plots.append(
            plot_extrapolation(
                model=trainer.model,
                preprocessor=preprocessor,
                train_range=(generator.x_min, generator.x_max),
                bands=extrapolation_dataset,
                band_ranges={
                    f"extrap_{band['name']}": (band["x_min"], band["x_max"])
                    for band in generator.extrapolation_bands
                },
                metrics=metrics,
                target_fn=generator.target_fn,
                title=f"{run_name} — Extrapolation",
                save_path=(Path(output_dir) / "plots" / "extrapolation.png"),
            )
        )

    for fig in plots:
        plt.close(fig)


def _log_run(config, trainer, history, metrics, baseline, output_dir) -> None:
    """Record configuration, epoch curves, metrics, and artifacts in the active run."""
    # Log all configuration fields as MLflow parameters.
    for section in (
        "data",
        "preprocessing",
        "model",
        "training",
    ):
        for key, value in config[section].items():
            mlflow.log_param(f"{section}.{key}", value)

    mlflow.log_param("seed", config["experiment"]["seed"])
    mlflow.log_param("split_strategy", "interpolation")

    # Log training curves as time-series metrics.
    for epoch, (train_loss, val_loss) in enumerate(
        zip(history["train_loss"], history["val_loss"]),
        start=1,
    ):
        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch,
        )
        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch,
        )
        mlflow.log_metric("epoch_train_rmse", float(np.sqrt(train_loss)), step=epoch)
        mlflow.log_metric("epoch_val_rmse", float(np.sqrt(val_loss)), step=epoch)

    mlflow.log_metric("best_epoch", trainer.best_epoch)
    mlflow.log_metric("epochs_completed", len(history["train_loss"]))
    mlflow.log_metric("best_val_loss", trainer.best_val_loss)
    for split, values in baseline["metrics"].items():
        for name, value in values.items():
            mlflow.log_metric(f"baseline_{split}_{name}", value)

    for split, values in metrics.items():
        for name, value in values.items():
            mlflow.log_metric(f"{split}_{name}", value)

    log_output_artifacts(output_dir, config["tracking"])


def run_experiment(config_path: str | Path) -> Path:
    """Train one config-driven experiment and export reproducible run artifacts.

    Args:
        config_path: Experiment YAML file; includes are resolved before validation.

    Returns:
        Unique timestamped run directory containing the resolved configuration,
        selected checkpoint, fitted preprocessor, metrics, histories, and timing.
        Prediction arrays, plots, and MLflow uploads follow configuration switches.

    Raises:
        FileNotFoundError: The experiment config or an include is missing.
        ValueError: Configuration or generated observations are invalid.
        FloatingPointError: Training/evaluation produces nonfinite values.

    Notes:
        Fits preprocessing on training inputs only; validation selects the model.
        Extrapolation data are evaluated separately and do not select parameters.
        Failed runs may leave diagnostic artifacts; existing runs are not overwritten."""
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)

    experiment_cfg = config["experiment"]
    data_cfg = config["data"]
    preprocessing_cfg = config["preprocessing"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    tracking_cfg = config["tracking"]

    seed = experiment_cfg["seed"]
    set_seed(seed)

    # Each execution receives a unique output directory.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = experiment_cfg["run_name"]

    output_root = PROJECT_ROOT / experiment_cfg["output_dir"]
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("model", "data", "metrics", "diagnostics", "plots"):
        (output_dir / folder).mkdir()

    # Save the exact configuration used for this execution.
    with (output_dir / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # 1. Data generation
    generator = DataGenerator(**data_cfg)

    raw_dataset = generator.generate()
    # Evaluation-only bands, possibly outside the training domain. They are
    # kept out of raw_dataset so preprocessing and training never see them.
    extrapolation_dataset = generator.generate_extrapolation()
    evaluation_dataset = {**raw_dataset, **extrapolation_dataset}

    # 2. Preprocessing: fit on train only
    preprocessor = DataPreprocessor(**preprocessing_cfg)
    processed_dataset = preprocessor.fit_transform_dataset(raw_dataset)

    loaders = create_dataloaders(
        processed_dataset,
        batch_size=training_cfg["batch_size"],
        seed=seed,
    )

    # 3. Model initialization
    model_type = model_cfg["type"]

    architecture = {key: value for key, value in model_cfg.items() if key != "type"}

    model = build_model(model_type, architecture)

    fit_started = time.perf_counter()
    initialization_seconds = 0.0
    if model_type == "fourier":
        _initialize_fourier(model, processed_dataset["train"], training_cfg, output_dir)
        initialization_seconds = time.perf_counter() - fit_started

    trainer = Trainer(
        model=model,
        learning_rate=training_cfg["learning_rate"],
        max_epochs=training_cfg["max_epochs"],
        patience=training_cfg["patience"],
        min_delta=training_cfg["min_delta"],
        device=training_cfg["device"],
        optimizer="lbfgs" if training_cfg.get("fourier_fit") == "lbfgs" else "adam",
        snake_learning_rate=training_cfg.get("snake_learning_rate"),
    )

    # 4. Optional MLflow tracking
    tracking_enabled = tracking_cfg.get("enabled", False)

    if tracking_enabled:
        configure_mlflow(experiment_cfg["name"], tracking_cfg.get("tracking_uri"))

    context = mlflow.start_run(run_name=run_name) if tracking_enabled else nullcontext()
    with context as run:
        if run is not None:
            print(f"MLflow run ID: {run.info.run_id}")
        # Train and restore best validation checkpoint.
        optimization_started = time.perf_counter()
        if training_cfg.get("fourier_fit") == "spectral_only":
            trainer.retain_initialized_model(loaders["val"])
            history = trainer.history
        elif model_type == "fourier":
            history = trainer.fit(
                loaders["train"], loaders["val"], include_initial=True
            )
        else:
            history = trainer.fit(loaders["train"], loaders["val"])
        _save_json(
            (Path(output_dir) / "diagnostics" / "timing.json"),
            {
                "initialization_seconds": initialization_seconds,
                "optimization_and_selection_seconds": time.perf_counter()
                - optimization_started,
                "fit_seconds": time.perf_counter() - fit_started,
            },
        )

        # Evaluate using original x and y coordinates.
        metrics = evaluate_splits(
            trainer.model,
            preprocessor,
            evaluation_dataset,
        )

        baseline = compute_baseline_metrics(evaluation_dataset)

        if model_type == "fourier":
            spectrum = {
                "frequency_units": "radians per model-input unit",
                "input_transform": {
                    "mean": preprocessor.mean if preprocessor.mean is not None else 0.0,
                    "scale": preprocessor.std if preprocessor.std is not None else 1.0,
                },
                "best": {"epoch": trainer.best_epoch, **trainer.model.spectrum()},
            }
            _save_json((Path(output_dir) / "diagnostics" / "spectrum.json"), spectrum)

        print(f"\nBest validation epoch: {trainer.best_epoch}")
        print(f"Best validation RMSE: {np.sqrt(trainer.best_val_loss):.6f}")

        # Save model and fitted preprocessor.
        trainer.save_checkpoint(
            (Path(output_dir) / "model" / "best_model.pt"),
            model_config=architecture,
            model_type=model_type,
        )
        joblib.dump(
            preprocessor,
            (Path(output_dir) / "model" / "preprocessor.joblib"),
        )

        _save_json((Path(output_dir) / "metrics" / "baseline_metrics.json"), baseline)

        _save_json((Path(output_dir) / "metrics" / "metrics.json"), metrics)

        _save_json((Path(output_dir) / "diagnostics" / "history.json"), history)

        if evaluation_cfg.get("generate_plots", True):
            _save_plots(
                trainer,
                preprocessor,
                generator,
                raw_dataset,
                extrapolation_dataset,
                metrics,
                history,
                run_name,
                output_dir,
            )

        # Optional prediction export.
        if evaluation_cfg.get("save_predictions", False):
            arrays = {}

            for split, (x, y) in evaluation_dataset.items():
                arrays[f"{split}_x"] = x
                arrays[f"{split}_y"] = y
                arrays[f"{split}_pred"] = predict_values(
                    trainer.model,
                    preprocessor,
                    x,
                )

            np.savez_compressed(
                (Path(output_dir) / "data" / "predictions.npz"),
                **arrays,
            )

        print("\nEvaluation metrics:")
        for split, values in metrics.items():
            print(
                f"{split:14s} | "
                f"RMSE: {values['rmse']:.6f} | "
                f"MAE: {values['mae']:.6f} | "
                f"R²: {values['r2']:.6f}"
            )

        if tracking_enabled:
            _log_run(config, trainer, history, metrics, baseline, output_dir)

    print(f"\nExperiment completed: {output_dir}")
    return output_dir


def main() -> None:
    """Parse experiment config paths and train each requested experiment from the command line."""
    parser = argparse.ArgumentParser(
        description="Run an experiment from a YAML configuration."
    )

    parser.add_argument(
        "--config",
        required=True,
        nargs="+",
        type=str,
        help="One or more experiment YAML files, run in the given order.",
    )

    args = parser.parse_args()
    for config_path in args.config:
        run_experiment(config_path)


if __name__ == "__main__":
    main()
