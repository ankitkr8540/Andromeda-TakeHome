"""Run a reproducible function-approximation experiment from YAML."""

import argparse
import json
import random
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
    plot_checkpoint_comparison,
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
    """Seed Python, NumPy, and PyTorch, including available CUDA generators.

    Args:
        seed: Experiment seed, validated against NumPy's range by the YAML loader.

    Notes:
        This mutates global RNG state. Dataset generation and training-loader
        shuffling also use their own reproducibly seeded generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_splits(model, preprocessor, dataset):
    """Evaluate fixed weights on train, validation, and test observations.

    Args:
        model: Fitted scalar regression module.
        preprocessor: Fitted training input transform.
        dataset: Original-coordinate split mapping of ``(x, y)`` arrays.

    Returns:
        Per-split MSE, RMSE, MAE, and R-squared. Predictions use no labels;
        labels are used only afterward to calculate metrics.
    """
    results = {}

    for split in ("train", "val", "test"):
        x, y = dataset[split]
        predictions = predict_values(model, preprocessor, x)
        results[split] = compute_metrics(y, predictions)

    return results


def run_experiment(config_path: str | Path) -> Path:
    """Execute one validated YAML experiment and return its artifact directory.

    Args:
        config_path: YAML filename, resolved relative to the caller's directory.

    Returns:
        Absolute timestamped output directory. Relative ``experiment.output_dir``
        settings are resolved under the repository root; existing runs are retained.

    Raises:
        FileNotFoundError: The configuration file is absent.
        ValueError: Configuration or input data is invalid.
        FloatingPointError: Initialization, optimization, or prediction is nonfinite.

    Notes:
        Input preprocessing and optional Fourier initialization fit training
        samples only. Validation selects the checkpoint; test scores are computed
        after fitting. The selected and final models are evaluated separately.
        Tracking errors propagate when MLflow is enabled, while completed local
        artifacts remain available. No MLflow server is needed when tracking is off.
    """
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

    # Save the exact configuration used for this execution.
    with (output_dir / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # --------------------------------------------------
    # 1. Data generation
    # --------------------------------------------------
    generator = DataGenerator(
        **data_cfg,
        seed=seed,
    )

    raw_dataset = generator.generate()

    # --------------------------------------------------
    # 2. Preprocessing: fit on train only
    # --------------------------------------------------
    preprocessor = DataPreprocessor(**preprocessing_cfg)
    processed_dataset = preprocessor.fit_transform_dataset(raw_dataset)

    loaders = create_dataloaders(
        processed_dataset,
        batch_size=training_cfg["batch_size"],
        seed=seed,
    )

    # --------------------------------------------------
    # 3. Model initialization
    # --------------------------------------------------
    model_type = model_cfg["type"]

    architecture = {key: value for key, value in model_cfg.items() if key != "type"}

    model = build_model(model_type, architecture)

    if model_type in ("fourier", "damped_fourier"):
        initialization = {
            "method": training_cfg.get("fourier_initialization", "random")
        }
        if initialization["method"] == "spectral":
            initialization = model.initialize_from_data(*processed_dataset["train"])
        else:
            initialization["spectrum"] = model.spectrum()
        with torch.no_grad():
            initial_predictions = model(torch.from_numpy(processed_dataset["train"][0]))
            initialization["model_train_mse"] = float(
                np.mean(
                    (initial_predictions.numpy() - processed_dataset["train"][1]) ** 2
                )
            )
        with (output_dir / "initialization.json").open("w") as file:
            json.dump(initialization, file, indent=2, allow_nan=False)
        print(
            f"{model_type} initialization: {initialization['method']} | Training RMSE: {initialization['model_train_mse'] ** 0.5:.6f}"
        )

    trainer = Trainer(
        model=model,
        learning_rate=training_cfg["learning_rate"],
        max_epochs=training_cfg["max_epochs"],
        patience=training_cfg["patience"],
        min_delta=training_cfg["min_delta"],
        device=training_cfg["device"],
    )

    # --------------------------------------------------
    # 4. Optional MLflow tracking
    # --------------------------------------------------
    tracking_enabled = tracking_cfg.get("enabled", False)

    if tracking_enabled:
        configure_mlflow(experiment_cfg["name"], tracking_cfg.get("tracking_uri"))

    def execute_pipeline():
        # Train and restore best validation checkpoint.
        history = trainer.fit(
            loaders["train"],
            loaders["val"],
        )

        # Evaluate using original x and y coordinates.
        metrics = evaluate_splits(
            trainer.model,
            preprocessor,
            raw_dataset,
        )

        # Reconstruct a separate model; never replace the official weights.
        # Preserve RNG state so diagnostics do not affect later execution.
        with torch.random.fork_rng(devices=[]):
            final_model = build_model(model_type, architecture).to(trainer.device)
        final_model.load_state_dict(trainer.final_model_state)
        final_model.eval()
        final_metrics = evaluate_splits(final_model, preprocessor, raw_dataset)
        baseline = compute_baseline_metrics(raw_dataset)

        if model_type in ("fourier", "damped_fourier"):
            spectrum = {
                "frequency_units": "radians per model-input unit",
                "input_transform": {
                    "mean": preprocessor.mean if preprocessor.mean is not None else 0.0,
                    "scale": preprocessor.std if preprocessor.std is not None else 1.0,
                },
                "best": {"epoch": trainer.best_epoch, **trainer.model.spectrum()},
                "final": {"epoch": trainer.final_epoch, **final_model.spectrum()},
            }
            with (output_dir / "spectrum.json").open("w") as file:
                json.dump(spectrum, file, indent=2, allow_nan=False)

        comparison = {
            "best": {
                "epoch": trainer.best_epoch,
                "validation_loss": trainer.best_val_loss,
                "metrics": metrics,
            },
            "final": {
                "epoch": trainer.final_epoch,
                "last_recorded_train_loss": history["train_loss"][-1],
                "last_recorded_val_loss": history["val_loss"][-1],
                "metrics": final_metrics,
            },
        }

        print("\nCheckpoint diagnostics:")
        print(f"Best validation epoch: {trainer.best_epoch}")
        print(f"Final training epoch: {trainer.final_epoch}")
        print(f"Best validation RMSE: {np.sqrt(trainer.best_val_loss):.6f}")
        print(
            "Last recorded training RMSE (during parameter updates): "
            f"{np.sqrt(history['train_loss'][-1]):.6f}"
        )
        print(f"Last recorded validation RMSE: {np.sqrt(history['val_loss'][-1]):.6f}")
        for split in ("train", "val", "test"):
            print(
                f"Recomputed {split:5s} RMSE | "
                f"Best: {metrics[split]['rmse']:.6f} | "
                f"Final: {final_metrics[split]['rmse']:.6f}"
                f" | Training-mean baseline: {baseline['metrics'][split]['rmse']:.6f}"
            )

        # Save model and fitted preprocessor.
        trainer.save_checkpoint(
            output_dir / "best_model.pt",
            model_config=architecture,
            model_type=model_type,
        )
        trainer.save_final_checkpoint(
            output_dir / "final_model.pt",
            model_config=architecture,
            model_type=model_type,
        )

        joblib.dump(
            preprocessor,
            output_dir / "preprocessor.joblib",
        )

        with (output_dir / "checkpoint_comparison.json").open("w") as file:
            json.dump(comparison, file, indent=2, allow_nan=False)
        with (output_dir / "baseline_metrics.json").open("w") as file:
            json.dump(baseline, file, indent=2, allow_nan=False)

        # Use the persisted fitted preprocessor for both comparison curves.
        comparison_preprocessor = joblib.load(output_dir / "preprocessor.joblib")
        fig = plot_checkpoint_comparison(
            best_model=trainer.model,
            final_model=final_model,
            preprocessor=comparison_preprocessor,
            dataset=raw_dataset,
            target_fn=generator.target_fn,
            split_strategy=data_cfg["split_strategy"],
            title=f"{run_name} — Checkpoint Comparison",
            save_path=output_dir / "checkpoint_comparison.png",
        )
        plt.close(fig)

        with (output_dir / "metrics.json").open("w") as file:
            json.dump(metrics, file, indent=2)

        with (output_dir / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        # Optional plots.
        if evaluation_cfg.get("generate_plots", True):
            figures = [
                plot_learning_curves(
                    history,
                    best_epoch=trainer.best_epoch,
                    title=f"{run_name} — Learning Curves",
                    save_path=output_dir / "learning_curves.png",
                ),
                plot_predictions(
                    model=trainer.model,
                    preprocessor=preprocessor,
                    dataset=raw_dataset,
                    target_fn=generator.target_fn,
                    title=f"{run_name} — Predictions",
                    save_path=output_dir / "predictions.png",
                ),
                plot_residuals(
                    model=trainer.model,
                    preprocessor=preprocessor,
                    dataset=raw_dataset,
                    title=f"{run_name} — Residuals",
                    save_path=output_dir / "residuals.png",
                ),
            ]

            for fig in figures:
                plt.close(fig)

        # Optional prediction export.
        if evaluation_cfg.get("save_predictions", False):
            arrays = {}

            for split, (x, y) in raw_dataset.items():
                arrays[f"{split}_x"] = x
                arrays[f"{split}_y"] = y
                arrays[f"{split}_pred"] = predict_values(
                    trainer.model,
                    preprocessor,
                    x,
                )

            np.savez_compressed(
                output_dir / "predictions.npz",
                **arrays,
            )

        print("\nEvaluation metrics:")
        for split, values in metrics.items():
            print(
                f"{split:5s} | "
                f"RMSE: {values['rmse']:.6f} | "
                f"MAE: {values['mae']:.6f} | "
                f"R²: {values['r2']:.6f}"
            )

        if tracking_enabled:
            # Log all configuration fields as MLflow parameters.
            for section in (
                "data",
                "preprocessing",
                "model",
                "training",
            ):
                for key, value in config[section].items():
                    mlflow.log_param(f"{section}.{key}", value)

            mlflow.log_param("seed", seed)
            mlflow.log_param("split_strategy", data_cfg["split_strategy"])

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
                mlflow.log_metric(
                    "epoch_train_rmse", float(np.sqrt(train_loss)), step=epoch
                )
                mlflow.log_metric(
                    "epoch_val_rmse", float(np.sqrt(val_loss)), step=epoch
                )

            mlflow.log_metric("best_epoch", trainer.best_epoch)
            mlflow.log_metric("final_epoch", trainer.final_epoch)
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)
            for split, values in baseline["metrics"].items():
                for name, value in values.items():
                    mlflow.log_metric(f"baseline_{split}_{name}", value)

            for checkpoint, checkpoint_metrics in (
                ("best", metrics),
                ("final", final_metrics),
            ):
                for split, values in checkpoint_metrics.items():
                    for name, value in values.items():
                        mlflow.log_metric(f"{checkpoint}_{split}_{name}", value)

            for split, values in metrics.items():
                for name, value in values.items():
                    mlflow.log_metric(f"{split}_{name}", value)

            log_output_artifacts(output_dir, tracking_cfg)

        return metrics

    if tracking_enabled:
        with mlflow.start_run(run_name=run_name) as run:
            print(f"MLflow run ID: {run.info.run_id}")
            execute_pipeline()
    else:
        execute_pipeline()

    print(f"\nExperiment completed: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment from a YAML configuration."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to experiment YAML file.",
    )

    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
