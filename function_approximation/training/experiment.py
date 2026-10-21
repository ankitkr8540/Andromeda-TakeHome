"""MLflow experiment tracking for the Andromeda take-home project."""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from function_approximation.data.generation import (
    DataGenerator,
    Dataset,
)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DATA_EXPERIMENT_NAME = "data_generation"


def configure_mlflow(
    experiment_name: str = DATA_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Configure the MLflow tracking server and experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Optional tracking server URL. Defaults to
            MLFLOW_TRACKING_URI or the local MLflow server.

    Returns:
        The configured tracking URI.

    Notes:
        URI precedence is an explicit argument, then ``MLFLOW_TRACKING_URI``,
        then ``http://127.0.0.1:5001``. This contacts the tracking store to select
        an experiment; it does not start a server or a training run.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    return uri


def log_output_artifacts(output_dir: str | Path, tracking_config: dict) -> None:
    """Upload run artifacts while honoring the YAML logging switches.

    Local artifacts are always retained. Checkpoints, the preprocessor,
    and figures can independently be excluded from MLflow uploads.

    Args:
        output_dir: Completed run directory; only immediate files are uploaded.
        tracking_config: Logging flags. ``log_model`` controls ``.pt`` files,
            ``log_preprocessor`` controls ``.joblib``, and ``log_figures`` controls
            ``.png``. JSON reports and configuration files are always included.

    Notes:
        Requires an active MLflow run configured by the caller. This uploads
        artifacts under ``experiment_outputs`` and does not register a model.
    """
    for path in sorted(Path(output_dir).iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".pt" and not tracking_config.get("log_model", True):
            continue
        if path.suffix == ".joblib" and not tracking_config.get(
            "log_preprocessor", True
        ):
            continue
        if path.suffix == ".png" and not tracking_config.get("log_figures", True):
            continue
        mlflow.log_artifact(str(path), artifact_path="experiment_outputs")


def _dataset_parameters(generator: DataGenerator) -> dict[str, Any]:
    """Collect dataset-generation configuration parameters."""
    return {
        "n_train": generator.n_train,
        "n_val": generator.n_val,
        "n_test": generator.n_test,
        "x_min": generator.x_min,
        "x_max": generator.x_max,
        "seed": generator.seed,
        "sampling": generator.sampling,
        "split_strategy": generator.split_strategy,
    }


def _dataset_metrics(dataset: Dataset) -> dict[str, float]:
    """Compute diagnostic metrics for each dataset split."""
    metrics = {}

    for split, (x, y) in dataset.items():
        metrics.update(
            {
                f"{split}_count": float(len(x)),
                f"{split}_x_min": float(np.min(x)),
                f"{split}_x_max": float(np.max(x)),
                f"{split}_x_mean": float(np.mean(x)),
                f"{split}_x_std": float(np.std(x)),
                f"{split}_y_min": float(np.min(y)),
                f"{split}_y_max": float(np.max(y)),
                f"{split}_y_mean": float(np.mean(y)),
                f"{split}_y_std": float(np.std(y)),
            }
        )

    return metrics


def log_data_experiment(
    generator: DataGenerator,
    dataset: Dataset,
    run_name: str | None = None,
    experiment_name: str = DATA_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    config_path: str | Path | None = None,
    log_dataset: bool = True,
    show_target: bool = True,
) -> str:
    """Track a complete dataset-generation experiment in MLflow.

    Logs:
        - Dataset configuration parameters
        - Statistical diagnostics for each split
        - Dataset visualization
        - Optional dataset arrays as an NPZ artifact
        - Optional configuration file

    Args:
        generator: Configured DataGenerator instance.
        dataset: Generated train/validation/test data.
        run_name: Optional descriptive MLflow run name.
        experiment_name: MLflow experiment name.
        tracking_uri: Optional tracking server URL.
        config_path: Optional YAML configuration file.
        log_dataset: Whether to store dataset arrays.
        show_target: Whether to include the ground-truth curve
            in the exploratory dataset visualization.

    Returns:
        MLflow run ID.
    """
    configure_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    if run_name is None:
        run_name = (
            f"{generator.sampling}_{generator.split_strategy}_seed{generator.seed}"
        )

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log configuration parameters.
        mlflow.log_params(_dataset_parameters(generator))

        # 2. Add searchable experiment metadata.
        mlflow.set_tags(
            {
                "project": "andromeda_takehome",
                "phase": "data_generation",
                "experiment_type": generator.split_strategy,
                "target": "synthetic_periodic_function",
            }
        )

        # 3. Log dataset diagnostics.
        mlflow.log_metrics(_dataset_metrics(dataset))

        # 4. Log dataset visualization.
        fig, _ = generator.plot(
            dataset,
            show_target=show_target,
        )

        try:
            mlflow.log_figure(
                fig,
                "figures/dataset.png",
            )
        finally:
            plt.close(fig)

        # 5. Optionally store the generated dataset.
        if log_dataset:
            arrays = {}

            for split, (x, y) in dataset.items():
                arrays[f"{split}_x"] = x
                arrays[f"{split}_y"] = y

            with _temporary_npz(arrays) as dataset_path:
                mlflow.log_artifact(
                    str(dataset_path),
                    artifact_path="datasets",
                )

        # 6. Optionally log the experiment configuration.
        if config_path is not None:
            config_file = Path(config_path)

            if not config_file.is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            mlflow.log_artifact(
                str(config_file),
                artifact_path="configs",
            )

        return run.info.run_id


import tempfile
from contextlib import contextmanager


@contextmanager
def _temporary_npz(arrays: dict[str, np.ndarray]):
    """Create a temporary NPZ file for MLflow artifact logging."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.npz"
        np.savez_compressed(path, **arrays)
        yield path
