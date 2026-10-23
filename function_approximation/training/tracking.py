"""MLflow setup and artifact uploads for training experiments."""

import os
from pathlib import Path

import mlflow

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"


def configure_mlflow(
    experiment_name: str = "function_approximation",
    tracking_uri: str | None = None,
) -> str:
    """Select an experiment using explicit URI, environment, then local default.

    This connects to the tracking store; it does not start a server or a run.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def log_output_artifacts(output_dir: str | Path, tracking_config: dict) -> None:
    """Upload files from a completed run to the active MLflow run.

    Logging switches control checkpoints, preprocessing, and PNG/SVG/PDF figures.
    Other files, including resolved config and metrics, are always uploaded.
    Local files are retained regardless of upload settings; subfolders are skipped.
    """
    switches = {
        ".pt": "log_model",
        ".joblib": "log_preprocessor",
        ".png": "log_figures",
        ".svg": "log_figures",
        ".pdf": "log_figures",
    }
    for path in sorted(Path(output_dir).iterdir()):
        if not path.is_file():
            continue
        switch = switches.get(path.suffix.lower())
        if switch and not tracking_config.get(switch, True):
            continue
        mlflow.log_artifact(str(path), artifact_path="experiment_outputs")
