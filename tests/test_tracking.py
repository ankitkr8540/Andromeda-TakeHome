"""Tracking URI precedence and independent artifact switches."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from function_approximation.training.tracking import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    log_output_artifacts,
)


@pytest.mark.parametrize(
    "disabled, excluded",
    [
        (None, set()),
        ("log_model", {"best_model.pt"}),
        ("log_preprocessor", {"preprocessor.joblib"}),
        (
            "log_figures",
            {
                "predictions.png",
                "extrapolation.png",
                "distance.svg",
                "distance.pdf",
                "preview.PNG",
            },
        ),
    ],
)
def test_logging_switches_filter_only_uploads(
    tmp_path, monkeypatch, disabled, excluded
):
    names = {
        "best_model.pt",
        "preprocessor.joblib",
        "predictions.png",
        "extrapolation.png",
        "distance.svg",
        "distance.pdf",
        "preview.PNG",
        "metrics.json",
        "config.yaml",
        "baseline_metrics.json",
    }
    for name in names:
        (tmp_path / name).write_text("test")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.txt").write_text("nested artifact")
    upload = Mock()
    monkeypatch.setattr("mlflow.log_artifact", upload)
    log_output_artifacts(tmp_path, {disabled: False} if disabled else {})
    assert {
        Path(call.args[0]).name for call in upload.call_args_list
    } == names - excluded
    assert all(
        call.kwargs["artifact_path"] == "experiment_outputs"
        for call in upload.call_args_list
    )
    assert {path.name for path in tmp_path.iterdir() if path.is_file()} == names


def test_shared_uri_precedence(monkeypatch):
    setter = Mock()
    monkeypatch.setattr("mlflow.set_tracking_uri", setter)
    monkeypatch.setattr("mlflow.set_experiment", Mock())
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert configure_mlflow() == DEFAULT_TRACKING_URI == "http://127.0.0.1:5001"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:6000")
    assert configure_mlflow() == "http://localhost:6000"
    assert (
        configure_mlflow(tracking_uri="sqlite:///explicit.db")
        == "sqlite:///explicit.db"
    )
    setter.assert_called_with("sqlite:///explicit.db")
