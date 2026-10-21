"""Validate YAML experiments before allocating outputs or starting training."""

import math
import re
from pathlib import Path
from typing import Any

import torch
import yaml

REQUIRED_SECTIONS = (
    "experiment",
    "data",
    "preprocessing",
    "model",
    "training",
    "evaluation",
    "tracking",
)


def _number(
    section: dict,
    name: str,
    prefix: str,
    *,
    minimum: float = 0,
    integer: bool = False,
    inclusive: bool = True,
) -> int | float:
    value = section.get(name)
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid_type or not math.isfinite(value):
        kind = "integer" if integer else "number"
        raise ValueError(f"{prefix}.{name} must be a finite {kind}.")
    if value < minimum or (not inclusive and value == minimum):
        comparison = ">=" if inclusive else ">"
        raise ValueError(f"{prefix}.{name} must be {comparison} {minimum}.")
    return value


def _text(section: dict, name: str, prefix: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{name} must be a nonempty string.")
    return value


def _boolean(section: dict, name: str, prefix: str, default: bool) -> None:
    if not isinstance(section.get(name, default), bool):
        raise ValueError(f"{prefix}.{name} must be a boolean.")  # noqa: TRY004 -- invalid YAML setting


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and validate experiment settings before creating outputs.

    Args:
        path: Configuration filename. Relative paths use the caller's current
            directory, not the directory containing this module.

    Returns:
        Configuration mapping. Scientific-notation strings in supported real
        numeric fields are normalized to numbers; optional defaults are validated
        without inserting them into the returned mapping.

    Raises:
        FileNotFoundError: The configuration file does not exist.
        ValueError: Required sections, model settings, data counts, training
            values, device availability, or logging switches are invalid.

    Notes:
        This validates settings, not dataset contents or MLflow connectivity.
        Spectral Fourier initialization additionally requires at least
        ``2 * num_frequencies + 1`` training samples.
        YAML syntax errors propagate as ``yaml.YAMLError``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping.")  # noqa: TRY004 -- preserve public error type
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config[section], dict):
            raise ValueError(f"Section '{section}' must be a YAML mapping.")  # noqa: TRY004 -- preserve public error type

    # PyYAML's YAML 1.1 resolver reads 1e-3 as a string. Normalize only
    # scientific notation in known real-valued fields, then validate it.
    for section, names in (
        ("training", ("learning_rate", "min_delta")),
        ("data", ("x_min", "x_max")),
        ("model", ("frequency_min", "frequency_max")),
    ):
        for name in names:
            value = config[section].get(name)
            if isinstance(value, str) and re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", value
            ):
                config[section][name] = float(value)

    experiment = config["experiment"]
    for name in ("name", "run_name", "output_dir"):
        _text(experiment, name, "experiment")
    _number(experiment, "seed", "experiment", integer=True)
    if experiment["seed"] > 2**32 - 1:
        raise ValueError("experiment.seed must be in [0, 2**32 - 1] for NumPy seeding.")

    data = config["data"]
    allowed_data = {
        "n_train",
        "n_val",
        "n_test",
        "x_min",
        "x_max",
        "sampling",
        "split_strategy",
        "target",
    }
    if set(data) - allowed_data:
        raise ValueError(
            f"Unknown data settings: {sorted(set(data) - allowed_data)}; use experiment.seed for the seed."
        )
    for name in ("n_train", "n_val", "n_test"):
        _number(data, name, "data", minimum=2, integer=True)
    for name in ("x_min", "x_max"):
        value = data.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"data.{name} must be a finite number.")
    if data["x_min"] >= data["x_max"]:
        raise ValueError("data.x_min must be less than data.x_max.")
    if data.get("sampling") not in ("uniform", "grid"):
        raise ValueError("data.sampling must be 'uniform' or 'grid'.")
    if data.get("split_strategy") not in ("random", "chronological"):
        raise ValueError("data.split_strategy must be 'random' or 'chronological'.")
    if data.get("target", "rich_periodic") not in (
        "rich_periodic",
        "damped_oscillation",
    ):
        raise ValueError(
            "data.target must be 'rich_periodic' or 'damped_oscillation'."
        )

    preprocessing = config["preprocessing"]
    if set(preprocessing) - {"scaling"} or preprocessing.get("scaling") not in (
        "standard",
        "none",
    ):
        raise ValueError("preprocessing must specify scaling='standard' or 'none'.")

    model = config["model"]
    if model.get("type") not in ("mlp", "fourier", "damped_fourier"):
        raise ValueError("model.type must be 'mlp', 'fourier', or 'damped_fourier'.")
    if model["type"] == "mlp":
        defaults = {
            "input_dim": 1,
            "hidden_dim": 64,
            "num_hidden_layers": 2,
            "output_dim": 1,
        }
        allowed = {"type", *defaults}
    else:
        defaults = {"num_frequencies": 8}
        allowed = {"type", *defaults, "frequency_min", "frequency_max"}
        if model["type"] == "damped_fourier":
            allowed.add("initial_decay")
    if set(model) - allowed:
        raise ValueError(f"Unknown model settings: {sorted(set(model) - allowed)}")
    for name, default in defaults.items():
        _number(
            {name: model.get(name, default)}, name, "model", minimum=1, integer=True
        )
    if model.get("input_dim", 1) != 1 or model.get("output_dim", 1) != 1:
        raise ValueError(
            "Scalar regression requires model.input_dim=1 and model.output_dim=1."
        )
    if model["type"] in ("fourier", "damped_fourier"):
        for name, default in (("frequency_min", 0.1), ("frequency_max", 5.0)):
            _number({name: model.get(name, default)}, name, "model", inclusive=False)
        if model.get("frequency_min", 0.1) >= model.get("frequency_max", 5.0):
            raise ValueError(
                "model.frequency_min must be less than model.frequency_max."
            )
    if model["type"] == "damped_fourier":
        _number(
            {"initial_decay": model.get("initial_decay", 0.02)},
            "initial_decay",
            "model",
            inclusive=False,
        )
    training = config["training"]
    if "fourier_initialization" in training:
        if model["type"] not in ("fourier", "damped_fourier"):
            raise ValueError(
                "training.fourier_initialization requires a Fourier-family model."
            )
        if training["fourier_initialization"] not in ("random", "spectral"):
            raise ValueError(
                "training.fourier_initialization must be 'random' or 'spectral'."
            )
    if (
        model["type"] in ("fourier", "damped_fourier")
        and training.get("fourier_initialization", "random") == "spectral"
        and data["n_train"] < 2 * model.get("num_frequencies", 8) + 1
    ):
        raise ValueError(
            "Spectral initialization requires data.n_train >= 2 * model.num_frequencies + 1."
        )
    for name in ("batch_size", "max_epochs", "patience"):
        _number(training, name, "training", minimum=1, integer=True)
    _number(training, "learning_rate", "training", inclusive=False)
    _number(training, "min_delta", "training")
    device_name = _text(training, "device", "training")
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid training.device: {device_name}") from error
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("training.device must be cpu, cuda, or mps.")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested CUDA device is unavailable: {device_name}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable.")

    evaluation = config["evaluation"]
    for name, default in (("generate_plots", True), ("save_predictions", False)):
        _boolean(evaluation, name, "evaluation", default)
    metrics = evaluation.get("metrics", ["mse", "rmse", "mae", "r2"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(
            not isinstance(metric, str) or metric not in ("mse", "rmse", "mae", "r2")
            for metric in metrics
        )
    ):
        raise ValueError("evaluation.metrics must list supported regression metrics.")

    tracking = config["tracking"]
    for name, default in (
        ("enabled", False),
        ("log_model", True),
        ("log_preprocessor", True),
        ("log_figures", True),
    ):
        _boolean(tracking, name, "tracking", default)
    if "tracking_uri" in tracking:
        _text(tracking, "tracking_uri", "tracking")
    return config
