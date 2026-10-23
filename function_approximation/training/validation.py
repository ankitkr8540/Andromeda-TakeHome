"""Configuration checks grouped by experiment section.

Run once after YAML composition, before allocating outputs or training models.
"""

import math
import re

import torch

from function_approximation.data.generation import DataGenerator, split_counts

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
) -> None:
    """Validate a required finite numeric field, its type, and its allowed lower bound."""
    value = section.get(name)
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid_type or not math.isfinite(value):
        kind = "integer" if integer else "number"
        raise ValueError(f"{prefix}.{name} must be a finite {kind}.")
    if value < minimum or (not inclusive and value == minimum):
        comparison = ">=" if inclusive else ">"
        raise ValueError(f"{prefix}.{name} must be {comparison} {minimum}.")


def _text(section: dict, name: str, prefix: str) -> str:
    """Return a required nonblank string, raising ValueError with its field path otherwise."""
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{name} must be a nonempty string.")
    return value


def _boolean(section: dict, name: str, prefix: str, default: bool) -> None:
    """Validate an optional boolean setting using the supplied default when absent."""
    if not isinstance(section.get(name, default), bool):
        raise ValueError(f"{prefix}.{name} must be a boolean.")  # noqa: TRY004 -- invalid YAML setting


def validate_config(config: dict) -> None:
    """Normalize numeric YAML settings and validate an entire experiment in place.

    Args:
        config: Resolved mapping with experiment, data, preprocessing, model,
            training, evaluation, and tracking sections.

    Raises:
        ValueError: A required section/field is missing, unsupported, or invalid.

    Notes:
        Mutates known scientific-notation strings into floats. Does not create
        outputs, initialize model parameters, generate data, or train a model."""
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config[section], dict):
            raise ValueError(f"Section '{section}' must be a YAML mapping.")  # noqa: TRY004 -- preserve public error type

    # PyYAML's YAML 1.1 resolver reads 1e-3 as a string. Normalize only
    # scientific notation in known real-valued fields, then validate it.
    for section, names in (
        ("training", ("learning_rate", "min_delta", "snake_learning_rate")),
        ("data", ("x_min", "x_max")),
        ("model", ("frequency_min", "frequency_max", "a0", "x_scale")),
    ):
        for name in names:
            value = config[section].get(name)
            if isinstance(value, str) and re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", value
            ):
                config[section][name] = float(value)

    _validate_experiment(config["experiment"])
    _validate_data(config["data"])
    _validate_preprocessing(config["preprocessing"])
    _validate_model(config["model"])
    _validate_training(config)
    _validate_evaluation(config["evaluation"])
    _validate_tracking(config["tracking"])


def _validate_experiment(experiment: dict) -> None:
    """Validate run identifiers, output location, and the NumPy-compatible random seed."""
    for name in ("name", "run_name", "output_dir"):
        _text(experiment, name, "experiment")
    _number(experiment, "seed", "experiment", integer=True)
    if experiment["seed"] > 2**32 - 1:
        raise ValueError("experiment.seed must be in [0, 2**32 - 1] for NumPy seeding.")


def _validate_data(data: dict) -> None:
    """Validate data configuration keys and delegate sampling rules to DataGenerator."""
    allowed_data = {
        "sample_size",
        "x_min",
        "x_max",
        "target",
        "extrapolation_bands",
    }
    if set(data) - allowed_data:
        raise ValueError(
            f"Unknown data settings: {sorted(set(data) - allowed_data)}; use experiment.seed for the seed."
        )
    _number(data, "sample_size", "data", minimum=20, integer=True)
    for name in ("x_min", "x_max"):
        _number(data, name, "data", minimum=-math.inf)
    if not isinstance(data.get("extrapolation_bands", []), list):
        raise ValueError("data.extrapolation_bands must be a list.")  # noqa: TRY004 -- invalid YAML setting
    # Use the same interval, count, target and band rules as direct Python calls.
    DataGenerator(**data)


def _validate_preprocessing(preprocessing: dict) -> None:
    """Require one supported input scaling mode and reject unknown settings."""
    if set(preprocessing) - {"scaling"} or preprocessing.get("scaling") not in (
        "standard",
        "none",
    ):
        raise ValueError("preprocessing must specify scaling='standard' or 'none'.")


def _validate_model(model: dict) -> None:
    """Validate model-family settings, widths, frequencies, and scalar regression dimensions."""
    if model.get("type") not in ("mlp", "fourier", "snake"):
        raise ValueError("model.type must be 'mlp', 'fourier', or 'snake'.")
    if model["type"] == "mlp":
        defaults = {
            "input_dim": 1,
            "output_dim": 1,
        }
        allowed = {"type", "hidden_dims", "activation", *defaults}
    elif model["type"] == "snake":
        defaults = {}
        allowed = {"type", "dims", "a0", "x_scale"}
        widths = model.get("dims", [64, 256, 128])
        if (
            not isinstance(widths, list)
            or not widths
            or any(
                isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in widths
            )
        ):
            raise ValueError("model.dims must be a nonempty list of positive integers.")
        _number(
            {"a0": model.get("a0", 1.0)}, "a0", "model", minimum=1e-6, inclusive=False
        )
        _number(
            {"x_scale": model.get("x_scale", 1.0)}, "x_scale", "model", inclusive=False
        )
    else:
        defaults = {"num_frequencies": 8}
        allowed = {
            "type",
            *defaults,
            "frequency_min",
            "frequency_max",
            "frequency_initialization",
        }
    if set(model) - allowed:
        raise ValueError(f"Unknown model settings: {sorted(set(model) - allowed)}")
    for name, default in defaults.items():
        _number(
            {name: model.get(name, default)}, name, "model", minimum=1, integer=True
        )
    if model["type"] == "mlp":
        if model.get("activation", "tanh") not in ("tanh", "relu", "silu", "snake"):
            raise ValueError("model.activation must be tanh, relu, silu, or snake.")
        widths = model.get("hidden_dims", [64, 64])
        if (
            not isinstance(widths, list)
            or not widths
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in widths
            )
        ):
            raise ValueError(
                "model.hidden_dims must be a nonempty list of positive integers."
            )
    if model.get("input_dim", 1) != 1 or model.get("output_dim", 1) != 1:
        raise ValueError(
            "Scalar regression requires model.input_dim=1 and model.output_dim=1."
        )
    if model["type"] == "fourier":
        if model.get("frequency_initialization", "uniform") not in (
            "uniform",
            "log_uniform",
        ):
            raise ValueError(
                "model.frequency_initialization must be uniform or log_uniform."
            )
        for name, default in (("frequency_min", 0.1), ("frequency_max", 5.0)):
            _number({name: model.get(name, default)}, name, "model", inclusive=False)
        if model.get("frequency_min", 0.1) >= model.get("frequency_max", 5.0):
            raise ValueError(
                "model.frequency_min must be less than model.frequency_max."
            )


def _validate_training(config: dict) -> None:
    """Validate optimizer, stopping rules, device, and model-specific fitting options."""
    allowed = {
        "batch_size",
        "learning_rate",
        "max_epochs",
        "patience",
        "min_delta",
        "device",
        "snake_learning_rate",
        "fourier_fit",
        "fourier_initialization",
    }
    unknown = set(config["training"]) - allowed
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    model, data = config["model"], config["data"]
    training = config["training"]
    if "snake_learning_rate" in training:
        _number(training, "snake_learning_rate", "training", inclusive=False)
        if model["type"] != "snake" and not (
            model["type"] == "mlp" and model.get("activation") == "snake"
        ):
            raise ValueError(
                "training.snake_learning_rate requires a Snake model or model.activation=snake."
            )
    fit_mode = training.get("fourier_fit", "adam")
    if fit_mode not in ("adam", "lbfgs", "spectral_only"):
        raise ValueError("training.fourier_fit must be adam, lbfgs, or spectral_only.")
    if "snake_learning_rate" in training and fit_mode != "adam":
        raise ValueError("training.snake_learning_rate requires Adam.")
    if "fourier_fit" in training and model["type"] != "fourier":
        raise ValueError("training.fourier_fit requires the Fourier model.")
    if (
        fit_mode == "spectral_only"
        and training.get("fourier_initialization") != "spectral"
    ):
        raise ValueError("spectral_only requires spectral initialization.")
    if "fourier_initialization" in training:
        if model["type"] != "fourier":
            raise ValueError(
                "training.fourier_initialization requires the Fourier model."
            )
        if training["fourier_initialization"] not in ("random", "spectral"):
            raise ValueError(
                "training.fourier_initialization must be 'random' or 'spectral'."
            )
    if (
        model["type"] == "fourier"
        and training.get("fourier_initialization", "random") == "spectral"
        and split_counts(data["sample_size"])[0]
        < 2 * model.get("num_frequencies", 8) + 1
    ):
        raise ValueError(
            "Spectral initialization requires the derived training count >= 2 * model.num_frequencies + 1."
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


def _validate_evaluation(evaluation: dict) -> None:
    """Validate evaluation switches and reject unsupported settings."""
    for name, default in (("generate_plots", True), ("save_predictions", False)):
        _boolean(evaluation, name, "evaluation", default)
    if set(evaluation) - {"generate_plots", "save_predictions"}:
        raise ValueError(
            f"Unknown evaluation settings: {sorted(set(evaluation) - {'generate_plots', 'save_predictions'})}"
        )


def _validate_tracking(tracking: dict) -> None:
    """Validate MLflow settings and artifact-logging switches."""
    for name, default in (
        ("enabled", False),
        ("log_model", True),
        ("log_preprocessor", True),
        ("log_figures", True),
    ):
        _boolean(tracking, name, "tracking", default)
    if "tracking_uri" in tracking:
        _text(tracking, "tracking_uri", "tracking")
