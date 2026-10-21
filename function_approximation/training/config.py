
"""YAML configuration loading and validation."""

from pathlib import Path
from typing import Any

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


def load_experiment_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate an experiment YAML file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping.")

    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section not in config
    ]

    if missing:
        raise ValueError(
            f"Missing required configuration sections: {missing}"
        )

    for section in REQUIRED_SECTIONS:
        if not isinstance(config[section], dict):
            raise ValueError(
                f"Section '{section}' must be a YAML mapping."
            )

    if config["model"].get("type") != "mlp":
        raise ValueError(
            "Only model.type='mlp' is currently supported."
        )

    if config["preprocessing"].get("scaling") not in (
        "standard",
        "none",
    ):
        raise ValueError("Unsupported preprocessing scaling method.")

    if config["data"].get("split_strategy") not in (
        "random",
        "chronological",
    ):
        raise ValueError("Unsupported split strategy.")

    if config["training"].get("batch_size", 0) <= 0:
        raise ValueError("Training batch_size must be positive.")

    return config
