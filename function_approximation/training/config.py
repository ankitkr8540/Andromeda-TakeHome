"""Compose experiment YAML files and validate the resolved configuration."""

from pathlib import Path
from typing import Any

import yaml

from function_approximation.training.validation import validate_config


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; later scalar values and lists replace earlier ones."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config(path: str | Path, stack: tuple[Path, ...] = ()) -> dict:
    """Resolve nested YAML includes and merge experiment settings.

    Args:
        path: YAML file to load. Included paths are relative to their owner.
        stack: Active include chain used internally to detect cycles.

    Returns:
        New merged dictionary. Includes apply in listed order, then local
        values override them. Mappings merge recursively; lists/scalars replace.

    Raises:
        FileNotFoundError: A config or included file does not exist.
        ValueError: Includes form a cycle or YAML mappings/include lists are invalid.

    Notes:
        Resolves composition only; use ``load_experiment_config`` for validation.
        YAML parser errors propagate unchanged."""
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Circular config include: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping.")  # noqa: TRY004 -- configuration errors use ValueError
    includes = config.pop("includes", [])
    if not isinstance(includes, list) or any(
        not isinstance(v, str) or not v for v in includes
    ):
        raise ValueError("includes must be a list of nonempty paths.")
    merged = {}
    for include in includes:
        merged = _merge(merged, resolve_config(path.parent / include, (*stack, path)))
    return _merge(merged, config)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load, compose, normalize, and validate one experiment configuration.

    Args:
        path: Experiment YAML file with optional relative includes.

    Returns:
        Fully resolved settings, including normalized scientific-notation values.

    Raises:
        FileNotFoundError: A requested configuration file does not exist.
        ValueError: Composition or experiment settings are invalid.

    Notes:
        Validation occurs before the runner creates outputs or trains a model.
        YAML parser errors propagate unchanged."""
    config = resolve_config(path)
    validate_config(config)
    return config
