"""Reproducible synthetic data generation for function approximation."""

from __future__ import annotations

import math
import re
from numbers import Integral, Real
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from function_approximation.data.dataset import rich_periodic

if TYPE_CHECKING:
    from matplotlib.figure import Figure

Array: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
Dataset: TypeAlias = dict[str, tuple[Array, Array]]


def split_counts(sample_size: int) -> tuple[int, int, int]:
    """Allocate an in-domain sample budget using a 70:10:20 ratio.

    Args:
        sample_size: Integer total of at least 20; booleans are rejected.

    Returns:
        Tuple ``(n_train, n_val, n_test)``. Training and validation counts are
        rounded down; testing receives the remainder so the total is preserved.

    Raises:
        ValueError: The sample budget is not an integer of at least 20."""
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, Integral)
        or sample_size < 20
    ):
        raise ValueError("data.sample_size must be an integer >= 20.")
    n_train = 7 * int(sample_size) // 10
    n_val = int(sample_size) // 10
    return n_train, n_val, int(sample_size) - n_train - n_val


def _validate_interval(lower: float, upper: float, prefix: str) -> None:
    """Reject nonfinite, reversed, zero-width, or overflowing intervals with field context."""
    for name, value in (("x_min", lower), ("x_max", upper)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{prefix}.{name} must be a finite number.")
    if lower >= upper:
        raise ValueError(f"{prefix}.x_min must be less than x_max.")
    if not math.isfinite(upper - lower):
        raise ValueError(f"{prefix} interval span must be finite.")


def _validate_bands(bands: list[dict] | None) -> list[dict]:
    """Validate evaluation-band mappings and return independent copies of their settings."""
    if bands is None:
        return []
    if not isinstance(bands, list):
        raise ValueError("data.extrapolation_bands must be a list.")  # noqa: TRY004 -- public validation errors use ValueError
    names = set()
    result = []
    for index, band in enumerate(bands):
        prefix = f"data.extrapolation_bands[{index}]"
        if not isinstance(band, dict) or set(band) != {"name", "x_min", "x_max", "n"}:
            raise ValueError(f"{prefix} must define exactly name, x_min, x_max, and n.")
        name = band["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9_]+", name)
            or name in names
        ):
            raise ValueError(f"{prefix}.name must be unique lowercase [a-z0-9_].")
        names.add(name)
        _validate_interval(band["x_min"], band["x_max"], prefix)
        count = band["n"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
            raise ValueError(f"{prefix}.n must be an integer >= 2.")
        result.append(dict(band))
    return result


def _grid_representations(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compare the coordinates consumed by both raw and standardized models."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = x.astype(np.float32)
        scale = train.std()
        standardized = ((x - train.mean()) / scale).astype(np.float32)
    values = (x, raw, standardized)
    if any(not np.isfinite(v).all() or np.unique(v).size != v.size for v in values):
        raise ValueError(
            "Domain precision is insufficient for the requested grid in float32."
        )
    return values


class DataGenerator:
    """Generate and visualize synthetic function-approximation datasets.

    Supports disjoint interpolation grids and evaluation-only extrapolation bands.

    The ground-truth function is used only to generate labels and
    optional exploratory visualizations. Models must learn exclusively
    from the generated training examples.

    Args:
        sample_size: Total in-domain samples, split 70:10:20; at least 20.
            Validated for both direct Python calls and experiment YAML.
        x_min: Inclusive lower sampling boundary.
        x_max: Inclusive upper training boundary.
        target: The supported name rich_periodic (kept for saved-config compatibility).
        extrapolation_bands: Optional evaluation-only intervals, each a mapping
            with ``name``, ``x_min``, ``x_max``, and sample count ``n``. They may
            lie outside ``[x_min, x_max]`` and never enter the fitting splits.

    Raises:
        ValueError: Invalid target, counts, intervals, or band settings.

    Notes:
        Split counts are derived from sample_size. Extrapolation counts are separate.
    """

    def __init__(
        self,
        target: str = "rich_periodic",
        sample_size: int = 10000,
        x_min: float = -10.0 * np.pi,
        x_max: float = 10.0 * np.pi,
        extrapolation_bands: list[dict] | None = None,
    ) -> None:
        """Store validated sampling settings without generating observations.

        Args:
            target: Supported reference name, ``rich_periodic``. Retained for
                compatibility with saved experiment configurations.
            sample_size: Total in-domain observations, at least 20, split 70:10:20.
            x_min: Finite lower training boundary, included in the training grid.
            x_max: Finite upper training boundary, greater than ``x_min``.
            extrapolation_bands: Optional list of mappings containing exactly
                ``name``, ``x_min``, ``x_max``, and ``n``. Names must be unique
                lowercase identifiers; each band requires at least two samples.

        Raises:
            ValueError: Target, sample count, boundaries, or band settings are invalid."""
        self.n_train, self.n_val, self.n_test = split_counts(sample_size)
        self.sample_size = int(sample_size)

        _validate_interval(x_min, x_max, "data")
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        self.target = target
        self.extrapolation_bands = _validate_bands(extrapolation_bands)

        if target != "rich_periodic":
            raise ValueError(
                f"Unsupported target: {target}; data.target must be rich_periodic."
            )
        self.target_fn = rich_periodic

    def generate(self) -> Dataset:
        """Generate deterministic, disjoint train/validation/test grids and labels.

        Returns:
            Mapping from ``train``, ``val``, and ``test`` to float64 ``(x, y)``
            vectors. Counts follow ``split_counts(sample_size)``. Training includes
            both boundaries; held-out points lie inside the interval.

        Raises:
            ValueError: Available precision cannot support unique, disjoint grids.

        Notes:
            Held-out grids remain evenly spaced. Their sub-cell offsets adjust
            deterministically if float32 rounding would merge points in raw or
            standardized model coordinates. Seeds affect training, not these grids.
            Historical saved runs may therefore use different held-out coordinates."""
        inputs = {"train": np.linspace(self.x_min, self.x_max, self.n_train)}
        train = inputs["train"]
        representations = {"train": _grid_representations(train, train)}
        for name, count, initial_offset in (
            ("val", self.n_val, np.sqrt(2) / 4),
            ("test", self.n_test, np.sqrt(3) / 4),
        ):
            # Preserve even spacing and sample counts. If rounding merges splits,
            # move the entire held-out grid by a deterministic sub-cell offset.
            for attempt in range(100):
                offset = (initial_offset + attempt / 100) % 1
                positions = (np.arange(count, dtype=np.float64) + offset) / count
                x = self.x_min + (self.x_max - self.x_min) * positions
                candidate = _grid_representations(x, train)
                if all(
                    not np.intersect1d(a, b).size
                    for previous in representations.values()
                    for a, b in zip(candidate, previous, strict=True)
                ):
                    inputs[name] = x
                    representations[name] = candidate
                    break
            else:
                raise ValueError(
                    "Domain precision causes overlapping split grids in float32."
                )
        return {name: (x, self.target_fn(x)) for name, x in inputs.items()}

    def generate_extrapolation(self) -> Dataset:
        """Generate evaluation-only observations in each configured (lower, upper] band.

        Returns:
            Mapping of ``extrap_<name>`` to float64 input/label vectors, each with
            the configured sample count. Returns an empty mapping if no bands exist.

        Raises:
            ValueError: The input grid loses uniqueness or finiteness in float32.

        Notes:
            Excluding the lower endpoint separates the usual near band from the
            training boundary. These observations must not enter fitting or selection."""
        bands = {}
        for band in self.extrapolation_bands:
            x = np.linspace(band["x_min"], band["x_max"], band["n"] + 1)[1:]
            with np.errstate(over="ignore", invalid="ignore"):
                rounded = x.astype(np.float32)
            if not np.isfinite(rounded).all() or np.unique(rounded).size != x.size:
                raise ValueError(
                    "Domain precision is insufficient for the extrapolation grid in float32."
                )
            bands[f"extrap_{band['name']}"] = (x, self.target_fn(x))
        return bands

    def plot(
        self,
        dataset: Dataset | None = None,
        show_target: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Render the generated data through the shared visualization module.

        Args:
            dataset: Optional split mapping in original input/target coordinates.
                If omitted, generate the in-domain splits; missing configured
                evaluation bands are generated separately.
            show_target: Whether to overlay dense reference-function curves.
            figsize: Optional figure width and height in inches.

        Returns:
            Matplotlib figure and one-dimensional axes array, with one in-domain
            panel followed by one panel per configured extrapolation band.

        Raises:
            ValueError: Required train/validation/test entries are missing.

        Notes:
            Compatibility wrapper for ``evaluation.visualization.plot_data``.
            The caller is responsible for saving or closing the figure."""
        from function_approximation.evaluation.visualization import plot_data

        return plot_data(self, dataset, show_target, figsize)
