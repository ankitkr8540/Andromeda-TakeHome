"""Reproducible synthetic data generation for function approximation."""

from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from function_approximation.data.dataset import rich_periodic

Array: TypeAlias = NDArray[np.float64]
Dataset: TypeAlias = dict[str, tuple[Array, Array]]


def split_counts(sample_size: int) -> tuple[int, int, int]:
    """Allocate 70:10:20, rounding train/validation down and keeping the remainder."""
    n_train = 7 * int(sample_size) // 10
    n_val = int(sample_size) // 10
    return n_train, n_val, int(sample_size) - n_train - n_val


class DataGenerator:
    """Generate and visualize synthetic function-approximation datasets.

    Supports disjoint interpolation grids and evaluation-only extrapolation bands.

    The ground-truth function is used only to generate labels and
    optional exploratory visualizations. Models must learn exclusively
    from the generated training examples.

    Args:
        sample_size: Total in-domain samples, split 70:10:20; at least 20.
            Experiment YAML validation enforces counts and interval settings.
        x_min: Inclusive lower sampling boundary.
        x_max: Inclusive upper training boundary.
        target: The supported name rich_periodic (kept for saved-config compatibility).
        extrapolation_bands: Optional evaluation-only intervals, each a mapping
            with ``name``, ``x_min``, ``x_max``, and sample count ``n``. They may
            lie outside ``[x_min, x_max]`` and never enter the fitting splits.

    Raises:
        ValueError: The target is unsupported.

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
        self.n_train, self.n_val, self.n_test = split_counts(sample_size)
        self.sample_size = int(sample_size)

        self.x_min = x_min
        self.x_max = x_max

        self.target = target
        self.extrapolation_bands = list(extrapolation_bands or [])

        if target != "rich_periodic":
            raise ValueError(f"Unsupported target: {target}")
        self.target_fn = rich_periodic

    def generate(self) -> Dataset:
        """Return deterministic, evenly spaced grids for all three splits.

        Training includes both boundaries. Validation and test use distinct
        irrational cell offsets to avoid shared points with each other and the
        endpoint-aligned training grid. Changing sample_size changes all three grid sizes. Seeds affect model training, not input locations.
        """
        inputs = {"train": np.linspace(self.x_min, self.x_max, self.n_train)}
        for name, count, offset in (
            ("val", self.n_val, np.sqrt(2) / 4),
            ("test", self.n_test, np.sqrt(3) / 4),
        ):
            positions = (np.arange(count, dtype=np.float64) + offset) / count
            inputs[name] = self.x_min + (self.x_max - self.x_min) * positions
        # Guard against floating-point collisions in extremely narrow domains.
        names = list(inputs)
        for index, name in enumerate(names):
            if np.unique(inputs[name]).size != inputs[name].size:
                raise ValueError(
                    "Domain precision is insufficient for the requested grid."
                )
            for other in names[:index]:
                if np.intersect1d(inputs[name], inputs[other]).size:
                    raise ValueError("Domain precision causes overlapping split grids.")
        return {name: (x, self.target_fn(x)) for name, x in inputs.items()}

    def generate_extrapolation(self) -> Dataset:
        """Return fixed evaluation grids in (lower, upper] for each band.

        Excluding the lower endpoint keeps the near band separate from the
        training boundary. These grids never enter fitting or model selection.
        """
        bands = {}
        for band in self.extrapolation_bands:
            x = np.linspace(band["x_min"], band["x_max"], band["n"] + 1)[1:]
            bands[f"extrap_{band['name']}"] = (x, self.target_fn(x))
        return bands

    def plot(
        self,
        dataset: Dataset | None = None,
        show_target: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, NDArray]:
        """Plot sampled splits and the target curve in original coordinates.

        With no dataset, generate the in-domain splits. Configured extrapolation
        bands appear in separate panels, avoiding a misleading continuous axis
        across gaps. Supplied extrapolation samples take precedence over generated
        ones. This is an exploratory ground-truth plot, not model evaluation.

        Returns:
            Figure and a one-dimensional array of axes. Call ``plt.show()`` to
            display or ``fig.savefig(...)`` to save; close the figure afterward.
        """
        # Import when plotting to avoid a data/visualization import cycle.
        from function_approximation.evaluation.visualization import (
            GROUND_TRUTH,
            SPLIT_COLORS,
            style_figure,
        )

        dataset = self.generate() if dataset is None else dataset
        missing = {"train", "val", "test"} - dataset.keys()
        if missing:
            raise ValueError(f"Plot dataset is missing splits: {sorted(missing)}")
        missing_bands = any(
            f"extrap_{band['name']}" not in dataset for band in self.extrapolation_bands
        )
        extrapolation = self.generate_extrapolation() if missing_bands else {}
        panels = 1 + len(self.extrapolation_bands)
        fig, axes = plt.subplots(
            panels,
            1,
            figsize=figsize or (13, 3.5 * panels),
            squeeze=False,
            layout="constrained",
        )
        axes = axes[:, 0]
        styles = {
            "train": ("Training", SPLIT_COLORS["train"], "o"),
            "val": ("Validation", SPLIT_COLORS["val"], "^"),
            "test": ("Interpolation test", SPLIT_COLORS["test"], "s"),
        }
        intervals = [(self.x_min, self.x_max)] + [
            (band["x_min"], band["x_max"]) for band in self.extrapolation_bands
        ]
        for index, (ax, (lower, upper)) in enumerate(zip(axes, intervals)):
            if show_target:
                dense = np.linspace(lower, upper, 6000)
                ax.plot(
                    dense,
                    self.target_fn(dense),
                    color=GROUND_TRUTH,
                    lw=3,
                    alpha=0.8,
                    label="Target function",
                    zorder=1,
                )
            if index == 0:
                for name, (label, color, marker) in styles.items():
                    x, y = dataset[name]
                    ax.scatter(
                        x,
                        y,
                        s=10,
                        color=color,
                        marker=marker,
                        alpha=0.5,
                        linewidths=0,
                        label=f"{label} (n={len(x):,})",
                        zorder=2,
                    )
                ax.set_title(
                    "In-domain data — training / validation / interpolation test (70:10:20)",
                    loc="left",
                )
            else:
                band = self.extrapolation_bands[index - 1]
                name = f"extrap_{band['name']}"
                x, y = dataset[name] if name in dataset else extrapolation[name]
                ax.scatter(
                    x,
                    y,
                    s=10,
                    color=GROUND_TRUTH,
                    alpha=0.5,
                    linewidths=0,
                    label=f"Evaluation only (n={len(x):,})",
                    zorder=2,
                )
                ax.set_title(f"Extrapolation — {band['name']}", loc="left")
            ticks = np.linspace(lower, upper, 5)
            ax.set_xticks(
                ticks, [f"{value / np.pi:g}π" if value else "0" for value in ticks]
            )
            ax.set(xlim=(lower, upper), xlabel="Input x", ylabel="Target value")
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)
            ax.legend(loc="best", fontsize=9, ncol=2)
        fig.suptitle(
            self.target.replace("_", " ").title(), fontsize=17, fontweight="bold"
        )
        style_figure(fig)
        return fig, axes
