
"""Reproducible synthetic data generation for function approximation."""

from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

Array = NDArray[np.float64]
Dataset = dict[str, tuple[Array, Array]]

SamplingStrategy = Literal["uniform", "grid"]
SplitStrategy = Literal["random", "chronological"]
TargetName = Literal["rich_periodic", "damped_oscillation"]


class DataGenerator:
    """Generate and visualize synthetic function-approximation datasets.

    Supports random interpolation and ordered extrapolation experiments.

    The ground-truth function is used only to generate labels and
    optional exploratory visualizations. Models must learn exclusively
    from the generated training examples.

    Args:
        n_train: Positive training sample count.
        n_val: Positive validation sample count.
        n_test: Positive final evaluation sample count.
        x_min: Inclusive lower sampling boundary.
        x_max: Upper boundary; uniform draws exclude it, grid sampling includes it.
        seed: Seed for an independent NumPy random generator.
        sampling: ``"uniform"`` random inputs or an evenly spaced ``"grid"``.
        split_strategy: ``"random"`` permutation or ``"chronological"`` sorting
            before partitioning into nonoverlapping splits.
        target: Built-in synthetic target used when ``target_fn`` is omitted.
        target_fn: Optional vectorized label generator returning one finite value
            per input. It takes precedence over the named built-in target.

    Raises:
        ValueError: Counts, sampling boundaries, or strategy names are invalid.

    Notes:
        Direct generation permits one observation per split. The experiment
        configuration requires at least two because evaluation reports R-squared.
    """

    def __init__(
        self,
        n_train: int = 1000,
        n_val: int = 200,
        n_test: int = 200,
        x_min: float = -10.0 * np.pi,
        x_max: float = 10.0 * np.pi,
        seed: int = 42,
        sampling: SamplingStrategy = "uniform",
        split_strategy: SplitStrategy = "random",
        target: TargetName = "rich_periodic",
        target_fn: Callable[[Array], Array] | None = None,
    ) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        self.x_min = x_min
        self.x_max = x_max

        self.seed = seed
        self.sampling = sampling
        self.split_strategy = split_strategy
        self.target = target

        if target_fn is not None:
            self.target_fn = target_fn
        elif target == "rich_periodic":
            self.target_fn = self._target_function
        elif target == "damped_oscillation":
            self.target_fn = self._damped_oscillation
        else:
            raise ValueError(f"Unsupported target: {target}")

        self._validate_config()

    @staticmethod
    def _target_function(x: Array) -> Array:
        """Return a richer bounded periodic target for the default experiment.

        The generator knows this expression only to create labels and reference
        plots. Models receive sampled input/output pairs and are not given its
        harmonic frequencies or phases.
        """
        return (
            1.00 * np.sin(x)
            + 0.55 * np.cos(2.0 * x)
            + 0.35 * np.sin(3.0 * x + 0.40)
            + 0.20 * np.cos(5.0 * x - 0.70)
            + 0.12 * np.sin(7.0 * x)
            + 0.08 * np.cos(11.0 * x + 0.20)
        )

    @staticmethod
    def _damped_oscillation(x: Array) -> Array:
        """Return a nonstationary damped-oscillation target for experiments."""
        return np.exp(-0.1 * x) * np.sin(x) + 0.5 * np.cos(2.0 * x)

    def _validate_config(self) -> None:
        """Validate dataset configuration before generation."""

        for name, value in (
            ("n_train", self.n_train),
            ("n_val", self.n_val),
            ("n_test", self.n_test),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if not np.isfinite([self.x_min, self.x_max]).all():
            raise ValueError("Domain boundaries must be finite.")

        if self.x_min >= self.x_max:
            raise ValueError(
                "x_min must be strictly smaller than x_max."
            )

        if self.sampling not in ("uniform", "grid"):
            raise ValueError(
                f"Unsupported sampling strategy: {self.sampling}"
            )

        if self.split_strategy not in ("random", "chronological"):
            raise ValueError(
                f"Unsupported split strategy: {self.split_strategy}"
            )

    def _sample_inputs(
        self,
        rng: np.random.Generator,
    ) -> Array:
        """Sample inputs within the configured domain."""

        n_total = self.n_train + self.n_val + self.n_test

        if self.sampling == "uniform":
            x = rng.uniform(
                low=self.x_min,
                high=self.x_max,
                size=n_total,
            )

        elif self.sampling == "grid":
            x = np.linspace(
                start=self.x_min,
                stop=self.x_max,
                num=n_total,
                dtype=np.float64,
            )

        else:
            raise ValueError(
                f"Unsupported sampling strategy: {self.sampling}"
            )

        return np.asarray(x, dtype=np.float64)

    def _split_indices(
        self,
        x: Array,
        rng: np.random.Generator,
    ) -> NDArray[np.intp]:
        """Return indices for random or chronological splitting."""

        if self.split_strategy == "random":
            return rng.permutation(len(x))

        if self.split_strategy == "chronological":
            return np.argsort(x, kind="stable")

        raise ValueError(
            f"Unsupported split strategy: {self.split_strategy}"
        )

    def generate(self) -> Dataset:
        """Generate reproducible training, validation, and test sets.

        Random splitting evaluates interpolation within the domain.
        Chronological splitting evaluates extrapolation to larger x.

        Returns:
            Dictionary with ``train``, ``val``, and ``test`` entries, each an
            ``(x, y)`` pair of one-dimensional float64 arrays. Repeated calls
            with the same settings reproduce the same sample pairs.

        Raises:
            ValueError: The target callable returns an incompatible shape or
                nonfinite labels.
        """

        rng = np.random.default_rng(self.seed)

        # Step 1: Sample input values.
        x = self._sample_inputs(rng)

        # Step 2: Generate labels using the ground-truth function.
        y = np.asarray(
            self.target_fn(x),
            dtype=np.float64,
        )

        if y.shape != x.shape:
            raise ValueError(
                "Target function must return one output per input."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Target function returned non-finite values."
            )

        # Step 3: Determine the ordering for splitting.
        indices = self._split_indices(x, rng)

        x = x[indices]
        y = y[indices]

        # Step 4: Partition into non-overlapping datasets.
        train_end = self.n_train
        val_end = train_end + self.n_val

        return {
            "train": (
                x[:train_end],
                y[:train_end],
            ),
            "val": (
                x[train_end:val_end],
                y[train_end:val_end],
            ),
            "test": (
                x[val_end:],
                y[val_end:],
            ),
        }

    def plot(
        self,
        dataset: Dataset,
        show_target: bool = True,
        figsize: tuple[float, float] = (12, 5),
    ) -> tuple[Figure, Axes]:
        """Visualize existing sample splits and an optional target reference.

        Args:
            dataset: Generated split mapping; this method does not generate data.
            show_target: Whether to call the target on a dense visual reference grid.
            figsize: Matplotlib figure size in inches.

        Returns:
            Figure and axes. The caller owns the figure and should close it
            after saving or displaying it.
        """

        fig, ax = plt.subplots(figsize=figsize)

        styles = {
            "train": {
                "color": "tab:blue",
                "marker": "o",
                "s": 14,
            },
            "val": {
                "color": "tab:orange",
                "marker": "^",
                "s": 25,
            },
            "test": {
                "color": "tab:green",
                "marker": "s",
                "s": 20,
            },
        }

        # Draw ground truth first so data points remain visible.
        if show_target:
            x_dense = np.linspace(
                self.x_min,
                self.x_max,
                num=2000,
                dtype=np.float64,
            )

            y_dense = np.asarray(
                self.target_fn(x_dense),
                dtype=np.float64,
            )

            ax.plot(
                x_dense,
                y_dense,
                color="black",
                linewidth=1.5,
                alpha=0.8,
                label="Ground truth",
                zorder=1,
            )

        # Plot each split.
        for split in ("train", "val", "test"):
            x, y = dataset[split]

            ax.scatter(
                x,
                y,
                label=f"{split.capitalize()} ({len(x)})",
                alpha=0.6,
                zorder=2,
                **styles[split],
            )

        ax.set(
            title=(
                "Synthetic Dataset — "
                f"{self.split_strategy.capitalize()} Split"
            ),
            xlabel="Input (x)",
            ylabel="Output f(x)",
            xlim=(self.x_min, self.x_max),
        )

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        return fig, ax
