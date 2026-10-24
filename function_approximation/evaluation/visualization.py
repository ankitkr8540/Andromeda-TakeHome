"""Visualization utilities for function approximation experiments."""

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import nn

from function_approximation.data.generation import DataGenerator, Dataset
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.inference import predict_values

BURGUNDY = "#4B1535"
PREDICTION = "#963263"
GROUND_TRUTH = "#969FA5"
INK = "#414141"
GRID = "#E8E8E8"
SPLIT_COLORS = {"train": BURGUNDY, "val": PREDICTION, "test": GROUND_TRUTH}


def pi_label(value: float) -> str:
    """Format a coordinate in radians as a compact multiple of π.

    Args:
        value: Scalar coordinate in original input units.

    Returns:
        Tick label such as ``0``, ``5π``, or ``−10π``."""
    scaled = value / np.pi
    return "0" if abs(scaled) < 1e-10 else f"{scaled:.4g}π".replace("-", "−")


def pi_axis(ax, low: float, high: float, ticks: int = 3) -> None:
    """Set axis limits and evenly spaced π tick labels in place.

    Args:
        ax: Matplotlib axis to update.
        low: Lower input coordinate in radians.
        high: Upper input coordinate in radians.
        ticks: Number of evenly spaced tick locations, including endpoints."""
    positions = np.linspace(low, high, ticks)
    ax.set_xticks(positions, [pi_label(value) for value in positions])
    ax.set_xlim(low, high)


def format_rmse(value: float) -> str:
    """Keep small nonzero errors visible rather than rounding them to zero."""
    return f"{value:.2e}" if 0 < abs(value) < 1e-3 else f"{value:.4f}"


def style_figure(fig) -> None:
    """Apply the same typography, light grid, and clean spines to each panel."""
    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color(INK)
            ax.spines[name].set_linewidth(1.1)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
        ax.tick_params(labelsize=12, colors=INK)
        for label in [
            ax.xaxis.label,
            ax.yaxis.label,
            ax.title,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]:
            label.set_fontfamily("DejaVu Sans")
            label.set_color(INK)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.title.set_fontsize(16)
        left_title = ax.get_title(loc="left")
        if left_title:
            ax.set_title(
                left_title, loc="left", fontsize=17, color=INK, fontfamily="DejaVu Sans"
            )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for label in legend.get_texts():
                label.set_fontfamily("DejaVu Sans")
    for text in fig.texts:
        text.set_fontfamily("DejaVu Sans")
    for legend in fig.legends:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontfamily("DejaVu Sans")


def _prediction_label(model) -> str:
    """Choose the display label from the model family, including Snake activations in an MLP."""
    kind = getattr(model, "model_type", "mlp")
    if kind == "snake" or any(
        module.__class__.__name__ == "Snake" for module in model.modules()
    ):
        return "Snake MLP prediction"
    return {"fourier": "Fourier prediction", "mlp": "MLP prediction"}.get(
        kind, f"{kind} prediction"
    )


def _save_figure(
    fig: Figure,
    save_path: str | Path | None,
) -> None:
    """Save a figure if an output path is provided."""
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")


def plot_learning_curves(
    history: dict[str, list[float]],
    best_epoch: int | None = None,
    title: str = "Learning Curves",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot epoch RMSE from stored MSE losses without modifying history.

    Training values summarize batches during parameter updates, rather than
    a fresh evaluation of the completed epoch's model.

    Args:
        history: Nonempty equal-length ``train_loss`` and ``val_loss`` MSE lists.
        best_epoch: Optional one-based selected epoch to mark.
        title: Figure title.
        save_path: Optional image filename; parent directories are created.

    Returns:
        Matplotlib figure owned by the caller. Close it after use.

    Raises:
        ValueError: Histories are incompatible or the marked epoch is out of range.
    """
    train_mse = np.asarray(history["train_loss"], dtype=float)
    val_mse = np.asarray(history["val_loss"], dtype=float)
    if train_mse.ndim != 1 or train_mse.size == 0 or train_mse.shape != val_mse.shape:
        raise ValueError("Training and validation histories must match.")
    if any(
        not np.isfinite(loss).all() or (loss < 0).any() for loss in (train_mse, val_mse)
    ):
        raise ValueError("Loss histories must be finite and nonnegative.")
    if best_epoch is not None and (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, Integral)
        or not 1 <= best_epoch <= train_mse.size
    ):
        raise ValueError("best_epoch is outside the recorded history.")
    train_loss, val_loss = np.sqrt(train_mse), np.sqrt(val_mse)

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        epochs,
        train_loss,
        color=GROUND_TRUTH,
        linewidth=2,
        label="Training RMSE (during updates)",
    )
    ax.plot(epochs, val_loss, color=PREDICTION, linewidth=1.6, label="Validation RMSE")

    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            color=BURGUNDY,
            linestyle=":",
            label=f"Best epoch: {best_epoch}",
        )

    ax.set(
        xlabel="Epoch",
        ylabel="RMSE",
        title=title,
    )
    ax.legend()
    ax.grid(alpha=0.25)

    style_figure(fig)
    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_predictions(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    dataset: Dataset,
    target_fn=None,
    title: str = "Interpolation Predictions",
    save_path: str | Path | None = None,
    n_points: int = 2000,
) -> Figure:
    """Plot model predictions and available ground-truth observations.

    Optionally accepts target_fn for a dense reference curve.
    Ground truth is used only for visualization, never inference.

    Args:
        model: Fitted scalar regression model.
        preprocessor: Its fitted training input transform.
        dataset: Original-coordinate split mapping of ``(x, y)`` arrays.
        target_fn: Optional callable for the dense visual reference only.
        title: Figure title.
        save_path: Optional output image filename.
        n_points: At least two prediction-grid locations across observed x.

    Returns:
        Figure with prediction/reference curves and a split-coverage panel.
        The caller should close it after use.
    """
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2.")

    x_all = np.concatenate([dataset[s][0].ravel() for s in ("train", "val", "test")])

    x_grid = np.linspace(x_all.min(), x_all.max(), n_points)
    y_pred = predict_values(model, preprocessor, x_grid)

    fig, (ax, coverage_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    ax.plot(
        x_grid,
        y_pred,
        label=_prediction_label(model),
        color=PREDICTION,
        linewidth=2,
        zorder=3,
    )

    if target_fn is not None:
        y_true = np.asarray(target_fn(x_grid)).ravel()
        ax.plot(
            x_grid,
            y_true,
            label="Ground truth",
            color=GROUND_TRUTH,
            linewidth=4,
            zorder=1,
        )

    for index, split in enumerate(SPLIT_COLORS):
        x, y = dataset[split]
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        # Display a subset of observations to avoid clutter.
        indices = np.linspace(
            0,
            len(x) - 1,
            min(len(x), 80),
            dtype=int,
        )

        if target_fn is None:
            ax.scatter(
                x[indices],
                y[indices],
                s=9,
                alpha=0.3,
                color=SPLIT_COLORS[split],
                label=f"{split.capitalize()} samples",
            )

        coverage_ax.scatter(
            x,
            np.full_like(x, index, dtype=float),
            s=5,
            alpha=0.5,
            color=SPLIT_COLORS[split],
        )

    ax.set(
        ylabel="f(x)",
        title=title,
    )
    if target_fn is not None:
        ax.legend(
            [ax.lines[1], ax.lines[0]],
            ["Ground truth", _prediction_label(model)],
            fontsize=11,
            ncol=2,
            loc="upper right",
        )
    else:
        ax.legend(fontsize=11, ncol=2, loc="upper right")
    ax.grid(alpha=0.25)

    coverage_ax.set_yticks([0, 1, 2])
    coverage_ax.set_yticklabels(["Train", "Validation", "Test"])
    coverage_ax.set_xlabel("Original input x")
    pi_axis(coverage_ax, float(x_all.min()), float(x_all.max()), ticks=5)
    coverage_ax.grid(axis="x", alpha=0.25)

    style_figure(fig)
    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_residuals(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    dataset: Dataset,
    title: str = "Residual Analysis",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot prediction residuals for each dataset split.

    Residual = observed target - predicted target.

    Args:
        model: Fitted scalar regression model.
        preprocessor: Its fitted training input transform.
        dataset: Original-coordinate split mapping of observations.
        title: Figure title.
        save_path: Optional output image filename.

    Returns:
        Figure with residual scatter points by split, owned by the caller.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for split, color in SPLIT_COLORS.items():
        x, y = dataset[split]
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        y_pred = predict_values(model, preprocessor, x)
        residuals = y - y_pred

        ax.scatter(
            x,
            residuals,
            s=12,
            alpha=0.55,
            color=color,
            label=split.capitalize(),
        )

    ax.axhline(
        0,
        color=GROUND_TRUTH,
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        xlabel="Original input x",
        ylabel="Residual (y true - y predicted)",
        title=title,
    )
    ax.legend()
    pi_axis(
        ax,
        float(min(np.min(x) for x, _ in dataset.values())),
        float(max(np.max(x) for x, _ in dataset.values())),
        ticks=5,
    )
    style_figure(fig)
    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_extrapolation(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    train_range: tuple[float, float],
    bands: Dataset,
    metrics: dict[str, dict[str, float]],
    target_fn=None,
    title: str = "Extrapolation",
    save_path: str | Path | None = None,
    n_points: int = 1500,
    band_ranges: dict[str, tuple[float, float]] | None = None,
) -> Figure:
    """Plot predictions on the training domain and on each evaluation band.

    Args:
        model: Fitted scalar regression model.
        preprocessor: Its fitted training input transform.
        train_range: ``(x_min, x_max)`` of the training domain.
        bands: Mapping from ``extrap_<name>`` to evaluation ``(x, y)`` samples.
        metrics: Per-split metrics; each band's RMSE is shown in its title.
        target_fn: Optional callable for the dense visual reference only.
        title: Figure title.
        save_path: Optional output image filename.
        n_points: Prediction-grid locations per panel.
        band_ranges: Optional configured (lower, upper] boundaries by split.

    Returns:
        Figure with one stacked panel per region. The caller should close it.
    """
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2.")
    in_domain_label = "In-domain predictions"
    if "test" in metrics:
        in_domain_label += (
            f" · Held-out test RMSE {format_rmse(metrics['test']['rmse'])}"
        )
    regions = [(in_domain_label, train_range, None, False)]
    for split, (x, y) in bands.items():
        low, high = (
            band_ranges[split]
            if band_ranges and split in band_ranges
            else (float(np.min(x)), float(np.max(x)))
        )
        label = f"{split.removeprefix('extrap_').capitalize()} band · RMSE {format_rmse(metrics[split]['rmse'])}"
        regions.append(
            (label, (low, high), (x, y), bool(band_ranges and split in band_ranges))
        )

    fig, axes = plt.subplots(
        len(regions), 1, figsize=(14, 2.6 * len(regions) + 1), squeeze=False
    )
    for index, (ax, (label, (low, high), samples, lower_excluded)) in enumerate(
        zip(axes[:, 0], regions, strict=True)
    ):
        x_grid = np.linspace(low, high, n_points)
        if target_fn is not None:
            ax.plot(
                x_grid,
                np.asarray(target_fn(x_grid)).ravel(),
                color=GROUND_TRUTH,
                linewidth=4,
                label="Ground truth",
                zorder=1,
            )
        elif samples is not None:
            ax.scatter(
                np.asarray(samples[0]).ravel(),
                np.asarray(samples[1]).ravel(),
                s=8,
                alpha=0.6,
                color=GROUND_TRUTH,
                label="Observed targets",
                zorder=1,
            )
        ax.plot(
            x_grid,
            predict_values(model, preprocessor, x_grid),
            color=PREDICTION,
            linewidth=2,
            linestyle="-" if index == 0 else "--",
            label=_prediction_label(model),
            zorder=3,
        )
        # Actual configured boundaries are supplied by the experiment runner.
        if index > 0:
            bracket = "(" if lower_excluded else "["
            label = f"{label}   x ∈ {bracket}{pi_label(low)}, {pi_label(high)}]"
        ax.set_title(label, loc="left", pad=12)
        ax.set_ylabel("f(x)")
        pi_axis(ax, low, high, ticks=5 if index == 0 else 3)
    legend_axis = axes[0, 0] if target_fn is not None else axes[-1, 0]
    handles, labels = legend_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.935),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.5,
    )
    fig.suptitle(title, fontsize=18, y=0.985)
    style_figure(fig)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.06, top=0.81, hspace=0.85)
    _save_figure(fig, save_path)
    return fig


def plot_data(
    generator: DataGenerator,
    dataset: Dataset | None = None,
    show_target: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot reference observations in separate in-domain and extrapolation panels.

    Args:
        generator: Configured DataGenerator supplying intervals and the reference.
        dataset: Optional split mapping in original coordinates; required entries
            are ``train``, ``val``, and ``test``. Missing configured evaluation
            bands are generated automatically.
        show_target: Whether to overlay a dense reference curve in each panel.
        figsize: Optional figure width and height in inches.

    Returns:
        Matplotlib figure and one-dimensional axes array, one axis per region.
        Uses shared plot styling and input ticks labeled in multiples of π.

    Raises:
        ValueError: Supplied observations omit a required in-domain split.

    Notes:
        Reference evaluation is for visualization only. This function neither
        fits models nor saves files; the caller saves/closes the figure."""
    dataset = generator.generate() if dataset is None else dataset
    missing = {"train", "val", "test"} - dataset.keys()
    if missing:
        raise ValueError(f"Plot dataset is missing splits: {sorted(missing)}")
    missing_bands = any(
        f"extrap_{band['name']}" not in dataset
        for band in generator.extrapolation_bands
    )
    extrapolation = generator.generate_extrapolation() if missing_bands else {}
    panels = 1 + len(generator.extrapolation_bands)
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
    intervals = [(generator.x_min, generator.x_max)] + [
        (band["x_min"], band["x_max"]) for band in generator.extrapolation_bands
    ]
    for index, (ax, (lower, upper)) in enumerate(zip(axes, intervals)):
        if show_target:
            dense = np.linspace(lower, upper, 6000)
            ax.plot(
                dense,
                generator.target_fn(dense),
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
            band = generator.extrapolation_bands[index - 1]
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
        generator.target.replace("_", " ").title(), fontsize=17, fontweight="bold"
    )
    style_figure(fig)
    return fig, axes


def plot_checkpoint_comparison(
    best_model: nn.Module,
    final_model: nn.Module,
    preprocessor: DataPreprocessor,
    dataset: Dataset,
    target_fn,
    split_strategy: str = "random",
    title: str = "Checkpoint Comparison",
    save_path: str | Path | None = None,
    n_points: int = 2000,
) -> Figure:
    """Compare checkpoints on an original-coordinate dense grid.

    The fitted training preprocessor is shared by both models. Ground truth
    is evaluated only for the reference curve, never for model inference.
    Retained for callers that compare selected and final weights during the
    incremental pipeline migration. Uses the shared presentation plot styling.
    Shaded chronological regions show observed split ranges; random splits
    overlap and therefore receive no sequential regions.

    Args:
        best_model: Model holding selected validation weights.
        final_model: Separate model holding final-epoch weights.
        preprocessor: Shared fitted training transform; never refitted here.
        dataset: Original-coordinate split mapping.
        target_fn: Label callable used only for the dense reference curve.
        split_strategy: ``"random"`` or ``"chronological"`` coverage semantics.
        title: Figure title.
        save_path: Optional output image filename.
        n_points: Prediction-grid size, at least two.

    Returns:
        Matplotlib figure owned by the caller. Neither model's weights are changed.
    """
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if split_strategy not in ("random", "chronological"):
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    x_all = np.concatenate([dataset[s][0].ravel() for s in ("train", "val", "test")])
    x_grid = np.linspace(x_all.min(), x_all.max(), n_points)
    best_predictions = predict_values(best_model, preprocessor, x_grid)
    final_predictions = predict_values(final_model, preprocessor, x_grid)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x_grid,
        np.asarray(target_fn(x_grid)).ravel(),
        color=GROUND_TRUTH,
        linewidth=4,
        label="Ground truth",
    )
    ax.plot(
        x_grid,
        best_predictions,
        label="Best-validation checkpoint",
        color=PREDICTION,
        linewidth=2,
    )
    ax.plot(
        x_grid,
        final_predictions,
        label="Final-epoch checkpoint",
        color=BURGUNDY,
        linestyle="--",
        linewidth=2,
    )

    if split_strategy == "chronological":
        for split, label, color in (
            ("train", "Train region", SPLIT_COLORS["train"]),
            ("val", "Validation region", SPLIT_COLORS["val"]),
            ("test", "Test region", SPLIT_COLORS["test"]),
        ):
            x = dataset[split][0]
            ax.axvspan(
                float(x.min()), float(x.max()), color=color, alpha=0.1, label=label
            )

    ax.set(xlabel="Original input x", ylabel="f(x)", title=title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    pi_axis(ax, float(x_all.min()), float(x_all.max()), ticks=5)
    style_figure(fig)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_error_vs_distance(
    rows: list[dict],
    labels: dict[str, str] | None = None,
    baseline_rmse: float | None = None,
    coordinate: str = "input",
    title: str = "Error versus distance",
    save_path: str | Path | None = None,
    *,
    series_styles: dict[str, tuple[str, str]] | None = None,
    baseline_variant: str | None = None,
) -> Figure:
    """Plot precomputed band RMSE against input position or extrapolation distance.

    Args:
        rows: Records containing ``variant``, positive finite ``rmse``, and
            ``center_x_pi`` for input coordinates, or
            ``distance_from_train_boundary_pi`` for boundary-relative distance.
            Coordinate values are already in multiples of π, not radians.
        labels: Optional mapping from variant identifiers to legend labels.
            Variants retain their first-seen order; their points sort by position.
        baseline_rmse: Optional positive constant reference error, evaluated by
            the caller; displayed as a dotted training-mean baseline.
        coordinate: ``input`` for band centres measured from the origin, or
            ``boundary`` for distances from the upper training boundary.
        title: Figure heading, preserved verbatim; empty string omits it.
        series_styles: Optional variant-to-(color, linestyle) overrides, so styles
            remain consistent when the caller changes the order of model runs.
        baseline_variant: Optional per-band baseline series, drawn dotted and
            annotated directly instead of occupying a legend entry.
        save_path: Optional PNG, SVG, or other Matplotlib-supported destination.
            Missing parent directories are created.

    Returns:
        Matplotlib figure with both axes logarithmic, π input labels, shared
        grey/burgundy styling, and an optional baseline. The caller closes it.

    Raises:
        ValueError: Records are empty, required fields are missing, or numeric
            values are nonpositive/nonfinite. Log scales cannot display zero.

    Notes:
        This function only renders metrics. It does not load checkpoints,
        generate labels, fit models, or select configurations. A point is the
        RMSE over an evaluation band, not the pointwise error at its centre.
        Use separate figures for negative-input evaluation bands.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if coordinate not in ("input", "boundary"):
        raise ValueError("coordinate must be 'input' or 'boundary'.")
    if not rows:
        raise ValueError("At least one error-versus-distance record is required.")
    field = (
        "center_x_pi" if coordinate == "input" else "distance_from_train_boundary_pi"
    )
    series = {}
    for row in rows:
        if not isinstance(row, dict) or not {"variant", "rmse", field} <= row.keys():
            raise ValueError(f"Each record must contain variant, rmse, and {field}.")
        name = row["variant"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each variant must be a nonempty string.")
        try:
            if isinstance(row[field], bool) or isinstance(row["rmse"], bool):
                raise TypeError("Boolean coordinates and errors are invalid.")
            position, error = float(row[field]), float(row["rmse"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Band coordinates and RMSE must be positive finite numbers."
            ) from exc
        if not np.isfinite([position, error]).all() or position <= 0 or error <= 0:
            raise ValueError(
                "Band coordinates and RMSE must be positive finite numbers."
            )
        series.setdefault(name, []).append((position, error))
    if baseline_rmse is not None and (
        isinstance(baseline_rmse, bool)
        or not isinstance(baseline_rmse, (int, float))
        or not np.isfinite(baseline_rmse)
        or baseline_rmse <= 0
    ):
        raise ValueError("baseline_rmse must be positive and finite.")

    if baseline_variant is not None and baseline_variant not in series:
        raise ValueError("baseline_variant must name a supplied series.")
    if baseline_variant is not None and baseline_rmse is not None:
        raise ValueError("Supply either a per-band baseline or a constant baseline.")

    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    defaults = (
        (GROUND_TRUTH, "-"),
        ("#BD3156", "-"),
        (PREDICTION, "--"),
        (BURGUNDY, "-"),
    )
    model_index = 0
    baseline_points = None
    for variant, points in series.items():
        values = np.asarray(sorted(points))
        if variant == baseline_variant:
            baseline_points = values
            continue
        color, linestyle = (series_styles or {}).get(
            variant, defaults[model_index % len(defaults)]
        )
        model_index += 1
        ax.plot(
            values[:, 0],
            values[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=(labels or {}).get(variant, variant),
        )
    if baseline_rmse is not None:
        ax.axhline(
            baseline_rmse, color=INK, linestyle=":", linewidth=1.5, label="_nolegend_"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    positions = [position for points in series.values() for position, _ in points]
    lower = 10 ** np.floor(np.log10(min(positions)))
    upper = 10 ** np.ceil(np.log10(max(positions)))
    ax.set_xlim(lower, upper if upper > lower else lower * 10)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: pi_label(value * np.pi))
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    if baseline_points is not None:
        # Preserve measured band errors; extend only the endpoint values to the
        # plot edges so the reference spans the same visible range as the axes.
        ax.plot(
            np.r_[lower, baseline_points[:, 0], ax.get_xlim()[1]],
            np.r_[baseline_points[0, 1], baseline_points[:, 1], baseline_points[-1, 1]],
            color=INK,
            linestyle=":",
            linewidth=1.5,
            label="_nolegend_",
        )
    reference = (
        baseline_rmse
        if baseline_rmse is not None
        else (baseline_points[0, 1] if baseline_points is not None else None)
    )
    if reference is not None:
        ax.text(
            lower * 1.1,
            reference * 0.82,
            f"predict-the-mean ({reference:.2f})",
            va="top",
            fontsize=11,
        )
    errors = [error for points in series.values() for _, error in points]
    if reference is not None:
        errors.append(reference)
    ax.set_ylim(min(1e-7, min(errors) * 0.5), max(20, max(errors) * 1.3))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    if coordinate == "input":
        widths = [
            float(row["x_max"]) - float(row["x_min"])
            for row in rows
            if "x_min" in row and "x_max" in row
        ]
        if widths and np.allclose(widths, widths[0]):
            band = pi_label(widths[0])
            description = f"Distance from origin (centre of each {band} band)"
        else:
            description = "Distance from origin (band centre)"
    else:
        description = "Distance from training boundary (band centre)"
    ax.set_xlabel(description)
    ax.set_ylabel("RMSE per band (log)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=10)
    style_figure(fig)
    _save_figure(fig, save_path)
    return fig
