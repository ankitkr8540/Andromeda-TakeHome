
"""Visualization utilities for function approximation experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

from function_approximation.data.generation import Dataset
from function_approximation.data.preprocessing import DataPreprocessor


def _save_figure(
    fig: Figure,
    save_path: str | Path | None,
) -> None:
    """Save a figure if an output path is provided."""
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def predict_values(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    x: np.ndarray,
) -> np.ndarray:
    """Generate predictions in the original input coordinate system.

    The ground-truth function is never called.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    x_processed = preprocessor.transform(x)

    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_processed).to(device)

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            predictions = model(x_tensor)
            return predictions.detach().cpu().numpy().ravel()
    finally:
        model.train(was_training)


def plot_learning_curves(
    history: dict[str, list[float]],
    best_epoch: int | None = None,
    title: str = "MLP Learning Curves",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot epoch RMSE from stored MSE losses without modifying history.

    Training values summarize batches during parameter updates, rather than
    a fresh evaluation of the completed epoch's model.
    """
    train_loss = np.sqrt(np.asarray(history["train_loss"]))
    val_loss = np.sqrt(np.asarray(history["val_loss"]))

    if len(train_loss) != len(val_loss) or len(train_loss) == 0:
        raise ValueError("Training and validation histories must match.")

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(epochs, train_loss, label="Training RMSE (during updates)")
    ax.plot(epochs, val_loss, label="Validation RMSE")

    if best_epoch is not None:
        if not 1 <= best_epoch <= len(epochs):
            raise ValueError("best_epoch is outside the recorded history.")

        ax.axvline(
            best_epoch,
            color="tab:green",
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

    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_predictions(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    dataset: Dataset,
    target_fn=None,
    title: str = "MLP Predictions",
    save_path: str | Path | None = None,
    n_points: int = 2000,
) -> Figure:
    """Plot model predictions and available ground-truth observations.

    Optionally accepts target_fn for a dense reference curve.
    Ground truth is used only for visualization, never inference.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    x_all = np.concatenate(
        [dataset[s][0].ravel() for s in ("train", "val", "test")]
    )

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
        label="MLP prediction",
        color="tab:orange",
        linewidth=2,
    )

    if target_fn is not None:
        y_true = np.asarray(target_fn(x_grid)).ravel()
        ax.plot(
            x_grid,
            y_true,
            label="Ground truth",
            color="tab:blue",
            linewidth=1.5,
            alpha=0.9,
        )

    colors = {
        "train": "tab:blue",
        "val": "tab:green",
        "test": "tab:red",
    }

    for index, split in enumerate(("train", "val", "test")):
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

        ax.scatter(
            x[indices],
            y[indices],
            s=9,
            alpha=0.3,
            color=colors[split],
            label=f"{split.capitalize()} samples",
        )

        coverage_ax.scatter(
            x,
            np.full_like(x, index, dtype=float),
            s=5,
            alpha=0.5,
            color=colors[split],
        )

    ax.set(
        ylabel="f(x)",
        title=title,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    coverage_ax.set_yticks([0, 1, 2])
    coverage_ax.set_yticklabels(["Train", "Validation", "Test"])
    coverage_ax.set_xlabel("Original input x")
    coverage_ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig


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
    Shaded chronological regions show observed split ranges; random splits
    overlap and therefore receive no sequential regions.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if split_strategy not in ("random", "chronological"):
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    x_all = np.concatenate([dataset[s][0].ravel() for s in ("train", "val", "test")])
    x_grid = np.linspace(x_all.min(), x_all.max(), n_points)
    best_predictions = predict_values(best_model, preprocessor, x_grid)
    final_predictions = predict_values(final_model, preprocessor, x_grid)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_grid, np.asarray(target_fn(x_grid)).ravel(),
            color="black", alpha=0.7, label="Ground truth")
    ax.plot(x_grid, best_predictions, label="Best-validation checkpoint",
            color="tab:blue", linewidth=2)
    ax.plot(x_grid, final_predictions, label="Final-epoch checkpoint",
            color="tab:orange", linestyle="--", linewidth=2)

    if split_strategy == "chronological":
        for split, label, color in (
            ("train", "Train region", "tab:blue"),
            ("val", "Validation region", "tab:green"),
            ("test", "Test region", "tab:red"),
        ):
            x = dataset[split][0]
            ax.axvspan(float(x.min()), float(x.max()),
                       color=color, alpha=0.1, label=label)

    ax.set(xlabel="Original input x", ylabel="f(x)", title=title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_residuals(
    model: nn.Module,
    preprocessor: DataPreprocessor,
    dataset: Dataset,
    title: str = "MLP Residual Analysis",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot prediction residuals for each dataset split.

    Residual = observed target - predicted target.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = {
        "train": "tab:blue",
        "val": "tab:green",
        "test": "tab:red",
    }

    for split in ("train", "val", "test"):
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
            color=colors[split],
            label=split.capitalize(),
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        xlabel="Original input x",
        ylabel="Residual (y true - y predicted)",
        title=title,
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_path)

    return fig
