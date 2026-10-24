"""Compare explicit saved model runs in successive held-out distance bands."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from function_approximation.data.dataset import rich_periodic
from function_approximation.evaluation.visualization import plot_error_vs_distance
from function_approximation.inference import load_predictor
from function_approximation.training.config import load_experiment_config


def distance_bands(train_max, max_x, width, count):
    """Yield consecutive, disjoint evaluation grids beyond the training boundary.

    Args:
        train_max: Upper training boundary and lower edge of the first band.
        max_x: Maximum evaluation input coordinate.
        width: Positive band width in input units.
        count: Positive number of samples per band.

    Yields:
        Tuples ``(lower, upper, x)`` for evenly spaced samples in
        ``(lower, upper]``. The final band is clipped to ``max_x``.

    Notes:
        Limits must be finite and ordered, with positive width and sample count.
        This helper creates coordinates only; it does not select or refit models."""
    if not all(np.isfinite(v) for v in (train_max, max_x, width)) or width <= 0:
        raise ValueError("Band limits must be finite and width must be positive.")
    if max_x <= train_max:
        raise ValueError("max_x must exceed train_max.")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    lower = train_max
    while lower < max_x - 1e-10:
        upper = min(lower + width, max_x)
        if upper <= lower:
            raise ValueError("Band width is too small to advance at these coordinates.")
        x = np.linspace(lower, upper, count + 1)[1:]
        yield lower, upper, x
        lower = upper


def evaluate_runs(
    runs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    max_pi: float = 1000.0,
    band_width_pi: float = 10.0,
    samples_per_band: int = 2000,
) -> Path:
    """Evaluate named saved runs on shared grids; export metrics, provenance, and plots.

    Args:
        runs: Unique plot labels mapped to explicit saved run directories.
        output_dir: Destination for CSV, source JSON, PNG, and SVG files.
        max_pi: Maximum input coordinate in multiples of pi.
        band_width_pi: Positive width of each evaluation band in multiples of pi.
        samples_per_band: At least two evenly spaced evaluation points per band.

    Returns:
        Absolute output directory. Source checkpoint files are never modified.

    Raises:
        ValueError: Settings are invalid or runs use different training domains,
            targets, or sample counts. Labels must be nonempty and unique.
        FileNotFoundError: Saved configuration or inference artifacts are missing.

    Notes:
        No fitting or checkpoint selection occurs. The evaluator alone calls the
        reference function. If the first run has baseline_metrics.json, its saved
        training-target mean is evaluated separately in every distance band.
    """
    if not runs or any(
        not isinstance(label, str) or not label.strip() for label in runs
    ):
        raise ValueError("Supply at least one saved run with a nonempty label.")
    if not np.isfinite(max_pi) or not np.isfinite(max_pi * np.pi):
        raise ValueError("max_pi must be finite in input units.")
    if not np.isfinite(band_width_pi) or band_width_pi <= 0:
        raise ValueError("band_width_pi must be positive and finite.")
    if (
        isinstance(samples_per_band, bool)
        or not isinstance(samples_per_band, int)
        or samples_per_band < 2
    ):
        raise ValueError("samples_per_band must be an integer >= 2.")
    loaded = {}
    sources = {}
    series_styles = {}
    expected = None
    baseline_prediction = None
    baseline_label = "Training-mean baseline"
    if baseline_label in runs:
        raise ValueError(f"Label {baseline_label!r} is reserved for the baseline.")
    for label, directory in runs.items():
        run = Path(directory).resolve()
        config = load_experiment_config(run / "config.yaml")
        data = config["data"]
        signature = (
            data.get("target", "rich_periodic"),
            data["x_min"],
            data["x_max"],
            data["sample_size"],
        )
        if expected is None:
            expected = signature
            baseline_path = Path(run) / "metrics" / "baseline_metrics.json"
            if baseline_path.is_file():
                baseline_prediction = float(
                    json.loads(baseline_path.read_text())["prediction"]
                )
                if not np.isfinite(baseline_prediction):
                    raise ValueError("Saved training-mean prediction must be finite.")
        elif signature != expected:
            raise ValueError(
                "Compared runs must share target, training domain, and sample count."
            )
        kind = config["model"]["type"]
        if kind == "snake" or config["model"].get("activation") == "snake":
            series_styles[label] = ("#BD3156", "-")
        elif kind == "fourier":
            series_styles[label] = (
                ("#963263", "--")
                if config["training"].get("fourier_fit") == "spectral_only"
                else ("#4B1535", "-")
            )
        else:
            series_styles[label] = ("#969FA5", "-")
        loaded[label] = load_predictor(run)
        sources[label] = {
            "run_dir": str(run),
            "model_type": config["model"]["type"],
            "seed": config["experiment"]["seed"],
        }
    train_max = expected[2]
    if max_pi * np.pi <= train_max:
        raise ValueError("max_pi must extend beyond the training boundary.")
    rows = []
    for lower, upper, x in distance_bands(
        train_max, max_pi * np.pi, band_width_pi * np.pi, samples_per_band
    ):
        truth = rich_periodic(x)
        predictions = {
            label: predictor.predict(x) for label, predictor in loaded.items()
        }
        if baseline_prediction is not None:
            predictions[baseline_label] = np.full(x.shape, baseline_prediction)
        for label, prediction in predictions.items():
            residual = np.asarray(prediction, dtype=np.float64).ravel() - truth
            rows.append(
                {
                    "variant": label,
                    "x_min": lower,
                    "x_max": upper,
                    "center_x_pi": (lower + upper) / (2 * np.pi),
                    "distance_from_train_boundary_pi": ((lower + upper) / 2 - train_max)
                    / np.pi,
                    "n": len(x),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                }
            )
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig = plot_error_vs_distance(
        rows,
        coordinate="input",
        title="",
        series_styles=series_styles,
        baseline_variant=baseline_label if baseline_prediction is not None else None,
        save_path=out / "rmse_vs_distance.png",
    )
    try:
        fig.savefig(
            out / "rmse_vs_distance.svg", facecolor="white", bbox_inches="tight"
        )
    finally:
        plt.close(fig)
    with (out / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "source.json").write_text(
        json.dumps(
            {
                "checkpoints": sources,
                "selection": "Explicit saved checkpoints; no refitting or reselection. Evaluation only.",
                "precision": "Saved float32 inference; error includes model mismatch and numerical precision.",
                "training_domain": [expected[1], train_max],
                "baseline_prediction": baseline_prediction,
                "baseline_source": str(
                    Path(Path(next(iter(runs.values()))).resolve())
                    / "metrics"
                    / "baseline_metrics.json"
                )
                if baseline_prediction is not None
                else None,
                "band_width_pi": band_width_pi,
                "samples_per_band": samples_per_band,
                "max_x_pi": max_pi,
                "plot_coordinate": "Input x in π units at each band centre",
                "plot_axes": "Both x and RMSE use logarithmic scales",
            },
            indent=2,
            allow_nan=False,
        )
    )
    return out


def main() -> None:
    """Compare --run LABEL PATH pairs without inferring variants or seeds from filenames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Repeat for each saved model to compare.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pi", type=float, default=1000.0)
    parser.add_argument("--band-width-pi", type=float, default=10.0)
    parser.add_argument("--samples-per-band", type=int, default=2000)
    args = parser.parse_args()
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("Each --run label must be unique.")
    torch.set_num_threads(1)
    print(
        evaluate_runs(
            dict(args.run),
            args.output_dir,
            max_pi=args.max_pi,
            band_width_pi=args.band_width_pi,
            samples_per_band=args.samples_per_band,
        )
    )


if __name__ == "__main__":
    main()
