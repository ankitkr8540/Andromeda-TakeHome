"""Plot behavior independent of the MLP architecture."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor
from function_approximation.evaluation.visualization import (
    plot_extrapolation,
    plot_learning_curves,
    plot_predictions,
    plot_residuals,
)
from function_approximation.inference import predict_values
from function_approximation.models.fourier import FourierRegressor


@pytest.mark.parametrize(
    "history,epoch",
    [
        ({"train_loss": [-1.0], "val_loss": [1.0]}, None),
        ({"train_loss": [float("nan")], "val_loss": [1.0]}, None),
        ({"train_loss": [], "val_loss": []}, None),
        ({"train_loss": [1.0], "val_loss": [1.0]}, 2),
    ],
)
def test_invalid_history_leaves_no_open_figure(history, epoch):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_learning_curves(history, best_epoch=epoch)
    assert plt.get_fignums() == before


def test_plots_use_model_predictions_and_separate_regions(tmp_path):
    generator = DataGenerator(
        sample_size=100,
        extrapolation_bands=[
            {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 20},
            {"name": "far", "x_min": 50 * np.pi, "x_max": 60 * np.pi, "n": 20},
        ],
    )
    data = generator.generate()
    preprocessor = DataPreprocessor().fit(data["train"][0])
    model = FourierRegressor(num_frequencies=2)
    bands = generator.generate_extrapolation()
    metrics = {name: {"rmse": 1.0} for name in bands}
    figures = []
    try:
        prediction = plot_predictions(
            model,
            preprocessor,
            data,
            generator.target_fn,
            save_path=tmp_path / "plots/predictions.png",
        )
        figures.append(prediction)
        line = prediction.axes[0].lines[0]
        np.testing.assert_allclose(
            line.get_ydata(), predict_values(model, preprocessor, line.get_xdata())
        )
        figures.append(plot_residuals(model, preprocessor, data))
        extrapolation = plot_extrapolation(
            model,
            preprocessor,
            (-10 * np.pi, 10 * np.pi),
            bands,
            metrics,
            generator.target_fn,
        )
        figures.append(extrapolation)
        assert len(extrapolation.axes) == 3
        assert model.training
        assert (tmp_path / "plots/predictions.png").is_file()
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("n_points", [1, True, 2.5])
def test_extrapolation_invalid_grid_does_not_create_figure(n_points):
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="n_points"):
        plot_extrapolation(None, None, (-1, 1), {}, {}, n_points=n_points)
    assert plt.get_fignums() == before


def test_small_rmse_labels_and_configured_band_boundaries():
    generator = DataGenerator(
        sample_size=40,
        extrapolation_bands=[
            {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 20}
        ],
    )
    data = generator.generate()
    preprocessor = DataPreprocessor().fit(data["train"][0])
    model = FourierRegressor(num_frequencies=2)
    fig = plot_extrapolation(
        model,
        preprocessor,
        (-10 * np.pi, 10 * np.pi),
        generator.generate_extrapolation(),
        {"test": {"rmse": 1e-6}, "extrap_near": {"rmse": 2e-6}},
        generator.target_fn,
        title="Original run header",
        band_ranges={"extrap_near": (10 * np.pi, 20 * np.pi)},
    )
    try:
        assert "1.00e-06" in fig.axes[0].get_title(loc="left")
        assert "2.00e-06" in fig.axes[1].get_title(loc="left")
        assert "(10π, 20π]" in fig.axes[1].get_title(loc="left")
        assert fig.texts[0].get_text() == "Original run header"
        assert fig.axes[1].get_xlim() == (10 * np.pi, 20 * np.pi)
        line = fig.axes[1].lines[-1]
        np.testing.assert_allclose(
            line.get_ydata(), predict_values(model, preprocessor, line.get_xdata())
        )
    finally:
        plt.close(fig)


def test_data_plot_wrapper_preserves_panels_and_pi_ticks():
    from function_approximation.evaluation.visualization import plot_data

    generator = DataGenerator(
        sample_size=40,
        extrapolation_bands=[
            {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 20}
        ],
    )
    for render in (generator.plot, lambda **kwargs: plot_data(generator, **kwargs)):
        fig, axes = render(show_target=False)
        try:
            assert len(axes) == 2
            assert all(not ax.lines for ax in axes)
            assert "π" in axes[0].get_xticklabels()[0].get_text()
            assert "Extrapolation" in axes[1].get_title(loc="left")
        finally:
            plt.close(fig)


def test_extrapolation_matches_presentation_palette_and_line_styles(tmp_path):
    from matplotlib.colors import to_hex

    from function_approximation.evaluation.visualization import (
        GROUND_TRUTH,
        PREDICTION,
    )

    generator = DataGenerator(
        sample_size=40,
        extrapolation_bands=[
            {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 20},
            {"name": "far", "x_min": 50 * np.pi, "x_max": 60 * np.pi, "n": 20},
        ],
    )
    data = generator.generate()
    preprocessor = DataPreprocessor("none").fit(data["train"][0])
    model = FourierRegressor(num_frequencies=2)
    metrics = {name: {"rmse": 1e-6} for name in ["test", "extrap_near", "extrap_far"]}
    destination = tmp_path / "plots" / "presentation.png"
    fig = plot_extrapolation(
        model,
        preprocessor,
        (generator.x_min, generator.x_max),
        generator.generate_extrapolation(),
        metrics,
        generator.target_fn,
        title="Original run header",
        save_path=destination,
        band_ranges={
            f"extrap_{b['name']}": (b["x_min"], b["x_max"])
            for b in generator.extrapolation_bands
        },
    )
    try:
        assert GROUND_TRUTH == "#969FA5"
        assert PREDICTION == "#963263"
        assert to_hex(fig.get_facecolor()) == "#ffffff"
        assert fig._suptitle.get_text() == "Original run header"
        assert len(fig.legends) == 1
        assert not fig.legends[0].get_frame_on()
        for index, ax in enumerate(fig.axes):
            truth, prediction = ax.lines
            assert truth.get_color() == GROUND_TRUTH
            assert prediction.get_color() == PREDICTION
            assert truth.get_linewidth() > prediction.get_linewidth()
            assert prediction.get_linestyle() == ("-" if index == 0 else "--")
            assert not ax.spines["top"].get_visible()
            assert not ax.spines["right"].get_visible()
            assert "π" in ax.get_xticklabels()[0].get_text()
            # Dense reference lines replace the old band-sample marker clutter.
            assert not ax.collections
            assert "1.00e-06" in ax.get_title(loc="left")
        assert destination.is_file()
    finally:
        plt.close(fig)


def test_learning_curve_styles_preserve_epoch_axis():
    from function_approximation.evaluation.visualization import (
        BURGUNDY,
        GROUND_TRUTH,
        PREDICTION,
    )

    fig = plot_learning_curves(
        {"train_loss": [4.0, 1.0], "val_loss": [9.0, 4.0]},
        best_epoch=2,
    )
    try:
        ax = fig.axes[0]
        assert [line.get_color() for line in ax.lines] == [
            GROUND_TRUTH,
            PREDICTION,
            BURGUNDY,
        ]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.0, 1.0])
        assert ax.get_xlabel() == "Epoch"
        assert "π" not in "".join(label.get_text() for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_legacy_checkpoint_comparison_keeps_import_and_ppt_style():
    from function_approximation.evaluation.visualization import (
        GROUND_TRUTH,
        PREDICTION,
        plot_checkpoint_comparison,
    )

    generator = DataGenerator(sample_size=40)
    data = generator.generate()
    preprocessor = DataPreprocessor("none").fit(data["train"][0])
    fig = plot_checkpoint_comparison(
        FourierRegressor(num_frequencies=2),
        FourierRegressor(num_frequencies=2),
        preprocessor,
        data,
        generator.target_fn,
        n_points=100,
    )
    try:
        ax = fig.axes[0]
        assert ax.lines[0].get_color() == GROUND_TRUTH
        assert ax.lines[1].get_color() == PREDICTION
        assert "π" in ax.get_xticklabels()[0].get_text()
    finally:
        plt.close(fig)


def test_distance_plot_sorts_records_and_preserves_ppt_style(tmp_path):
    from function_approximation.evaluation.visualization import plot_error_vs_distance

    rows = [
        {"variant": "spectral", "center_x_pi": 55, "rmse": 0.15},
        {"variant": "spectral", "center_x_pi": 15, "rmse": 0.05},
        {"variant": "adam", "center_x_pi": 15, "rmse": 3e-6},
        {"variant": "adam", "center_x_pi": 55, "rmse": 1e-5},
    ]
    before = [dict(row) for row in rows]
    path = tmp_path / "plots/distance.svg"
    fig = plot_error_vs_distance(
        rows,
        labels={"spectral": "Spectral only", "adam": "Spectral + Adam"},
        baseline_rmse=0.8619,
        title="Error versus distance",
        save_path=path,
    )
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == ax.get_yscale() == "log"
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [15, 55])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.05, 0.15])
        assert ax.lines[0].get_color() == "#969FA5"
        assert ax.lines[1].get_color() == "#4B1535"
        assert ax.lines[-1].get_label() == "Training-mean baseline (0.8619)"
        assert "π" in ax.xaxis.get_major_formatter()(20, 0)
        assert "Input x" in ax.get_xlabel()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert path.is_file()
        assert rows == before
    finally:
        plt.close(fig)


def test_distance_plot_can_show_distance_from_training_boundary():
    from function_approximation.evaluation.visualization import plot_error_vs_distance

    fig = plot_error_vs_distance(
        [{"variant": "model", "distance_from_train_boundary_pi": 5, "rmse": 0.1}],
        coordinate="boundary",
    )
    try:
        assert fig.axes[0].lines[0].get_xdata().tolist() == [5]
        assert "Distance from training boundary" in fig.axes[0].get_xlabel()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "rows,kwargs",
    [
        ([], {}),
        ([{"variant": "model", "center_x_pi": 15}], {}),
        ([{"variant": "model", "center_x_pi": 15, "rmse": 0}], {}),
        ([{"variant": "model", "center_x_pi": -15, "rmse": 0.1}], {}),
        ([{"variant": "model", "center_x_pi": 15, "rmse": np.nan}], {}),
        ([{"variant": "model", "center_x_pi": 15, "rmse": 0.1}], {"baseline_rmse": 0}),
        (
            [{"variant": "model", "center_x_pi": 15, "rmse": 0.1}],
            {"coordinate": "unknown"},
        ),
    ],
)
def test_distance_plot_invalid_data_creates_no_figure(rows, kwargs):
    from function_approximation.evaluation.visualization import plot_error_vs_distance

    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_error_vs_distance(rows, **kwargs)
    assert plt.get_fignums() == before
