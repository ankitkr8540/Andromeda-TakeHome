import numpy as np
import pytest

from function_approximation.data.generation import DataGenerator


@pytest.fixture
def generator():
    return DataGenerator(
        sample_size=1200,
    )


def test_dataset_shapes(generator):
    data = generator.generate()

    assert data["train"][0].shape == (840,)
    assert data["val"][0].shape == (120,)
    assert data["test"][0].shape == (240,)


def test_reproducibility(generator):
    first = generator.generate()
    second = generator.generate()

    for split in ("train", "val", "test"):
        for a, b in zip(first[split], second[split]):
            np.testing.assert_array_equal(a, b)


def test_input_bounds():
    generator = DataGenerator()
    data = generator.generate()

    for x, y in data.values():
        assert np.all(x >= generator.x_min)
        assert np.all(x <= generator.x_max)
        assert np.isfinite(y).all()


BANDS = [
    {"name": "near", "x_min": 10 * np.pi, "x_max": 20 * np.pi, "n": 50},
    {"name": "far", "x_min": -200 * np.pi, "x_max": -190 * np.pi, "n": 30},
]


def test_extrapolation_bands_are_reproducible_and_bounded():
    generator = DataGenerator(extrapolation_bands=BANDS)
    first = generator.generate_extrapolation()
    second = generator.generate_extrapolation()

    assert list(first) == ["extrap_near", "extrap_far"]
    for band in BANDS:
        x, y = first[f"extrap_{band['name']}"]
        assert x.shape == y.shape == (band["n"],)
        assert np.all((x >= band["x_min"]) & (x <= band["x_max"]))
        np.testing.assert_array_equal(y, generator.target_fn(x))
        np.testing.assert_array_equal(x, second[f"extrap_{band['name']}"][0])


def test_extrapolation_bands_do_not_change_fitting_splits():
    plain = DataGenerator().generate()
    with_bands = DataGenerator(extrapolation_bands=BANDS)
    with_bands.generate_extrapolation()

    for split in ("train", "val", "test"):
        for a, b in zip(plain[split], with_bands.generate()[split]):
            np.testing.assert_array_equal(a, b)


def test_periodic_target_contract():
    x = np.linspace(-100, 100, 5000)
    periodic = DataGenerator(target="rich_periodic").target_fn
    np.testing.assert_allclose(periodic(x + 2 * np.pi), periodic(x), atol=1e-12)
    assert np.max(np.abs(periodic(x))) <= 2.3


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="Unsupported target"):
        DataGenerator(target="nope")


@pytest.mark.parametrize(
    "setting",
    [
        "seed",
        "sampling",
        "evaluation_sampling",
        "split_strategy",
        "target_fn",
        "n_train",
        "n_val",
        "n_test",
    ],
)
def test_removed_options_are_rejected(setting):
    with pytest.raises(TypeError, match=setting):
        DataGenerator(**{setting: None})


@pytest.mark.parametrize("sample_size", [10000, 60, 20, 113])
def test_all_grids_are_disjoint_and_evenly_spaced(sample_size):
    data = DataGenerator(sample_size=sample_size).generate()
    names = list(data)
    for i, name in enumerate(names):
        x = data[name][0]
        if len(x) > 2:
            np.testing.assert_allclose(
                np.diff(x), np.diff(x)[0], rtol=1e-10, atol=1e-12
            )
        for other in names[:i]:
            assert not np.intersect1d(x, data[other][0]).size
    repeated = DataGenerator(sample_size=sample_size).generate()
    for name in names:
        np.testing.assert_array_equal(data[name][0], repeated[name][0])


@pytest.mark.parametrize(
    "sample_size,expected",
    [(10000, (7000, 1000, 2000)), (23, (16, 2, 5)), (20, (14, 2, 4))],
)
def test_ratio_rounding_and_total(sample_size, expected):
    generator = DataGenerator(sample_size=sample_size)
    counts = tuple(
        len(generator.generate()[name][0]) for name in ("train", "val", "test")
    )
    assert counts == expected
    assert sum(counts) == sample_size


@pytest.mark.parametrize("sample_size", [0, 19, -1, 20.8, True, "100", None])
def test_invalid_sample_size_rejected_directly(sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        DataGenerator(sample_size=sample_size)


@pytest.mark.parametrize(
    "lower,upper",
    [(5, -5), (1, 1), (np.nan, 1), (0, np.inf), (True, 2), (-1e308, 1e308)],
)
def test_invalid_domain_rejected_directly(lower, upper):
    with pytest.raises(ValueError, match="data"):
        DataGenerator(x_min=lower, x_max=upper)


@pytest.mark.parametrize(
    "bands",
    [
        {},
        [None],
        [{"name": "near"}],
        [{"name": "Near", "x_min": 40, "x_max": 50, "n": 2}],
        [{"name": "near", "x_min": 50, "x_max": 40, "n": 2}],
        [{"name": "near", "x_min": 40, "x_max": np.inf, "n": 2}],
        [{"name": "near", "x_min": 40, "x_max": 50, "n": n} for n in [2, 2]],
        [{"name": "near", "x_min": 40, "x_max": 50, "n": 0}],
        [{"name": "near", "x_min": 40, "x_max": 50, "n": 2.5}],
        [{"name": "near", "x_min": 40, "x_max": 50, "n": True}],
    ],
)
def test_invalid_bands_rejected_directly(bands):
    with pytest.raises(ValueError, match="extrapolation_bands"):
        DataGenerator(extrapolation_bands=bands)


def test_generator_copies_band_settings():
    settings = [{"name": "near", "x_min": 40, "x_max": 50, "n": 2}]
    generator = DataGenerator(extrapolation_bands=settings)
    settings[0]["n"] = 0
    assert len(generator.generate_extrapolation()["extrap_near"][0]) == 2


@pytest.mark.parametrize("scaling", ["none", "standard"])
def test_default_grids_remain_disjoint_in_model_coordinates(scaling):
    from function_approximation.data.preprocessing import DataPreprocessor

    raw = DataGenerator().generate()
    processed = DataPreprocessor(scaling).fit_transform_dataset(raw)
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert np.intersect1d(processed[left][0], processed[right][0]).size == 0
    assert sum(len(x) for x, _ in raw.values()) == 10000
    # Labels belong to the adjusted coordinates, not to the previous grid.
    for x, y in raw.values():
        np.testing.assert_array_equal(y, DataGenerator().target_fn(x))
        np.testing.assert_allclose(np.diff(x), np.diff(x)[0], rtol=1e-10, atol=1e-12)


def test_domain_too_narrow_for_float32_rejected():
    with pytest.raises(ValueError, match="precision"):
        DataGenerator(sample_size=20, x_min=1, x_max=1 + 1e-8).generate()


def test_extrapolation_precision_loss_rejected():
    generator = DataGenerator(
        extrapolation_bands=[{"name": "far", "x_min": 100, "x_max": 100 + 1e-8, "n": 5}]
    )
    with pytest.raises(ValueError, match="precision"):
        generator.generate_extrapolation()
