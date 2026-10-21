
import numpy as np
import pytest

from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor


@pytest.fixture
def dataset():
    generator = DataGenerator(
        n_train=840,
        n_val=140,
        n_test=420,
        seed=42,
        sampling="uniform",
        split_strategy="random",
    )
    return generator.generate()


@pytest.mark.parametrize("scaling", ["standard", "none"])
def test_processed_shapes(dataset, scaling):
    preprocessor = DataPreprocessor(scaling=scaling)
    processed = preprocessor.fit_transform_dataset(dataset)

    for split in ("train", "val", "test"):
        x, y = processed[split]
        expected_size = len(dataset[split][0])

        assert x.shape == (expected_size, 1)
        assert y.shape == (expected_size, 1)
        assert x.dtype == np.float32
        assert y.dtype == np.float32


def test_standard_scaling(dataset):
    preprocessor = DataPreprocessor(scaling="standard")
    processed = preprocessor.fit_transform_dataset(dataset)

    x_train = processed["train"][0]

    assert np.isclose(x_train.mean(), 0.0, atol=1e-6)
    assert np.isclose(x_train.std(), 1.0, atol=1e-6)


def test_scaler_fits_training_data_only(dataset):
    preprocessor = DataPreprocessor(scaling="standard")
    preprocessor.fit_transform_dataset(dataset)

    x_train = dataset["train"][0]

    np.testing.assert_allclose(
        preprocessor.mean,
        np.mean(x_train),
    )

    np.testing.assert_allclose(
        preprocessor.std,
        np.std(x_train),
    )


def test_validation_and_test_do_not_affect_scaler(dataset):
    preprocessor = DataPreprocessor(scaling="standard")
    preprocessor.fit_transform_dataset(dataset)

    original_mean = preprocessor.mean
    original_std = preprocessor.std

    # Deliberately modify held-out data.
    modified = {
        split: (x.copy(), y.copy())
        for split, (x, y) in dataset.items()
    }

    for split in ("val", "test"):
        x, y = modified[split]
        modified[split] = (x + 1000.0, y)

    another = DataPreprocessor(scaling="standard")
    another.fit_transform_dataset(modified)

    assert np.isclose(another.mean, original_mean)
    assert np.isclose(another.std, original_std)


def test_inverse_transform(dataset):
    preprocessor = DataPreprocessor(scaling="standard")
    preprocessor.fit(dataset["train"][0])

    original = dataset["test"][0]
    transformed = preprocessor.transform(original)
    recovered = preprocessor.inverse_transform(transformed)

    np.testing.assert_allclose(
        recovered.ravel(),
        original,
        atol=2e-6,
    )


def test_no_scaling_preserves_values(dataset):
    preprocessor = DataPreprocessor(scaling="none")
    processed = preprocessor.fit_transform_dataset(dataset)

    for split in ("train", "val", "test"):
        np.testing.assert_allclose(
            processed[split][0].ravel(),
            dataset[split][0],
            rtol=1e-6,
            atol=1e-6,
        )


def test_targets_are_not_scaled(dataset):
    preprocessor = DataPreprocessor(scaling="standard")
    processed = preprocessor.fit_transform_dataset(dataset)

    for split in ("train", "val", "test"):
        np.testing.assert_allclose(
            processed[split][1].ravel(),
            dataset[split][1],
            rtol=1e-6,
            atol=1e-6,
        )


def test_transform_before_fit_raises():
    preprocessor = DataPreprocessor(scaling="standard")

    with pytest.raises(RuntimeError):
        preprocessor.transform(np.array([1.0, 2.0]))


def test_invalid_scaling_method():
    with pytest.raises(ValueError):
        DataPreprocessor(scaling="invalid")


@pytest.mark.parametrize(
    "invalid_input",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([]),
    ],
)
def test_invalid_input_rejected(invalid_input):
    preprocessor = DataPreprocessor()

    with pytest.raises(ValueError):
        preprocessor.fit(invalid_input)
