
import numpy as np
import pytest

from function_approximation.data.generation import DataGenerator


@pytest.fixture
def generator():
    return DataGenerator(
        n_train=840,
        n_val=140,
        n_test=420,
        seed=42,
    )


def test_dataset_shapes(generator):
    data = generator.generate()

    assert data["train"][0].shape == (840,)
    assert data["val"][0].shape == (140,)
    assert data["test"][0].shape == (420,)


def test_reproducibility(generator):
    first = generator.generate()
    second = generator.generate()

    for split in ("train", "val", "test"):
        for a, b in zip(first[split], second[split]):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("sampling", ["uniform", "grid"])
def test_input_bounds(sampling):
    generator = DataGenerator(sampling=sampling)
    data = generator.generate()

    for x, y in data.values():
        assert np.all(x >= generator.x_min)
        assert np.all(x <= generator.x_max)
        assert np.isfinite(y).all()


def test_chronological_split():
    generator = DataGenerator(
        n_train=840,
        n_val=140,
        n_test=420,
        split_strategy="chronological",
    )

    data = generator.generate()

    x_train = data["train"][0]
    x_val = data["val"][0]
    x_test = data["test"][0]

    assert x_train.max() < x_val.min()
    assert x_val.max() < x_test.min()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        DataGenerator(n_train=0)

    with pytest.raises(ValueError):
        DataGenerator(x_min=5, x_max=-5)
