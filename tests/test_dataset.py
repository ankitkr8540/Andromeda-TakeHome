import numpy as np
import pytest
import torch

from function_approximation.data.dataset import (
    FunctionDataset,
    create_dataloaders,
)
from function_approximation.data.generation import DataGenerator
from function_approximation.data.preprocessing import DataPreprocessor


@pytest.fixture
def processed_data():
    generator = DataGenerator(
        sample_size=1200,
    )

    dataset = generator.generate()

    preprocessor = DataPreprocessor(scaling="standard")
    return preprocessor.fit_transform_dataset(dataset)


def test_dataset_length(processed_data):
    x, y = processed_data["train"]
    dataset = FunctionDataset(x, y)

    assert len(dataset) == 840


def test_dataset_item_shape(processed_data):
    x, y = processed_data["train"]
    dataset = FunctionDataset(x, y)

    x_item, y_item = dataset[0]

    assert x_item.shape == (1,)
    assert y_item.shape == (1,)


def test_dataset_tensor_dtype(processed_data):
    x, y = processed_data["train"]
    dataset = FunctionDataset(x, y)

    assert dataset.x.dtype == torch.float32
    assert dataset.y.dtype == torch.float32


def test_dataset_preserves_values(processed_data):
    x, y = processed_data["train"]
    dataset = FunctionDataset(x, y)

    np.testing.assert_allclose(
        dataset.x.numpy(),
        x,
    )

    np.testing.assert_allclose(
        dataset.y.numpy(),
        y,
    )


def test_dataloader_batch_shape(processed_data):
    loaders = create_dataloaders(
        processed_data,
        batch_size=64,
        seed=42,
    )

    x_batch, y_batch = next(iter(loaders["train"]))

    assert x_batch.shape == (64, 1)
    assert y_batch.shape == (64, 1)


def test_dataloader_batch_count(processed_data):
    loaders = create_dataloaders(
        processed_data,
        batch_size=64,
    )

    assert len(loaders["train"]) == 14
    assert len(loaders["val"]) == 2
    assert len(loaders["test"]) == 4


def test_dataloader_reproducibility(processed_data):
    loaders_a = create_dataloaders(
        processed_data,
        batch_size=64,
        seed=42,
    )

    loaders_b = create_dataloaders(
        processed_data,
        batch_size=64,
        seed=42,
    )

    x_a, y_a = next(iter(loaders_a["train"]))
    x_b, y_b = next(iter(loaders_b["train"]))

    torch.testing.assert_close(x_a, x_b)
    torch.testing.assert_close(y_a, y_b)


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_order_preserved(processed_data, split):
    loaders = create_dataloaders(
        processed_data,
        batch_size=64,
    )

    x_batches = []
    y_batches = []

    for x_batch, y_batch in loaders[split]:
        x_batches.append(x_batch)
        y_batches.append(y_batch)

    x_result = torch.cat(x_batches).numpy()
    y_result = torch.cat(y_batches).numpy()

    np.testing.assert_allclose(
        x_result,
        processed_data[split][0],
    )

    np.testing.assert_allclose(
        y_result,
        processed_data[split][1],
    )


def test_invalid_batch_size(processed_data):
    with pytest.raises(ValueError):
        create_dataloaders(
            processed_data,
            batch_size=0,
        )


def test_mismatched_input_target_shapes():
    x = np.arange(10)
    y = np.arange(8)

    with pytest.raises(ValueError):
        FunctionDataset(x, y)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_nonfinite_data_rejected(invalid_value):
    x = np.array([1.0, invalid_value, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        FunctionDataset(x, y)


@pytest.mark.parametrize(
    "values", [np.array([]), np.ones((2, 2)), np.ones((2, 1, 1)), np.array([1e100])]
)
def test_invalid_scalar_dataset(values):
    with pytest.raises(ValueError):
        FunctionDataset(values, values)


@pytest.mark.parametrize("batch_size", [True, 1.5, -1])
def test_invalid_batch_size_type(processed_data, batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        create_dataloaders(processed_data, batch_size=batch_size)


def test_extrapolation_not_loaded_for_training(processed_data):
    data = {**processed_data, "extrap_far": (np.array([1000.0]), np.array([0.0]))}
    loaders = create_dataloaders(data)
    assert set(loaders) == {"train", "val", "test"}
