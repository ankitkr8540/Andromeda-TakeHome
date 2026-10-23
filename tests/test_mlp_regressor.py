"""Configured MLP depth and width must support real training."""

import pytest
import torch
from torch import nn

from function_approximation.models.mlp_regressor import MLPRegressor


@pytest.mark.parametrize("depth,width", [(1, 4), (2, 64), (4, 32)])
def test_configured_architecture_and_gradients(depth, width):
    model = MLPRegressor(hidden_dims=[width] * depth)
    linear = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    assert len(linear) == depth + 1
    assert [(layer.in_features, layer.out_features) for layer in linear] == (
        [(1, width)] + [(width, width)] * (depth - 1) + [(width, 1)]
    )
    prediction = model(torch.linspace(-2, 2, 16).reshape(-1, 1))
    assert prediction.shape == (16, 1)
    prediction.square().mean().backward()
    assert all(
        p.numel() > 0 and p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_variable_width_checkpoint_roundtrip(tmp_path):
    from function_approximation.models import build_model
    from function_approximation.training.trainer import Trainer

    architecture = {"hidden_dims": [64, 256, 128]}
    model = MLPRegressor(**architecture)
    dimensions = [
        (layer.in_features, layer.out_features)
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]
    assert dimensions == [(1, 64), (64, 256), (256, 128), (128, 1)]
    x = torch.linspace(-1, 1, 16).reshape(-1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, x.square()), batch_size=8
    )
    trainer = Trainer(model, max_epochs=1)
    trainer.fit(loader, loader)
    trainer.save_checkpoint(tmp_path / "best.pt", architecture)
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=True)
    restored = build_model(checkpoint["model_type"], checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("widths", [[], [0], [-1], [True], [1.5], 64, "64"])
def test_invalid_hidden_widths(widths):
    with pytest.raises(ValueError):
        MLPRegressor(hidden_dims=widths)


@pytest.mark.parametrize(
    "activation,expected", [("tanh", nn.Tanh), ("relu", nn.ReLU), ("silu", nn.SiLU)]
)
def test_activation_checkpoint_roundtrip(activation, expected):
    from function_approximation.models import build_model

    settings = {"hidden_dims": [4, 8], "activation": activation}
    model = build_model("mlp", settings)
    assert sum(isinstance(layer, expected) for layer in model.network) == 2
    x = torch.linspace(-1, 1, 16).reshape(-1, 1)
    model(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    restored = build_model("mlp", settings)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x))


def test_unknown_activation_rejected():
    with pytest.raises(ValueError, match="activation"):
        MLPRegressor(activation="unknown")
