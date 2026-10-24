"""Trainable Snake math, matched initialization, and saved inference."""

from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from function_approximation.inference import load_predictor
from function_approximation.models.mlp_regressor import MLPRegressor
from function_approximation.models.snake_mlp import Snake
from function_approximation.training.config import load_experiment_config
from scripts.run_ablation import PROJECT_ROOT, prepare_jobs
from scripts.run_experiments import run_experiment


def test_snake_formula_input_derivative_and_positive_trainable_alpha():
    layer = Snake(3).double()
    torch.testing.assert_close(layer.alpha, torch.ones(3, dtype=torch.float64))
    x = torch.tensor([[-2.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    y = layer(x)
    torch.testing.assert_close(y, x + torch.sin(layer.alpha * x).square() / layer.alpha)
    gradient = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(gradient, 1 + torch.sin(2 * layer.alpha * x))
    assert torch.autograd.gradcheck(layer, (x,))
    y.sum().backward()
    assert layer.raw_alpha.grad is not None
    assert torch.isfinite(layer.raw_alpha.grad).all()
    assert layer.raw_alpha.grad.abs().sum() > 0
    with torch.no_grad():
        layer.raw_alpha.fill_(-1000.0)
    assert (layer.alpha > 0).all()
    assert torch.isfinite(layer(x)).all()


def test_parameter_counts_and_identical_linear_initialization():
    torch.manual_seed(42)
    tanh = MLPRegressor(hidden_dims=[64, 256, 128])
    torch.manual_seed(42)
    snake = MLPRegressor(hidden_dims=[64, 256, 128], activation="snake")
    assert sum(p.numel() for p in tanh.parameters()) == 49793
    assert sum(p.numel() for p in snake.parameters()) == 50241
    assert sum(isinstance(layer, Snake) for layer in snake.network) == 3
    for a, b in zip(
        [l for l in tanh.network if isinstance(l, nn.Linear)],
        [l for l in snake.network if isinstance(l, nn.Linear)],
        strict=True,
    ):
        assert torch.equal(a.weight, b.weight)
        assert torch.equal(a.bias, b.bias)
    assert isinstance(snake.network[-1], nn.Linear)


def test_small_snake_experiment_reload_and_paired_settings(tmp_path):
    _, jobs = prepare_jobs(
        PROJECT_ROOT / "configs/ablations/snake/activation_comparison.yaml",
        tmp_path / "study",
        max_epochs=2,
    )
    configs = [load_experiment_config(j["config_path"]) for j in jobs]
    for section in ["data", "preprocessing", "training"]:
        assert configs[0][section] == configs[1][section]
    config = configs[1]
    assert config["model"]["activation"] == "snake"
    config["data"]["sample_size"] = 40
    config["model"]["hidden_dims"] = [4, 8]
    config["evaluation"]["generate_plots"] = False
    config["training"]["snake_learning_rate"] = "1e-3"
    path = tmp_path / "snake.yaml"
    path.write_text(yaml.safe_dump(config))
    run = run_experiment(path)
    predictor = load_predictor(run)
    arrays = np.load(Path(run) / "data" / "predictions.npz")
    np.testing.assert_allclose(predictor.predict(arrays["test_x"]), arrays["test_pred"])
    alpha = [
        layer.alpha for layer in predictor.model.network if isinstance(layer, Snake)
    ]
    assert len(alpha) == 2
    assert all((a > 0).all() for a in alpha)
    assert any(not torch.allclose(a, torch.ones_like(a)) for a in alpha)


def test_separate_alpha_learning_rate_groups_and_updates():
    from copy import deepcopy

    from function_approximation.training.trainer import Trainer

    model = MLPRegressor(hidden_dims=[4], activation="snake")
    reference = deepcopy(model)
    fast = Trainer(model, learning_rate=0.0001, snake_learning_rate=0.01)
    slow = Trainer(reference, learning_rate=0.0001)
    groups = fast.optimizer.param_groups
    assert [g["lr"] for g in groups] == [0.0001, 0.01]
    alpha_ids = {id(m.raw_alpha) for m in model.modules() if isinstance(m, Snake)}
    assert {id(p) for p in groups[1]["params"]} == alpha_ids
    parameters = [p for group in groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    initial = model.network[1].raw_alpha.detach().clone()
    x = torch.tensor([[0.2], [0.7], [1.3]])
    for trainer in [fast, slow]:
        trainer.optimizer.zero_grad()
        trainer.model(x).square().mean().backward()
        trainer.optimizer.step()
    fast_change = (model.network[1].raw_alpha - initial).abs().sum()
    slow_change = (reference.network[1].raw_alpha - initial).abs().sum()
    assert fast_change > 50 * slow_change
    assert (model.network[1].alpha > 0).all()
    torch.testing.assert_close(model.network[0].weight, reference.network[0].weight)


def test_separate_alpha_lr_config_and_validation(tmp_path):
    import pytest

    config = load_experiment_config(
        PROJECT_ROOT / "configs/experiments/mlp_rich_periodic.yaml"
    )
    config["model"]["activation"] = "snake"
    config["training"]["snake_learning_rate"] = 0.001
    valid_path = tmp_path / "valid.yaml"
    valid_path.write_text(yaml.safe_dump(config))
    assert (
        load_experiment_config(valid_path)["training"]["snake_learning_rate"] == 0.001
    )
    config["model"]["activation"] = "tanh"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="snake_learning_rate"):
        load_experiment_config(path)
