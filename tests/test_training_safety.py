"""Fail before a nonfinite or mismatched result becomes a checkpoint."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from function_approximation.data.dataset import create_dataloaders
from function_approximation.models.mlp_regressor import MLPRegressor
from function_approximation.training.trainer import Trainer


def make_trainer_and_loaders(targets):
    x = np.array([-1.0, 1.0], dtype=np.float32)
    dataset = {split: (x, targets) for split in ("train", "val", "test")}
    return (
        Trainer(MLPRegressor(hidden_dims=[4]), max_epochs=2, patience=2),
        create_dataloaders(dataset, batch_size=2),
    )


def test_overflow_aborts_before_optimizer_step_and_save(tmp_path):
    trainer, loaders = make_trainer_and_loaders(
        np.array([1e30, -1e30], dtype=np.float32)
    )
    initial = deepcopy(trainer.model.state_dict())
    trainer.optimizer.step = Mock()
    with pytest.raises(FloatingPointError, match="Nonfinite training MSE"):
        trainer.fit(loaders["train"], loaders["val"])
    trainer.optimizer.step.assert_not_called()
    assert trainer.history == {"train_loss": [], "val_loss": []}
    assert trainer.best_epoch == 0
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, initial[key])
    for save in (trainer.save_checkpoint,):
        with pytest.raises(RuntimeError):
            save(tmp_path / "invalid.pt", {})
    assert not (tmp_path / "invalid.pt").exists()


@pytest.mark.parametrize(
    "train_loss, val_loss", [(float("nan"), 1.0), (1.0, float("inf"))]
)
def test_epoch_guard_catches_nonfinite_custom_losses(monkeypatch, train_loss, val_loss):
    trainer, loaders = make_trainer_and_loaders(np.array([0.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(
        trainer,
        "_run_epoch",
        lambda loader, training: train_loss if training else val_loss,
    )
    with pytest.raises(FloatingPointError, match="Nonfinite loss at epoch 1"):
        trainer.fit(loaders["train"], loaders["val"])
    assert trainer.best_epoch == 0
    assert trainer.best_epoch == 0


def test_output_shape_must_match_scalar_targets():
    _, loaders = make_trainer_and_loaders(np.array([0.0, 1.0], dtype=np.float32))
    trainer = Trainer(MLPRegressor(output_dim=2), max_epochs=1)
    with pytest.raises(ValueError, match="must match target"):
        trainer.fit(loaders["train"], loaders["val"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"min_delta": float("inf")},
        {"patience": True},
        {"max_epochs": 1.5},
    ],
)
def test_direct_trainer_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Trainer(MLPRegressor(hidden_dims=[4]), **kwargs)


def test_failed_fit_after_valid_epoch_cannot_save(tmp_path, monkeypatch):
    trainer, loaders = make_trainer_and_loaders(np.array([0.0, 1.0], dtype=np.float32))
    losses = iter([1.0, 1.0, float("nan"), 1.0])
    monkeypatch.setattr(trainer, "_run_epoch", lambda *args, **kwargs: next(losses))
    with pytest.raises(FloatingPointError):
        trainer.fit(loaders["train"], loaders["val"])
    with pytest.raises(RuntimeError, match="Train the model"):
        trainer.save_checkpoint(tmp_path / "invalid.pt", {})
    assert not (tmp_path / "invalid.pt").exists()
