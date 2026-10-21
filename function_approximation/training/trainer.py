
"""Reusable PyTorch training pipeline."""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Train and validate a regression model."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        max_epochs: int = 500,
        patience: int = 50,
        min_delta: float = 0.0,
        device: str = "cpu",
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_epochs <= 0 or patience <= 0:
            raise ValueError("max_epochs and patience must be positive.")
        if min_delta < 0:
            raise ValueError("min_delta must be nonnegative.")

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.final_model_state = None
        self.final_epoch = 0

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> float:
        self.model.train(training)

        total_loss = 0.0
        total_samples = 0

        context = (
            torch.enable_grad()
            if training
            else torch.no_grad()
        )

        with context:
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                predictions = self.model(x)
                loss = self.criterion(predictions, y)

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("DataLoader contains no samples.")

        return total_loss / total_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Train and retain both final and best-validation checkpoints."""

        # Reset state so repeated calls don't reuse earlier results.
        self.history = {"train_loss": [], "val_loss": []}
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.final_model_state = None
        self.final_epoch = 0

        best_weights = deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(
                train_loader,
                training=True,
            )

            val_loss = self._run_epoch(
                val_loader,
                training=False,
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_weights = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train RMSE (during updates): {train_loss ** 0.5:.6f} | "
                    f"Val RMSE: {val_loss ** 0.5:.6f}"
                )

            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        # Preserve final-epoch weights BEFORE restoring the best model.
        self.final_model_state = deepcopy(self.model.state_dict())
        self.final_epoch = len(self.history["train_loss"])

        # Restore best-validation weights for official evaluation.
        self.model.load_state_dict(best_weights)

        print(
            f"\nBest epoch: {self.best_epoch} | "
            f"Best validation RMSE: {self.best_val_loss ** 0.5:.6f}"
        )
        print(f"Final epoch: {self.final_epoch}")

        return self.history


    def evaluate(self, loader: DataLoader) -> float:
        """Calculate MSE without updating model parameters."""
        return self._run_epoch(loader, training=False)

    def save_checkpoint(
        self,
        path: str | Path,
        model_config: dict,
    ) -> None:
        """Save the restored best-validation model and its architecture."""
        if self.final_model_state is None:
            raise RuntimeError("Train the model before saving checkpoints.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": model_config,
                "epoch": self.best_epoch,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "checkpoint_type": "best_validation",
            },
            path,
        )

    def save_final_checkpoint(
        self,
        path: str | Path,
        model_config: dict,
    ) -> None:
        """Save final-epoch weights without changing the active model."""

        if self.final_model_state is None:
            raise RuntimeError("Train the model before saving checkpoints.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.final_model_state,
                "model_config": model_config,
                "epoch": self.final_epoch,
                "checkpoint_type": "final",
            },
            path,
        )

