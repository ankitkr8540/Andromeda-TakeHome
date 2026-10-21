"""Reusable PyTorch training pipeline."""

import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Fit a scalar regression module with Adam and validation-based selection.

    Args:
        model: PyTorch module returning the same tensor shape as its targets.
        learning_rate: Positive finite Adam learning rate for all parameters.
        max_epochs: Positive upper bound on completed training epochs.
        patience: Number of epochs without a qualifying validation improvement
            before early stopping.
        min_delta: Minimum decrease in validation MSE required both to save
            selected weights and reset patience. This is not an RMSE threshold.
        device: PyTorch device string; experiment YAML validates availability.

    Raises:
        ValueError: Learning rate, epoch budget, patience, or tolerance is invalid.

    Notes:
        The Trainer owns one optimizer and moves the supplied model to its device.
        Use a fresh model and Trainer for an independent experiment. Repeated
        :meth:`fit` calls reset diagnostics but retain model and Adam state.
        Checkpoints contain parameters and diagnostics, not optimizer state for
        resuming an interrupted optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        max_epochs: int = 500,
        patience: int = 50,
        min_delta: float = 0.0,
        device: str = "cpu",
    ) -> None:
        if (
            isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_epochs, patience)
        ):
            raise ValueError("max_epochs and patience must be positive integers.")
        if isinstance(min_delta, bool) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative.")

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

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                predictions = self.model(x)
                if predictions.shape != y.shape:
                    raise ValueError(
                        f"Prediction shape {predictions.shape} must match target shape {y.shape}."
                    )
                loss = self.criterion(predictions, y)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    phase = "training" if training else "validation"
                    raise FloatingPointError(
                        f"Nonfinite {phase} MSE; check data and learning_rate."
                    )

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss_value * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("DataLoader contains no samples.")

        return total_loss / total_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Train, snapshot final weights, and restore selected validation weights.

        Args:
            train_loader: Batches of float input/target tensors with matching
                sample counts. Predictions must match target shape exactly.
                These are the only batches used for gradient updates.
            val_loader: Held-out batches evaluated without gradient updates.
                Do not pass the final test set here.

        Returns:
            History with ``train_loss`` and ``val_loss`` lists in MSE units.
            Training loss averages predictions made during optimizer updates;
            validation loss evaluates the completed epoch's parameters.

        Raises:
            ValueError: A loader is empty or prediction/target shapes differ.
            FloatingPointError: Losses or parameters become nonfinite.
            RuntimeError: Training finishes without a valid selected checkpoint.

        Notes:
            Final parameters are copied before restoration and remain available
            through ``final_model_state``. The active model holds the selected
            weights after return. A positive ``min_delta`` means selection need
            not be the absolute minimum of every recorded validation loss.
        """

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

            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Nonfinite loss at epoch {epoch}; training aborted."
                )
            if any(
                not torch.isfinite(parameter).all()
                for parameter in self.model.parameters()
            ):
                raise FloatingPointError(
                    f"Nonfinite model weights at epoch {epoch}; training aborted."
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
                    f"Train RMSE (during updates): {train_loss**0.5:.6f} | "
                    f"Val RMSE: {val_loss**0.5:.6f}"
                )

            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if self.best_epoch == 0 or not math.isfinite(self.best_val_loss):
            raise RuntimeError(
                "Training completed without a valid best-validation checkpoint."
            )

        # Preserve final-epoch weights BEFORE restoring the best model.
        self.final_model_state = deepcopy(self.model.state_dict())
        self.final_epoch = len(self.history["train_loss"])

        # Restore best-validation weights for official evaluation.
        self.model.load_state_dict(best_weights)

        print(
            f"\nBest epoch: {self.best_epoch} | "
            f"Best validation RMSE: {self.best_val_loss**0.5:.6f}"
        )
        print(f"Final epoch: {self.final_epoch}")

        return self.history

    def evaluate(self, loader: DataLoader) -> float:
        """Compute sample-weighted MSE without updating model parameters.

        Args:
            loader: Nonempty loader yielding matching input/target batches.

        Returns:
            Mean squared error across all samples, including partial batches.

        Notes:
            This switches the model to evaluation mode and leaves it there.
            Use a square root when reporting RMSE.
        """
        return self._run_epoch(loader, training=False)

    def save_checkpoint(
        self,
        path: str | Path,
        model_config: dict,
        model_type: str | None = None,
    ) -> None:
        """Save the active model, selected epoch, architecture, and history.

        Args:
            path: Checkpoint filename; missing parent directories are created.
            model_config: Constructor settings needed to reconstruct the model,
                excluding the YAML ``type`` key and training-only settings.
            model_type: Optional family override. Defaults to the model's
                ``model_type`` attribute, or ``"mlp"`` for older callers.

        Raises:
            RuntimeError: No completed fit with a valid selection is available.

        Notes:
            Call after :meth:`fit`, which restores selected weights. The saved
            file does not include the input preprocessor; persist that separately.
        """
        if (
            self.final_model_state is None
            or self.best_epoch < 1
            or not math.isfinite(self.best_val_loss)
        ):
            raise RuntimeError("Train the model before saving checkpoints.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": model_config,
                "model_type": model_type or getattr(self.model, "model_type", "mlp"),
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
        model_type: str | None = None,
    ) -> None:
        """Save the final-epoch snapshot without changing the active model.

        Args:
            path: Checkpoint filename; parent directories are created as needed.
            model_config: Model constructor settings, excluding YAML ``type``.
            model_type: Family override, with the same fallback as
                :meth:`save_checkpoint`.

        Raises:
            RuntimeError: A completed fit and final snapshot are unavailable.

        Notes:
            This diagnostic artifact is optional for inference. Official
            predictions use the selected checkpoint and its fitted preprocessor.
        """

        if (
            self.final_model_state is None
            or self.final_epoch < 1
            or self.best_epoch < 1
            or not math.isfinite(self.best_val_loss)
        ):
            raise RuntimeError("Train the model before saving checkpoints.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.final_model_state,
                "model_config": model_config,
                "model_type": model_type or getattr(self.model, "model_type", "mlp"),
                "epoch": self.final_epoch,
                "checkpoint_type": "final",
            },
            path,
        )
