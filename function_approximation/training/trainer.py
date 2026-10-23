"""Reusable PyTorch training pipeline."""

import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from function_approximation.models.snake_mlp import Snake


class Trainer:
    """Fit a regression model with Adam or L-BFGS and validation-based early stopping.

    min_delta is the required improvement in validation MSE. After fit(), the
    active model holds the selected weights.
    L-BFGS uses strong-Wolfe line search and up to 20 inner iterations per epoch.
    Use full-batch loaders for L-BFGS experiments.
    Repeated fits reset diagnostics but retain optimizer state. Checkpoints are
    for inference, not resumable training; preprocessing is saved separately."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        max_epochs: int = 500,
        patience: int = 50,
        min_delta: float = 0.0,
        device: str = "cpu",
        optimizer: str = "adam",
        snake_learning_rate: float | None = None,
    ) -> None:
        """Configure a regression optimizer, stopping rule, and device.

        Args:
            model: Trainable PyTorch regression model; moved to ``device`` in place.
            learning_rate: Finite positive optimizer step size.
            max_epochs: Positive maximum number of training epochs.
            patience: Positive number of epochs without a qualifying improvement.
            min_delta: Minimum improvement in validation MSE for selection/stopping.
            device: PyTorch device string such as ``cpu`` or ``cuda``.
            optimizer: ``adam`` or ``lbfgs``; L-BFGS should use full-batch loaders.
            snake_learning_rate: Optional Adam learning rate for Snake ``raw_alpha``
                parameters; other model weights retain ``learning_rate``.

        Raises:
            ValueError: Settings are invalid or a separate Snake rate is requested
                without both a Snake activation and Adam.

        Notes:
            Construction does not fit the model. Checkpoints support inference,
            not optimizer-state resume; save preprocessing separately."""
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
        parameters = list(self.model.parameters())
        if snake_learning_rate is not None:
            if (
                isinstance(snake_learning_rate, bool)
                or not math.isfinite(snake_learning_rate)
                or snake_learning_rate <= 0
            ):
                raise ValueError("snake_learning_rate must be finite and positive.")
            alpha_parameters = [
                module.raw_alpha
                for module in self.model.modules()
                if isinstance(module, Snake)
            ]
            if not alpha_parameters or optimizer != "adam":
                raise ValueError("snake_learning_rate requires a Snake model and Adam.")
            alpha_ids = {id(parameter) for parameter in alpha_parameters}
            parameters = [
                {
                    "params": [
                        parameter
                        for parameter in parameters
                        if id(parameter) not in alpha_ids
                    ],
                    "lr": learning_rate,
                    "name": "network",
                },
                {
                    "params": alpha_parameters,
                    "lr": snake_learning_rate,
                    "name": "snake_alpha",
                },
            ]
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        elif optimizer == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=learning_rate,
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
            )
        else:
            raise ValueError("optimizer must be adam or lbfgs.")

        self._reset_tracking()

    def _reset_tracking(self) -> None:
        """Start fresh diagnostics without changing model or optimizer state."""
        self.history = {"train_loss": [], "val_loss": []}
        self.best_epoch = 0
        self._fit_complete = False
        self.best_val_loss = float("inf")

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> float:
        """Return sample-weighted batch MSE, optionally updating the model.

        Args:
            loader: Nonempty loader yielding matching input and target tensors.
            training: Whether to enable gradients and optimizer steps.

        Returns:
            Average MSE across samples. Training measures losses during updates;
            evaluation measures the fixed model and leaves it in evaluation mode.

        Raises:
            ValueError: The loader is empty or prediction and target shapes differ.
            FloatingPointError: A batch or L-BFGS line-search loss is nonfinite."""
        self.model.train(training)

        total_loss = 0.0
        total_samples = 0

        with torch.set_grad_enabled(training):
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
                    if isinstance(self.optimizer, torch.optim.LBFGS):

                        def closure(x=x, y=y):
                            """Recompute the current batch loss and gradients for an L-BFGS line-search step."""
                            self.optimizer.zero_grad()
                            candidate = self.criterion(self.model(x), y)
                            if not torch.isfinite(candidate):
                                raise FloatingPointError(
                                    "Nonfinite L-BFGS line-search loss."
                                )
                            candidate.backward()
                            return candidate

                        self.optimizer.step(closure)
                    else:
                        loss.backward()
                        self.optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss_value * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("DataLoader contains no samples.")

        return total_loss / total_samples

    def retain_initialized_model(self, val_loader: DataLoader) -> None:
        """Select a fitted initialization as an epoch-zero inference model.

        Args:
            val_loader: Validation observations used only to record initial MSE.

        Notes:
            Resets histories and records completion without optimizer updates. Used
            by spectral-only experiments; the model can subsequently be checkpointed."""
        self._reset_tracking()
        self.best_val_loss = self.evaluate(val_loader)
        self._fit_complete = True

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        include_initial: bool = False,
    ) -> dict[str, list[float]]:
        """Train on training observations and restore validation-selected weights.

        Args:
            train_loader: Training input/target batches; full-batch for L-BFGS.
            val_loader: Held-out validation batches; never supply final test data.
            include_initial: Whether initialization may be selected as epoch zero.

        Returns:
            Mapping with ``train_loss`` and ``val_loss`` lists containing epoch MSE.
            Training loss averages batches during updates; validation scores the
            completed epoch. The selected model weights are restored before return.

        Raises:
            ValueError: A loader is empty or model/target tensor shapes differ.
            FloatingPointError: Training or validation loss becomes nonfinite.

        Notes:
            ``min_delta`` governs selection and patience, so the selected loss need
            not be the absolute history minimum. A repeated fit resets diagnostics
            but preserves optimizer state. Preprocessing is the caller's responsibility."""

        self._reset_tracking()

        best_weights = None
        if include_initial:
            self.best_val_loss = self.evaluate(val_loader)
            best_weights = deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)

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

            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train RMSE (during updates): {train_loss**0.5:.6f} | "
                    f"Val RMSE: {val_loss**0.5:.6f}"
                )

            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_weights is None:
            raise RuntimeError(
                "Training completed without a valid best-validation checkpoint."
            )

        # Restore best-validation weights for official evaluation.
        self.model.load_state_dict(best_weights)
        self._fit_complete = True

        print(
            f"\nBest epoch: {self.best_epoch} | "
            f"Best validation RMSE: {self.best_val_loss**0.5:.6f}"
        )

        return self.history

    def evaluate(self, loader: DataLoader) -> float:
        """Return sample-weighted MSE without updates; leave the model in eval mode."""
        return self._run_epoch(loader, training=False)

    def save_checkpoint(
        self,
        path: str | Path,
        model_config: dict,
        model_type: str | None = None,
    ) -> None:
        """Write the validation-selected model and inference metadata to disk.

        Args:
            path: Checkpoint destination; missing parent directories are created.
            model_config: Constructor settings needed to reconstruct the architecture.
            model_type: Optional model-family override; otherwise use the model tag.

        Raises:
            RuntimeError: Neither fitting nor initialization-only selection completed.

        Notes:
            Saves selected weights, model settings, selected epoch, validation MSE,
            and histories. Does not save preprocessing or optimizer state. An
            existing destination is overwritten, so use a unique run directory."""
        if not self._fit_complete:
            raise RuntimeError("Train the model before saving checkpoints.")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": model_config,
            "model_type": model_type or getattr(self.model, "model_type", "mlp"),
            "epoch": self.best_epoch,
            "checkpoint_type": "best_validation",
            "selected_initialization": self.best_epoch == 0,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
