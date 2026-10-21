# Quickstart

Run commands from the repository root. Python 3.11 and the checked-in `uv.lock`
define the supported local environment.

## Install

```bash
uv sync --locked --inexact
source .venv/bin/activate
```

If `uv` is not on PATH, use `.venv/bin/uv`. The `--inexact` option preserves
existing tools such as notebook packages. On an external drive, setting
`UV_LINK_MODE=copy` avoids hardlink warnings when the cache is on another drive.

## Start tracking

Start one server from the repository root and keep its terminal running:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5001
```

Open <http://127.0.0.1:5001>. An “address already in use” error means another
process already owns that port; reuse an existing MLflow server if it is yours.
After reconnecting an external drive, use `cd` to enter the repository again
if a terminal reports that its current directory no longer exists.

To run without a server, set `tracking.enabled: false` in a copy of the YAML.
Tracking failures propagate instead of being reported as successful uploads.

## Train

```bash
# Baseline MLP: random interpolation and chronological extrapolation
uv run python -m scripts.run_experiments --config configs/mlp_random.yaml
uv run python -m scripts.run_experiments --config configs/mlp_chronological.yaml

# Fourier model with training-only spectral initialization
uv run python -m scripts.run_experiments --config configs/fourier_random.yaml
uv run python -m scripts.run_experiments --config configs/fourier.yaml
```

The command prints the timestamped output directory. Every execution creates a
new directory and preserves previous runs. Configuration paths are relative to
the calling directory; relative output roots are resolved under the repository.
Budgets, preprocessing, and initialization can differ between configs, so compare
effective settings before treating runs as a controlled model comparison.

## Predict

Replace the path below with the directory printed by your completed run:

```bash
uv run python -m function_approximation.inference \
  --run-dir outputs/fourier_chronological_spectral_none_seed42/RUN_TIMESTAMP \
  --x -3.14 0 3.14
```

This returns JSON with original-coordinate inputs and predictions. Keep both
`best_model.pt` and `preprocessor.joblib`. The same command loads MLP or Fourier
weights; it does not generate labels or repeat spectral initialization.

```python
import numpy as np
from function_approximation.inference import load_predictor

predictor = load_predictor("outputs/YOUR_RUN/RUN_TIMESTAMP")
predictions = predictor.predict(np.array([-3.14, 0.0, 3.14]))
```

Only load artifacts you trust: joblib serialization uses Python pickle.

## Minimal Python example

This example estimates initial frequencies from a small training split and checks
the scalar prediction contract. It is also executed by the documentation's
doctest build.

```{doctest}
>>> import torch
>>> from function_approximation.data.generation import DataGenerator
>>> from function_approximation.data.preprocessing import DataPreprocessor
>>> from function_approximation.models.fourier import FourierRegressor
>>> raw = DataGenerator(n_train=48, n_val=12, n_test=12).generate()
>>> preprocessor = DataPreprocessor(scaling="none")
>>> processed = preprocessor.fit_transform_dataset(raw)
>>> model = FourierRegressor(num_frequencies=4)
>>> report = model.initialize_from_data(*processed["train"], grid_size=1024)
>>> report["training_samples"]
48
>>> tuple(model(torch.from_numpy(processed["train"][0])).shape)
(48, 1)
```

Train this initialized model with the same Trainer as the MLP:

```python
from function_approximation.data.dataset import create_dataloaders
from function_approximation.training.trainer import Trainer

loaders = create_dataloaders(processed, batch_size=16, seed=42)
trainer = Trainer(model, learning_rate=1e-3, max_epochs=300, patience=50)
history = trainer.fit(loaders["train"], loaders["val"])
```

For an independent trial, construct a fresh model and Trainer. Calling `fit`
again resets diagnostics but retains existing weights and Adam state.
