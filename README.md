# Andromeda function approximation

This project trains a reproducible PyTorch MLP to approximate a scalar black-box
function from sampled `(x, y)` pairs. The data generator samples inputs only
within `[-10π, 10π]`. The reference function is used to create labels and plots;
the trained model and inference code never call it.

The MLP baseline uses two `Linear → Tanh` hidden layers, a linear regression
output, Adam optimization, MSE training loss, validation-based checkpoint
selection, and RMSE/MAE/R² evaluation in the original target units. A learnable
Fourier regressor is also included as an enhancement for periodic signals and
extrapolation: its frequencies and sine/cosine coefficients are fitted from the
training data, not hardcoded from the target function.

## Prerequisites

- macOS, Linux, or another system supported by PyTorch.
- Python 3.11 available as `python3.11`.
- Git, if cloning the repository from a remote source.

Confirm Python is available:

```bash
python3.11 --version
```

On macOS, install it with Homebrew if the command is unavailable:

```bash
brew install python@3.11
```

Install Homebrew first if needed: <https://brew.sh>.

## Create the virtual environment

From the repository root, create and activate the project's isolated Python
environment. `--copies` avoids symlink issues when the repository is on an
external drive.

```bash
python3.11 -m venv --copies .venv
source .venv/bin/activate
```

After activation, your shell prompt starts with `(.venv)`. The `python` command
then refers to the Python interpreter inside this project environment.

## Install project dependencies

With `(.venv)` visible in the prompt, install
[uv](https://docs.astral.sh/uv/) inside the environment, then install the locked
project dependencies:

```bash
python -m pip install --upgrade pip uv
uv sync --locked --inexact
```

Every later command in this README assumes the environment remains active. When
opening a new terminal, return to the repository and activate it again:

```bash
source .venv/bin/activate
```

## Train an experiment

Two ready-to-run configurations use the same data size, model, and seed:

```bash
# Random split: interpolation across the sampled domain
uv run python -m scripts.run_experiments --config configs/mlp_random.yaml

# Chronological split: train on lower x values and test on higher x values
uv run python -m scripts.run_experiments --config configs/mlp_chronological.yaml

# Fourier enhancement: chronological split with training-only spectral initialization
uv run python -m scripts.run_experiments --config configs/fourier.yaml

# Fourier enhancement: random split
uv run python -m scripts.run_experiments --config configs/fourier_random.yaml

# Damped Fourier enhancement: chronological extrapolation of a damped signal
uv run python -m scripts.run_experiments --config configs/damped_fourier.yaml
```

Each run creates a new UTC-timestamped directory below its configured output
root, for example:

```text
outputs/mlp_random_standard_seed42/20261021T153000000000Z/
```

The runner prints this path when it finishes. Use it when examining artifacts or
running inference.

### MLflow tracking

The supplied configurations enable MLflow. Start the local tracking server in a
separate terminal from the repository root:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5001
```

Open <http://127.0.0.1:5001> to inspect parameters, per-epoch loss/RMSE, final
metrics, and uploaded artifacts. If port `5001` is already in use, either reuse
the existing server or choose another port and update `tracking_uri` in the YAML.

For a local run without MLflow, copy a configuration, set
`tracking.enabled: false`, and run that copy. Local artifacts are still written.

## Examine a completed run

Replace `RUN_DIR` with the printed output directory.

```bash
# View selected-checkpoint metrics for train, validation, and test splits
.venv/bin/python -m json.tool RUN_DIR/metrics.json

# Compare the selected checkpoint with the final epoch
.venv/bin/python -m json.tool RUN_DIR/checkpoint_comparison.json

# Open plots on macOS
open RUN_DIR/predictions.png
open RUN_DIR/learning_curves.png
open RUN_DIR/checkpoint_comparison.png
```

The main artifacts are:

| File | Purpose |
|---|---|
| `config.yaml` | Exact validated configuration used for the run. |
| `best_model.pt` | Checkpoint selected using validation MSE. |
| `final_model.pt` | Weights from the last completed epoch, retained for diagnosis. |
| `preprocessor.joblib` | Input scaler fitted only on the training split. |
| `metrics.json` | Selected-checkpoint MSE, RMSE, MAE, and R² for every split. |
| `baseline_metrics.json` | Training-mean reference metrics. |
| `checkpoint_comparison.json` | Best versus final checkpoint metrics and epochs. |
| `learning_curves.png` | Training and validation RMSE across epochs. |
| `predictions.png` | Predictions, samples, and reference curve. |
| `residuals.png` | Residuals on each split. |
| `checkpoint_comparison.png` | Dense predictions from the best and final checkpoints. |
| `predictions.npz` | Original inputs, labels, and selected-model predictions. |

Training history is stored as MSE in `history.json`; plots and console output use
RMSE, which is the square root of MSE. `min_delta` in the YAML is therefore also
measured in MSE units.

## Predict with the saved model

The inference CLI loads `best_model.pt` and its matching preprocessor. It accepts
original, unscaled x values and does not require labels or the target function.

```bash
uv run python -m function_approximation.inference \
  --run-dir RUN_DIR \
  --x -3.14 0 3.14
```

It returns JSON containing the supplied inputs and predictions. Predictions are
allowed outside the observed training interval, but their accuracy is not
guaranteed.

## Configuration

The YAML files in `configs/` control the seed, sampling, split strategy,
preprocessing, model architecture, optimizer settings, early stopping,
evaluation, and tracking. Configuration is validated before an output directory
is created.

The supplied data settings generate 7,000 training samples, 1,000 validation samples,
and 2,000 test samples over `[-10π, 10π]` with a fixed seed of 42. Standard scaling
is fitted on training inputs only. The test split is never used for scaling,
optimization, checkpoint selection, or early stopping.

`configs/damped_fourier.yaml` selects the `damped_oscillation` synthetic target
and the `damped_fourier` model. Its learned decay rates are constrained to be
nonnegative, while its frequencies, coefficients, and rates are fitted from
training data only. The rich periodic target remains the default for the MLP and
standard Fourier configurations.

## Verify the project

Run the automated tests and static checks from the repository root:

```bash
MPLBACKEND=Agg uv run python -m pytest -q
uv run ruff check function_approximation scripts tests docs/conf.py
```

Build the optional Sphinx documentation:

```bash
uv sync --locked --inexact --group docs
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html
```

The tests cover data generation, split handling, train-only preprocessing,
training safety, checkpoint restoration, inference without target calls,
configuration validation, and MLflow artifact logging.

## Limits and next experiments

The MLP is the flexible baseline: random splits measure interpolation, while the
chronological split exposes the harder extrapolation case. The Fourier model has
a useful periodic inductive bias, but can be less suitable for non-periodic,
discontinuous, or frequency-changing functions. For a final model comparison,
choose architecture and hyperparameters using validation metrics and keep the
test split for final reporting. Useful next experiments include varying network
width/depth, activation, sample density, noise level, target function, and split
seed.

## Data sampling API

`DataGenerator(sample_size=10000)` produces 7,000 training, 1,000 validation,
and 2,000 test observations on disjoint, evenly spaced interpolation grids.
The old `n_train`, `n_val`, `n_test`, `sampling`, `split_strategy`, and `seed`
constructor arguments are retired. Experiment seeds control training only.
Only `rich_periodic` is supported as a reference target.

Counts and intervals are validated for Python and YAML callers. Grid offsets
adjust deterministically when float32 rounding would merge held-out and training
inputs. Newly generated held-out coordinates can differ from historical runs;
saved artifacts are not rewritten. Scaling is fitted on training inputs only,
and conversion overflow is rejected. Plotting is delegated to the evaluation
module through the compatible `DataGenerator.plot()` method.
