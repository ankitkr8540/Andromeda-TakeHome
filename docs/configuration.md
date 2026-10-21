# Configuration and artifacts

## Validated YAML sections

Every experiment supplies `experiment`, `data`, `preprocessing`, `model`,
`training`, `evaluation`, and `tracking` mappings. Validation happens before
an output directory is created. Scientific notation such as `1e-3` is
normalized in supported real-valued fields.

| Section | Main settings | Contract |
|---|---|---|
| `experiment` | `name`, `run_name`, `output_dir`, `seed` | Nonempty names/paths; seed in NumPy's supported unsigned 32-bit range. |
| `data` | `n_train`, `n_val`, `n_test`, `x_min`, `x_max`, `sampling`, `split_strategy`, `target` | At least two samples per split; finite ordered boundaries; `uniform` or `grid`; `random` or `chronological`; built-in `rich_periodic` or `damped_oscillation` target. |
| `preprocessing` | `scaling` | `standard` fits training statistics; `none` preserves coordinates. Targets are not scaled. |
| `model` | `type` plus family-specific constructor settings | `mlp` or `fourier`; unknown architecture settings are rejected. |
| `training` | `batch_size`, `learning_rate`, `max_epochs`, `patience`, `min_delta`, `device` | Positive counts/rate, nonnegative MSE tolerance, available CPU/CUDA/MPS device. |
| `evaluation` | `metrics`, `generate_plots`, `save_predictions` | Supported metric names and boolean switches. The runner currently calculates all four metrics regardless of the requested display list. |
| `tracking` | `enabled`, `tracking_uri`, `log_model`, `log_preprocessor`, `log_figures` | Boolean switches; tracking is optional. |

See {py:func}`function_approximation.training.config.load_experiment_config` for
the loader's path rules and failure behavior. It does not verify server availability.

## Model settings

### MLP

```yaml
model:
  type: mlp
  input_dim: 1
  hidden_dim: 64
  num_hidden_layers: 2
  output_dim: 1
```

The pipeline requires scalar input/output. Hidden layers use Tanh and the
regression output is linear.

### Fourier

```yaml
model:
  type: fourier
  num_frequencies: 8
  frequency_min: 0.1
  frequency_max: 5.0
```

### Damped Fourier

```yaml
model:
  type: damped_fourier
  num_frequencies: 8
  frequency_min: 0.1
  frequency_max: 5.0
  initial_decay: 0.02
```

The model learns one nonnegative exponential damping rate per sinusoidal
component from training samples. `initial_decay` is only a generic numerical
starting value; it is not a fixed target property. The training-derived minimum
input is saved in the checkpoint as the envelope reference coordinate.

The bounds control random initialization and spectral search, in radians per
model-input unit. They are not constraints on subsequent Adam updates.

```yaml
training:
  fourier_initialization: spectral
```

`spectral` fits the initial spectrum using processed training samples. It
requires at least `2 * num_frequencies + 1` training samples. `random` retains
the constructor's random parameters; missing this option also means random
for backward compatibility. This training-only option is invalid for an MLP.

The complete current chronological config is included directly from the repository:

```{literalinclude} ../configs/fourier.yaml
:language: yaml
```

## Tracking behavior

URI precedence is explicit YAML/argument, then `MLFLOW_TRACKING_URI`, then
`http://127.0.0.1:5001`. Only enabled tracking contacts the service. With tracking
off, training still saves all local artifacts.

`log_model` controls `.pt` uploads, `log_preprocessor` controls `.joblib`, and
`log_figures` controls `.png`. JSON and configuration artifacts are always
uploaded, including Fourier spectra and initialization reports. Uploading a
checkpoint does not create an MLflow model-registry entry.

The server's `--artifacts-destination` determines server-side storage. Keep
experiment outputs and MLflow records distinct: removing a local output directory
does not remove its tracking record, and deleting a tracking record can prevent
an active training process from completing its upload.

## Local output files

| File | Meaning |
|---|---|
| `config.yaml` | Effective validated configuration, including normalized numbers. |
| `best_model.pt` | Selected validation weights, architecture, epoch, MSE, and history. |
| `final_model.pt` | Last completed epoch's weights, saved before restoring selected weights. |
| `preprocessor.joblib` | Training-fitted input transform, required for saved inference. |
| `metrics.json` | Selected-model MSE, RMSE, MAE, and R-squared by split. |
| `baseline_metrics.json` | Training-target mean and its fixed-prediction scores. |
| `checkpoint_comparison.json` | Selected/final metrics and epochs. Loss fields use MSE. |
| `history.json` | Epoch training and validation MSE lists. |
| `checkpoint_comparison.png` | Required dense-grid comparison of both checkpoints and reference target. |
| `learning_curves.png`, `predictions.png`, `residuals.png` | Optional plots controlled by `generate_plots`. |
| `predictions.npz` | Optional selected-model predictions, original x, and targets. |
| `initialization.json` | Fourier-family initial spectrum, training error, and search diagnostics. |
| `spectrum.json` | Fourier-family selected/final components and input-coordinate transform. |

Metrics are recomputed using fixed checkpoint weights and original target units.
MLflow preserves historical `train_loss`, `val_loss`, and `best_val_loss` in MSE;
`epoch_train_rmse` and `epoch_val_rmse` are their square roots. Summary names use
`best_<split>_<metric>`, `final_<split>_<metric>`, and `baseline_<split>_<metric>`.
Unprefixed `<split>_<metric>` summaries refer to the selected model.
