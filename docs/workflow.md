# Models, data boundaries, and checkpoints

## Data flow

1. Generate scalar observations and create nonoverlapping train/validation/test splits.
2. Fit the input preprocessor using training x only, then transform every split.
3. Optionally initialize Fourier frequencies and coefficients from processed training x/y.
4. Optimize training MSE with Adam and evaluate validation MSE after each epoch.
5. Snapshot final weights, restore the selected validation checkpoint, and report metrics.
6. Save weights and the fitted preprocessor for prediction on new inputs.

Data generation returns one-dimensional float64 arrays. Preprocessing and
DataLoaders provide float32 columns of shape `(N, 1)`. Targets retain their
original units. Model predictions and targets must have identical shapes;
the Trainer rejects broadcasting across mismatched dimensions.

| Operation | Data used to fit or select |
|---|---|
| Input mean and scale | Training x |
| Spectral frequency estimates and initial coefficients | Training x/y |
| Adam parameter updates | Training x/y |
| Checkpoint selection and early stopping | Validation |
| Final reported test metrics | Test, after fitting |

The target formula is called for label generation and visual references. It is
not called by model prediction. Fitting a periodic basis is an inductive
assumption; supplying known target frequencies instead of estimating them would
be a different procedure. Report the method as spectral initialization,
least-squares coefficient fitting, and Adam refinement.

Repeatedly inspecting test scores while developing a model can influence model
selection even when no test samples enter the optimizer. Treat such results as
exploratory. Freeze design choices using validation, and reserve an untouched
final holdout when the evaluation protocol permits one.

## MLP baseline

Hidden layers use `Linear → Tanh`; the regression output is linear. The MLP
maps preprocessed scalar inputs directly to predictions. Its performance on a
random split describes interpolation across the sampled domain. A chronological
split trains on lower x and tests on higher x, measuring extrapolation.

## Fourier regression

```{math}
\widehat{y}(s) = c + \sum_{k=1}^{K}
\left[a_k\sin(\omega_k s) + b_k\cos(\omega_k s)\right].
```

All frequencies, sine/cosine weights, and the bias are trainable. There are
`3*K + 1` scalar parameters. Frequencies are independent: the model is not
given harmonic ratios or the generating frequencies.

`s` is the model-input coordinate. With no scaling, `s = x`. With standard
scaling, `s = (x - training_mean) / training_scale`. A learned frequency in
standardized coordinates must be divided by that scale to express it in
radians per original x unit. Centering also changes sine/cosine phase.

### Spectral initialization

`FourierRegressor.initialize_from_data` searches the configured angular-frequency
interval with a Lomb–Scargle periodogram, suitable for irregular observations.
It selects a peak, refines it locally, and jointly refits the bias and all
selected sine/cosine coefficients using linear least squares. The remaining
residual is searched for another component, excluding nearby candidates.

The search uses a finite grid, defaulting to 4,096 locations. It requires at
least `2*K + 1` training observations and a nonzero x span. It can stop early
when the residual is negligible; unused components start with zero amplitudes.
Constant targets initialize only the bias. Least-squares calculations use
float64 before parameters are copied to the model's dtype.

Call initialization once before training, using the same processed coordinates
as the DataLoaders. Adam then refines all parameters. This starting procedure
does not guarantee successful frequency estimation outside its search interval
or under substantial noise. Forward prediction and checkpoint loading never
repeat it.

The implementation follows the [SciPy Lomb–Scargle API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html).
The earlier exploratory frequency-search/least-squares probe is a separate
optimizer and should not be described as an Adam-only experiment.

## Loss and selection

Training and stored histories use MSE. Console output and learning curves show
RMSE, its square root. Training history accumulates losses during parameter
updates; it is not a fresh evaluation of the final weights for that epoch.

The current selection rule is:

```python
validation_mse < selected_validation_mse - min_delta
```

The same qualifying improvement resets patience. Therefore positive
`min_delta` means the selected checkpoint is not necessarily the absolute
minimum in the recorded curve. A threshold of `1e-6` MSE can retain a checkpoint
near RMSE 0.001 while later errors are smaller. Choose a smaller threshold if
finer selection is required; this setting is not expressed in RMSE units.

## Why save checkpoints?

Checkpoint files are unnecessary for individual gradient updates, but allow
prediction after the training process ends. Keep the selected model and its
training-fitted preprocessor together. Final-epoch weights are a diagnostic:
they can fit training samples better while extrapolating worse.

Checkpoint files contain parameters, reconstruction settings, and diagnostics.
They do not contain Adam state for exact training resumption. Old checkpoints
without a `model_type` field are interpreted as MLPs. A saved Fourier model uses
its learned weights directly, without labels or reinitialization.

## Metrics and comparison

Every evaluated split receives MSE, RMSE, MAE, and R-squared in original target
units. The training-mean reference predicts the same scalar on every split and
is distinct from the MLP baseline model. Its coefficients never use held-out labels.

Compare both training fit and validation error, then report final test metrics.
Use identical data, budgets, and preprocessing when claiming a controlled
architecture comparison. One seed and a noiseless periodic dataset do not
establish general superiority or identify a unique governing equation.
