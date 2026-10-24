# Folder structure

Run project commands from the repository root. Python implementation, configuration,
run artifacts, and documentation have separate locations.

```text
Andromeda-TakeHome/
├── function_approximation/       # Reusable Python package
│   ├── data/                    # Reference labels, grids, tensors, preprocessing
│   ├── models/                  # Tanh/other MLPs, Snake, Fourier, model factory
│   ├── training/                # Config composition/validation, fitting, MLflow
│   ├── evaluation/              # Metrics and shared visualization functions
│   └── inference.py             # Load artifacts and predict without labels
├── configs/
│   ├── base.yaml                # Shared experiment defaults
│   ├── data/                    # Target and evaluation-band settings
│   ├── model/                   # Model-specific defaults
│   ├── experiments/             # Runnable composed experiment configs
│   └── ablations/
│       ├── mlp/                # Architecture and hyperparameter study
│       ├── snake/              # Controlled Tanh versus Snake comparison
│       └── fourier/            # Initialization, capacity, and optimization studies
├── scripts/                     # Experiment, study, evaluation, and plot commands
├── notebooks/                   # End-to-end Fourier walkthrough
├── tests/                       # Unit and integration tests
├── outputs/                     # Generated run artifacts and study summaries
├── docs/                        # Sphinx sources and generated HTML
├── README.md                    # Setup and usage (separate from the API documentation)
├── pyproject.toml               # Package metadata and dependency groups
└── uv.lock                      # Locked dependency versions
```

## Data

| Module | Responsibility |
|---|---|
| `data/dataset.py` | Define `rich_periodic`, wrap input/target tensors in `FunctionDataset`, and create reproducible loaders. |
| `data/generation.py` | Validate sample settings and generate disjoint interpolation grids and separate evaluation bands. |
| `data/preprocessing.py` | Fit input scaling on training observations only and apply the same transform during inference. |

The reference function generates labels and visual comparisons. It is not called
by model prediction. Data generation checks split overlap in raw and standardized float32 coordinates;
preprocessing rejects float32 overflow.

## Models

| Module | Responsibility |
|---|---|
| `models/mlp_regressor.py` | Build configurable hidden-layer MLPs with a linear output. |
| `models/snake_mlp.py` | Build the standalone Snake MLP. |
| `models/fourier_regressor.py` | Fit learned sinusoidal components; optionally initialize them from training data. |
| `models/__init__.py` | Construct the requested model family and reconstruct compatible saved architectures. |

## Training and inference

| Module | Responsibility |
|---|---|
| `training/config.py` | Resolve relative YAML includes and merge experiment settings. |
| `training/validation.py` | Normalize and validate settings before allocating outputs or training. |
| `training/trainer.py` | Optimize training MSE, select by validation MSE, and save the selected model. |
| `training/tracking.py` | Configure optional MLflow tracking and upload run artifacts. |
| `inference.py` | Load model weights and preprocessing and predict from original-coordinate inputs. |

## Evaluation

`evaluation/metrics.py` computes regression scores and the training-mean baseline.
`evaluation/visualization.py` owns all shared plot styles and rendering functions.
`DataGenerator.plot()` delegates to this module for compatibility.

## Command-line scripts

| Script module | Purpose |
|---|---|
| `scripts.run_experiments` | Train one or more experiment configurations. |
| `scripts.run_ablation` | Run or resume a configured MLP, Snake, or Fourier study and summarize its results. |
| `scripts.run_multiseed` | Run matched study settings across seeds and aggregate metrics. |
| `scripts.evaluate_extrapolation_distance` | Compare explicit saved MLP, Snake, and Fourier runs in shared distance bands. |

## Run artifacts

An experiment creates a timestamped directory beneath its configured output root.
Study runners additionally write job configurations, logs, and summary records.

| Artifact | Purpose |
|---|---|
| `config.yaml` | Fully resolved experiment settings. |
| `model/best_model.pt` | Validation-selected weights, architecture, selected epoch, and history. |
| `model/preprocessor.joblib` | Fitted training input transform; required for inference. |
| `metrics/metrics.json` | Regression metrics for the selected model on each split. |
| `metrics/baseline_metrics.json` | Scores for the fixed training-target-mean predictor. |
| `diagnostics/history.json` | Per-epoch training and validation MSE. |
| `diagnostics/timing.json` | Recorded experiment timing. |
| `data/predictions.npz` | Original observation arrays and predictions, when enabled. |
| `plots/*.png` | Diagnostic figures, when enabled. |

Fourier runs also record initialization details and fitted component spectra.
Checkpoints support inference; they do not contain optimizer state for resuming
training. Keep a checkpoint and its matching preprocessor together.

