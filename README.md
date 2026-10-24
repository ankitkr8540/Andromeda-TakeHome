# Andromeda: function approximation and extrapolation

A reproducible PyTorch benchmark comparing a Tanh MLP, a trainable Snake MLP,
and a learnable Fourier regressor on one noise-free periodic function. Models
learn from sampled `(x, y)` pairs. The reference function generates labels and
evaluation curves; saved-model inference never calls it.

The central question is whether a model that interpolates accurately also
continues the waveform beyond the observed interval.

## Setup

Use Python 3.11 and run commands from the repository root. Create an isolated
environment (`--copies` is useful when working on an external drive):

```bash
python3.11 -m venv --copies .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv sync --locked --inexact
```

In a new terminal, activate the environment again with
`source .venv/bin/activate`. The examples below use `uv run` and the locked
project dependencies. Training defaults to CPU; MLflow tracking is disabled.

## Repository structure

```text
function_approximation/
├── data/             # reference labels, deterministic grids, datasets, preprocessing
├── models/           # Tanh MLP, Snake MLP, Fourier regressor, model construction
├── training/         # config loading/validation, optimization, checkpoints, tracking
├── evaluation/       # metrics and shared presentation-style plots
└── inference.py      # prediction from a saved run without reference labels
configs/
├── base.yaml         # shared training, preprocessing, evaluation, tracking defaults
├── data/             # target, sample count, observed domain, evaluation bands
├── model/            # model-specific defaults
├── experiments/      # standalone MLP, Snake, and Fourier recipes
└── ablations/
    ├── mlp/          # architecture, activation, learning rate, batch size
    ├── snake/        # controlled Tanh versus Snake comparison
    └── fourier/      # initialization, capacity, random-start tuning, multiple seeds
scripts/              # experiment/study runners and distance evaluation
tests/               # automated tests
```

YAML includes resolve relative to the including file. Files merge in listed
order, followed by local overrides. Mappings merge recursively; lists replace.
Relative `experiment.output_dir` paths are anchored to the repository root,
which is detected from the runner's location, rather than your username or
current working directory. The supplied configs need no path edits after cloning.
Generated study and notebook configs also use repository-relative output paths
for destinations inside the project. Explicit absolute destinations outside the
project remain supported.

Every run saves its resolved configuration. Configuration validation happens
before the experiment runner creates a run directory.

## Data and evaluation protocol

The reference `rich_periodic` is defined in
[`dataset.py`](function_approximation/data/dataset.py):

```text
f(x) = sin(x) + 0.55 cos(2x) + 0.35 sin(3x + 0.4)
       + 0.20 cos(5x - 0.7) + 0.12 sin(7x) + 0.08 cos(11x + 0.2)
```

| Split | Inputs | Samples |
|---|---|---:|
| Training | [−10π, 10π], including boundaries | 7,000 |
| Validation | Interior of the same interval | 1,000 |
| Held-out interpolation test | Interior of the same interval | 2,000 |
| Near extrapolation | (10π, 20π] | 2,000 |
| Far extrapolation | (50π, 60π] | 2,000 |

All grids are deterministic. Validation and test use different interior cell
offsets, adjusted when necessary to remain disjoint after float32 conversion
for both raw and standardized inputs. `data.sample_size` controls the in-domain
70%/10%/20% split; evaluation-band sample counts are separate.

StandardScaler, when enabled, fits training inputs only. Training minimizes
MSE with Adam; checkpoint selection and early stopping use validation MSE.
`min_delta` is in **MSE units**. Test and extrapolation labels are used only for
evaluation. Metrics include MSE, RMSE, MAE, and R² in original target units,
plus a baseline that predicts the training-label mean.

## End-to-end notebook

Open [`notebooks/fourier_end_to_end.ipynb`](notebooks/fourier_end_to_end.ipynb)
and select the repository's `.venv/bin/python` kernel. Run cells in order to
generate 1,000 in-domain samples, inspect spectral initialization, train with
Adam, view metrics/plots, and reload the saved checkpoint for inference.
Change `SAMPLE_SIZE` in the notebook to use another budget. Each execution saves
a fresh run under `outputs/notebooks/fourier/`; it does not overwrite the study
results. A notebook interface and `ipykernel` are optional environment tools,
not runtime requirements of the training package.

## Train the standalone models

```bash
uv run python -m scripts.run_experiments --config configs/experiments/mlp_rich_periodic.yaml
uv run python -m scripts.run_experiments --config configs/experiments/snake_rich_periodic.yaml
uv run python -m scripts.run_experiments --config configs/experiments/fourier_rich_periodic.yaml
```

| Recipe | Model / preprocessing | Adam LR | Batch | Maximum epochs | Patience |
|---|---|---:|---:|---:|---:|
| Reference MLP | Tanh [64, 256, 128], standardized x | 0.001 | 64 | 20,000 | 150 |
| Tuned Snake | [64, 256, 128], raw x | 0.001 | 128 | 8,000 | 1,500 |
| Fourier | K=8, spectral initialization, raw x | 0.001 | 64 | 30,000 | 3,000 |

The selected Tanh model in the presentation is the **learning-rate 0.0001
ablation**, rather than the standalone reference recipe. Snake learns one
positive activation scale per hidden neuron using
`Snake(z) = z + sin²(a z) / a`; the standalone recipe uses LR 0.001 for both
network weights and activation parameters.

Fourier predictions are a bias plus K weighted sine/cosine pairs (3K+1
parameters). Spectral initialization scores candidate frequencies on training
residuals, refines selected candidates locally, and refits the coefficients.
Adam then jointly updates frequencies, coefficients, and bias.

The configured angular-frequency search and random-initialization interval is
**[0.1, 20] radians per model-input unit**, identical for the matched random and
spectral arms. It is an explicit bandwidth prior, not a bound derived
by the code from Nyquist or an automatically inferred target period. Adam does
not constrain learned frequencies to that interval. No exact target frequencies
are supplied to the fitting procedure.

## Run the studies

Each command runs the full configured budget with early stopping. These studies
can take substantial CPU time. Add `--dry-run` to prepare configs without
training. Use a fresh output directory when changing a study specification;
rerunning the same study directory resumes matching completed jobs.

```bash
# Eight MLP architecture/hyperparameter variants
uv run python -m scripts.run_ablation \
  --study configs/ablations/mlp/architecture.yaml \
  --output-dir outputs/ablations/mlp/architecture

# Two activation variants with the same standardized-input training recipe
uv run python -m scripts.run_ablation \
  --study configs/ablations/snake/activation_comparison.yaml \
  --output-dir outputs/ablations/snake/activation_comparison

# Five Fourier variants: random + Adam K=8; spectral only K=8;
# spectral + Adam K=4, K=8, K=16
uv run python -m scripts.run_ablation \
  --study configs/ablations/fourier/initialization_and_capacity.yaml \
  --output-dir outputs/ablations/fourier/initialization_and_capacity

# Seven exploratory random-start variants, capped at 1,500 epochs
uv run python -m scripts.run_ablation \
  --study configs/ablations/fourier/random_pilot.yaml \
  --output-dir outputs/ablations/fourier/random_pilot

# Log-uniform K=128 random-start confirmation, up to 30,000 epochs
uv run python -m scripts.run_ablation \
  --study configs/ablations/fourier/random_confirmation.yaml \
  --output-dir outputs/ablations/fourier/random_confirmation

# Matched K=8 random versus spectral initialization: six runs, three seeds
uv run python -m scripts.run_multiseed \
  --study configs/ablations/fourier/matched_multiseed.yaml \
  --output-dir outputs/ablations/fourier/matched_multiseed \
  --seeds 42 43 44
```

The ablation runner defaults to one CPU thread per run. The multi-seed runner
also uses one thread. Seeds control initialization and training shuffling;
the input grids do not change with seed. Matched arms share the Adam budget
and frequency interval, but spectral initialization performs additional
preprocessing work. The exploratory K=128 arm is not part of that matched study.

## Read the outputs

Standalone runs create a new UTC-timestamped directory beneath their configured
`experiment.output_dir`. The runner prints the run path on completion:

```text
<run>/
├── config.yaml
├── model/
│   ├── best_model.pt            # validation-selected checkpoint
│   └── preprocessor.joblib      # fitted preprocessing, including identity mode
├── data/
│   └── predictions.npz          # original inputs, labels, predictions by split
├── metrics/
│   ├── metrics.json
│   └── baseline_metrics.json
├── diagnostics/
│   ├── history.json             # MSE history
│   ├── timing.json
│   ├── initialization.json      # Fourier only
│   └── spectrum.json            # Fourier only
└── plots/
    ├── predictions.png
    ├── residuals.png
    ├── learning_curves.png
    └── extrapolation.png
```

Plots and prediction exports are configurable. Spectral-only runs omit learning
curves because they do not perform gradient updates. Curve plots use π axis
labels, gray reference curves, and magenta predictions.

Replace `RUN_DIR` below with the printed timestamped run directory:

```bash
uv run python -m json.tool RUN_DIR/metrics/metrics.json
# macOS; on other systems, use your image viewer
open RUN_DIR/plots/extrapolation.png
```

Study roots contain `configs/`, `results/`, `logs/`, and `runs/`, plus
`metrics.csv`, `progress.json`, `selected_models.json`, and comparison plots.
Their nested runs follow the same artifact layout. Outputs are ignored by Git;
a fresh clone must generate its own checkpoints.

## Inference from a saved run

```bash
uv run python -m function_approximation.inference \
  --run-dir RUN_DIR \
  --x -3.14 0 100
```

Pass the **run root**, not its `model/` subdirectory. The CLI loads the checkpoint
and matching preprocessor, accepts original unscaled inputs, and returns JSON
predictions. It also supports legacy flat run directories. No training data,
labels, or reference-function calls are needed. Extrapolation is permitted but
accuracy outside the observed interval is not guaranteed.

## Evaluate error versus distance

Supply the timestamped run roots for the selected models:

```bash
uv run python -m scripts.evaluate_extrapolation_distance \
  --run "Tanh MLP" path/to/selected/tanh/run \
  --run "Snake MLP" path/to/snake/run \
  --run "Fourier spectral only" path/to/spectral/only/run \
  --run "Fourier spectral + Adam" path/to/spectral/adam/run \
  --output-dir outputs/evaluation/model_distance \
  --max-pi 1000
```

The evaluator loads fixed checkpoints with a common target, training domain,
and sample count. It evaluates 10π-wide bands with 2,000 samples each by default;
no model is trained or reselected. Outputs include per-band `metrics.csv`,
`source.json` checkpoint provenance, and PNG/SVG figures. The x-axis reports band
centres; the last band (990π, 1000π] is plotted at 995π. The saved training mean
from the first run supplies the per-band baseline when available.

## Verified results and interpretation

A full reproduction on the local CPU environment verified 33 saved runs,
including checkpoint reloads, exported predictions, and recomputed metrics.
The following seed-42 scores use the current disjoint test grid:

| Selected model | Interpolation RMSE | Near RMSE | Far RMSE |
|---|---:|---:|---:|
| Tanh MLP, LR 0.0001 | 0.03640 | 1.60579 | 1.94616 |
| Tuned raw-input Snake MLP | 0.005507 | 0.12203 | 0.38740 |
| Fourier, spectral + Adam, K=8 | 1.55×10⁻⁶ | 3.37×10⁻⁶ | 1.22×10⁻⁵ |

These are recorded results, not numerical guarantees across hardware, library
versions, or thread settings. The tuned Snake also changes scaling, learning
rate, batch size, and patience; its advantage is not an activation-only result.
Use the controlled activation study for that comparison.

Fourier K=8 was selected by validation RMSE. K=4 left a larger approximation
error; K=16 did not improve the measured fit and had worse far extrapolation.
Small interior errors do not establish reliable extrapolation: frequency
mismatch accumulates phase error with distance, and float32 evaluation also
contributes at large inputs. These findings concern one noise-free, stationary
periodic target; they do not establish performance on noisy or non-periodic data.

Historical test scores can differ slightly because an older grid contained one
training/test coordinate collision after float32 conversion. The current grid
removes it. Use the saved prediction export to inspect historical samples;
do not mix old-grid metrics with newly generated evaluations.

## Tests and checks

```bash
MPLBACKEND=Agg uv run python -m pytest -q
uv run ruff check function_approximation scripts tests
```

The latest full preflight passed **320 tests across 21 test files**. Coverage
includes disjoint data grids, train-only preprocessing, config validation,
model construction, initialization, training safety, checkpoint restoration,
inference without target calls, metrics, visualization, study runners, and
MLflow artifact logging. The count describes that verification snapshot.

## Optional MLflow tracking

Set `tracking.enabled: true` in YAML and start a local server separately:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5001
```

The default tracking URI is `http://127.0.0.1:5001`. Runs log parameters,
training metrics, and enabled artifacts. Local outputs are saved whether or
not tracking is enabled.
