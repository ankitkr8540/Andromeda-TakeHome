"""Distance evaluation and matched-study summaries preserve experimental boundaries."""

import json

import numpy as np
import pytest

from scripts.evaluate_extrapolation_distance import distance_bands
from scripts.run_multiseed import aggregate


def test_distance_windows_exclude_training_and_clip_final_band():
    bands = list(distance_bands(10.0, 35.0, 10.0, 5))
    assert [(lo, hi) for lo, hi, _ in bands] == [
        (10.0, 20.0),
        (20.0, 30.0),
        (30.0, 35.0),
    ]
    x = np.concatenate([x for _, _, x in bands])
    assert x.min() > 10
    assert x.max() == 35
    assert len(np.unique(x)) == 15


@pytest.mark.parametrize(
    "args",
    [
        (10, 20, 0, 5),
        (10, 20, -1, 5),
        (10, 10, 1, 5),
        (10, float("inf"), 1, 5),
        (10, 20, 1, True),
        (10, 20, 1, 0),
    ],
)
def test_invalid_distance_bands_are_rejected(args):
    with pytest.raises(ValueError):
        list(distance_bands(*args))


def test_multiseed_summary_excludes_removed_arm_and_unrequested_seed(tmp_path):
    variants = ["random_adam_k8", "spectral_adam_k8"]
    for seed in (42, 43, 44, 45):
        folder = tmp_path / f"seed{seed}" / "results"
        folder.mkdir(parents=True)
        for variant in [*variants, "log_uniform_k128_adam"]:
            record = {
                "status": "complete",
                "variant": variant,
                "seed": seed,
                "metrics": {
                    region: {"rmse": float(seed - 40)}
                    for region in ("val", "test", "extrap_near", "extrap_far")
                },
                "timing": {"fit_seconds": 1.0},
            }
            (folder / f"{variant}.json").write_text(json.dumps(record))
    aggregate(tmp_path, variants=variants, seeds=[42, 43, 44])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 6
    assert len(summary["rows"]) == 8
    assert {row["variant"] for row in summary["rows"]} == set(variants)
    assert all(row["seeds"] == 3 for row in summary["rows"])
    assert all(row["mean_rmse"] == 3.0 for row in summary["rows"])
    assert all(row["sample_std_rmse"] == 1.0 for row in summary["rows"])
    assert len(list(tmp_path.glob("seed*/results/*.json"))) == 12


def test_matched_study_contains_only_same_capacity_initialization_comparison():
    import yaml

    from scripts.run_ablation import PROJECT_ROOT

    spec = yaml.safe_load(
        (PROJECT_ROOT / "configs/ablations/fourier/matched_multiseed.yaml").read_text()
    )
    assert spec["model"]["num_frequencies"] == 8
    assert set(spec["variants"]) == {"random_adam_k8", "spectral_adam_k8"}
    assert all("model" not in variant for variant in spec["variants"].values())
