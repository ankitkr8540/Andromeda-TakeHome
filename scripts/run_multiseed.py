"""Run an ablation specification across seeds and aggregate matched results."""

import argparse
import csv
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_ablation import prepare_jobs, run_study, write_json


def aggregate(output: Path, *, variants: list[str], seeds: list[int]) -> None:
    """Export per-variant mean RMSE, sample standard deviation, and mean fitting cost.

    Args:
        output: Study root containing ``seed*/results/*.json`` records.
        variants: Variant names in the requested study; exclude retired arms.
        seeds: Requested seeds; exclude results from other seeds.

    Notes:
        Includes completed runs only. Writes ``summary.json`` and, when results
        exist, ``summary.csv``. Standard deviation uses ddof=1; one seed gives None.
        Aggregates validation, test, near, and far scores without reselecting models."""
    records = []
    for path in output.glob("seed*/results/*.json"):
        record = json.loads(path.read_text())
        if (
            record.get("status") == "complete"
            and record.get("variant") in variants
            and record.get("seed") in seeds
        ):
            records.append(record)
    rows = []
    for variant in sorted({r["variant"] for r in records}):
        subset = [r for r in records if r["variant"] == variant]
        for region in ("val", "test", "extrap_near", "extrap_far"):
            scores = [r["metrics"][region]["rmse"] for r in subset]
            rows.append(
                {
                    "variant": variant,
                    "region": region,
                    "seeds": len(scores),
                    "mean_rmse": float(np.mean(scores)),
                    "sample_std_rmse": float(np.std(scores, ddof=1))
                    if len(scores) > 1
                    else None,
                    "mean_fit_seconds": float(
                        np.mean([r["timing"]["fit_seconds"] for r in subset])
                    ),
                }
            )
    if rows:
        with (output / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "completed": len(records),
            "rows": rows,
            "note": "Mean and sample standard deviation across completed seeds; report realized fitting cost alongside the configured optimization budget.",
        },
    )


def main():
    """Run the supplied study across unique seeds and aggregate completed metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique.")
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(args.study.read_text())
    with (output / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This study is already running.")
        for seed in args.seeds:
            directory = output / f"seed{seed}"
            directory.mkdir(exist_ok=True)
            study_path = directory / "study_spec.yaml"
            seed_spec = {**spec, "seed": seed}
            if (
                study_path.exists()
                and yaml.safe_load(study_path.read_text()) != seed_spec
            ):
                raise ValueError("Study changed; use a new output directory.")
            study_path.write_text(yaml.safe_dump(seed_spec, sort_keys=False))
            study, jobs = prepare_jobs(study_path, directory, args.max_epochs)
            if args.dry_run:
                print([j["run_id"] for j in jobs])
                continue
            run_study(study, jobs, directory)
            aggregate(output, variants=list(spec["variants"]), seeds=args.seeds)


if __name__ == "__main__":
    main()
