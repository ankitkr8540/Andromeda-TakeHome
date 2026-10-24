"""Run, resume, and summarize an MLP, Snake, or Fourier ablation study."""

import argparse
import csv
import fcntl
import hashlib
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from function_approximation.evaluation.visualization import (
    BURGUNDY,
    GROUND_TRUTH,
    style_figure,
)
from function_approximation.training.config import load_experiment_config
from scripts.run_experiments import PROJECT_ROOT, run_experiment


def prepare_jobs(
    study_path: Path, output_dir: Path, max_epochs: int | None = None
) -> tuple[dict, list[dict]]:
    """Resolve study variants into immutable experiment configs and fingerprinted jobs.

    Args:
        study_path: YAML study specification listing targets, variants, and seed.
        output_dir: Study root in which per-run configs are written.
        max_epochs: Optional override applied to every variant's maximum budget.

    Returns:
        Parsed study mapping and ordered list of jobs with config paths, run IDs,
        target/variant names, and configuration fingerprints.

    Raises:
        ValueError: Configs are invalid, existing settings changed, or the reference
            is not included among the study variants.

    Notes:
        Creates configuration files but does not train. Use a new output directory
        when changing settings to avoid mixing incompatible resumed results."""
    study = yaml.safe_load(study_path.read_text())
    jobs = []
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for target in study["targets"]:
        model_type = study.get("model_type", "mlp")
        template = PROJECT_ROOT / f"configs/experiments/{model_type}_{target}.yaml"
        for variant, overrides in study["variants"].items():
            config = load_experiment_config(template)
            config["model"].update(study["model"])
            for section, values in overrides.items():
                config[section].update(values)
            config["training"]["max_epochs"] = (
                max_epochs if max_epochs is not None else study["max_epochs"]
            )
            run_id = f"{target}_{variant}_seed{study['seed']}"
            run_output = (output_dir / "runs" / run_id).resolve()
            try:
                # Keep generated configs portable when outputs live in the repo.
                configured_output = run_output.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                # Explicit destinations outside the project remain supported.
                configured_output = str(run_output)
            config["experiment"].update(
                seed=study["seed"],
                run_name=run_id,
                output_dir=configured_output,
            )
            config_path = config_dir / f"{run_id}.yaml"
            fingerprint = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            # Keep a started study immutable; never silently mix different settings.
            if config_path.exists():
                existing = load_experiment_config(config_path)
                if existing != config:
                    raise ValueError(
                        f"Settings changed for {run_id}; choose a new --output-dir."
                    )
            else:
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            load_experiment_config(config_path)
            jobs.append(
                {
                    "run_id": run_id,
                    "target": target,
                    "variant": variant,
                    "config_path": str(config_path),
                    "fingerprint": fingerprint,
                }
            )
    if study["reference"] not in study["variants"]:
        raise ValueError("Study reference must be one of its variants.")
    return study, jobs


def write_json(path: Path, value) -> None:
    """Atomically replace a JSON progress record using a temporary sibling file.

    Args:
        path: Destination file whose parent directory already exists.
        value: JSON-serializable value; NaN and infinity are rejected.

    Notes:
        A temporary file is renamed over the destination to avoid partial records
        during interrupted study runs."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def summarize(study: dict, jobs: list[dict], output_dir: Path) -> int:
    """Export completed-run metrics and select variants by validation RMSE only.

    Args:
        study: Parsed specification with targets, variants, and reference settings.
        jobs: Prepared job records with the expected configuration fingerprints.
        output_dir: Study directory containing per-job result records.

    Returns:
        Number of completed records whose fingerprints match the current jobs.

    Notes:
        Writes summary tables, selected-configuration records, and comparison
        figures when results exist. Test and extrapolation scores are reported,
        but never used to choose the selected configuration."""
    records = []
    for job in jobs:
        path = output_dir / "results" / f"{job['run_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["fingerprint"] == job["fingerprint"]:
                records.append(record)
    rows = []
    for record in records:
        if record["status"] != "complete":
            continue
        for region, metrics in record["metrics"].items():
            rows.append(
                {
                    key: record[key]
                    for key in (
                        "target",
                        "variant",
                        "seed",
                        "parameters",
                        "run_seconds",
                        "best_epoch",
                        "epochs_completed",
                    )
                }
                | {"region": region}
                | record.get("timing", {})
                | metrics
            )
    if rows:
        with (output_dir / "metrics.csv").open("w", newline="") as file:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    complete = [r for r in records if r["status"] == "complete"]
    write_json(
        output_dir / "progress.json",
        {
            "expected": len(jobs),
            "completed": len(complete),
            "failed": sum(r["status"] == "failed" for r in records),
            "note": "One seed per configuration; no seed-variance estimate. Run time includes evaluation and artifact writing.",
        },
    )
    if len(complete) != len(jobs):
        return len(complete)
    winners = {}
    for target in study["targets"]:
        candidates = [r for r in complete if r["target"] == target]
        winner = min(candidates, key=lambda r: r["metrics"]["val"]["rmse"])
        winners[target] = {
            key: winner[key]
            for key in ("variant", "output_dir", "best_epoch", "metrics")
        }
    write_json(
        output_dir / "selected_models.json",
        {
            "selection_rule": "Lowest validation RMSE separately for each target; test scores are reporting only.",
            "seed": study["seed"],
            "models": winners,
        },
    )
    regions = [
        ("val", "Validation"),
        ("test", "Interpolation test"),
        ("extrap_near", "Near extrapolation"),
        ("extrap_far", "Far extrapolation"),
    ]
    fig, axes = plt.subplots(
        len(study["targets"]),
        4,
        figsize=(19, max(5.5, 4.5 * len(study["targets"]))),
        squeeze=False,
        layout="constrained",
    )
    variants = list(study["variants"])
    for row, target in enumerate(study["targets"]):
        by_variant = {r["variant"]: r for r in complete if r["target"] == target}
        selected = winners[target]["variant"]
        for ax, (region, title) in zip(axes[row], regions, strict=True):
            scores = [by_variant[v]["metrics"][region]["rmse"] for v in variants]
            colors = [BURGUNDY if v == selected else GROUND_TRUTH for v in variants]
            ax.bar(np.arange(len(variants)), scores, color=colors)
            ax.set_xticks(
                np.arange(len(variants)), variants, rotation=65, ha="right", fontsize=8
            )
            ax.set(title=f"{target}\n{title}", ylabel="RMSE")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.suptitle(
        f"{study.get('model_type', 'mlp').upper()} ablation — seed {study['seed']} (burgundy = validation-selected configuration)"
    )
    style_figure(fig)
    fig.savefig(output_dir / "comparison.png", dpi=250, facecolor="white")
    fig.savefig(output_dir / "comparison.svg")
    plt.close(fig)
    return len(complete)


def run_study(study: dict, jobs: list[dict], output_dir: Path) -> None:
    """Run pending jobs and safely resume matching completed study results.

    Args:
        study: Parsed study settings, including the training seed.
        jobs: Ordered prepared jobs with paths and fingerprints.
        output_dir: Study root receiving logs, results, and summaries.

    Raises:
        RuntimeError: Any jobs remain incomplete after all attempts.

    Notes:
        Captures each run's logs, records failures, and continues independent jobs.
        Completed jobs are skipped only when their fingerprint and checkpoint match.
        The command-line caller owns the process lock preventing concurrent writers."""
    for directory in ("results", "logs"):
        (output_dir / directory).mkdir(exist_ok=True)
    write_json(output_dir / "study.json", {"study": study, "jobs": jobs})
    summarize(study, jobs, output_dir)
    for index, job in enumerate(jobs, start=1):
        record_path = output_dir / "results" / f"{job['run_id']}.json"
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if (
                previous["fingerprint"] == job["fingerprint"]
                and previous["status"] == "complete"
                and (Path(previous["output_dir"]) / "model" / "best_model.pt").is_file()
            ):
                print(f"[{index}/{len(jobs)}] Resume: {job['run_id']}", flush=True)
                continue
        print(f"[{index}/{len(jobs)}] Running {job['run_id']}", flush=True)
        started = time.perf_counter()
        record = {**job, "seed": study["seed"]}
        try:
            with (
                (output_dir / "logs" / f"{job['run_id']}.log").open(
                    "w", buffering=1
                ) as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                run_dir = run_experiment(job["config_path"])
            checkpoint = torch.load(
                (Path(run_dir) / "model" / "best_model.pt"),
                map_location="cpu",
                weights_only=True,
            )
            history = json.loads(
                (Path(run_dir) / "diagnostics" / "history.json").read_text()
            )
            record.update(
                status="complete",
                output_dir=str(run_dir),
                run_seconds=time.perf_counter() - started,
                parameters=sum(
                    value.numel() for value in checkpoint["model_state_dict"].values()
                ),
                best_epoch=checkpoint["best_epoch"],
                epochs_completed=len(history["train_loss"]),
                metrics=json.loads(
                    (Path(run_dir) / "metrics" / "metrics.json").read_text()
                ),
                timing=json.loads(
                    (Path(run_dir) / "diagnostics" / "timing.json").read_text()
                ),
            )
            print(
                f"  Complete in {record['run_seconds']:.1f}s; validation RMSE {record['metrics']['val']['rmse']:.6f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 -- record failure and continue independent jobs
            record.update(
                status="failed",
                error=repr(error),
                run_seconds=time.perf_counter() - started,
            )
            print(f"  Failed: {error!r}", flush=True)
        write_json(record_path, record)
        summarize(study, jobs, output_dir)
    completed = summarize(study, jobs, output_dir)
    if completed != len(jobs):
        raise RuntimeError(
            f"{completed}/{len(jobs)} runs completed; inspect logs and rerun to resume."
        )
    print(f"Study complete: {output_dir}", flush=True)


def main() -> None:
    """Parse study settings and run or resume the configured ablation under a process lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, required=True, help="Ablation YAML specification."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for configs, results, and summaries.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override the budget; use a separate output directory.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads per run; defaults to 1 for small MLPs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "This study is already running; inspect its logs before resuming."
            )
        study, jobs = prepare_jobs(args.study.resolve(), output_dir, args.max_epochs)
        if args.dry_run:
            for job in jobs:
                print(job["run_id"])
            print(f"{len(jobs)} runs prepared. No training performed.")
            return
        run_study(study, jobs, output_dir)


if __name__ == "__main__":
    main()
