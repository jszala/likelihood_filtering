"""Paired, reproducible comparison of the OU score-root and likelihood baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import process_time
from typing import Any

import numpy as np

from .baselines import kalman_mle, particle_grid_mle
from .config import CaseConfig, ExperimentConfig
from .estimation import GridScoreRootEstimator
from .experiment import _filter_seed, _make_observation, _make_signal, _simulate
from .filtering import run_augmented_filter
from .persistence import read_json, write_json
from .randomness import seeded_rng
from .signals import OrnsteinUhlenbeckSignal

MAIN_FIELDS = [
    "case",
    "replicate",
    "method",
    "theta_true",
    "theta_hat",
    "error",
    "absolute_error",
    "cpu_s",
    "status",
    "enkf_loglik_argmax_theta",
]
CHECK_FIELDS = [
    "case",
    "replicate",
    "primary_particles",
    "check_particles",
    "theta_primary",
    "theta_check",
    "absolute_difference",
    "cpu_s",
]
SUMMARY_FIELDS = [
    "case",
    "method",
    "replicates",
    "mae",
    "bias",
    "median_cpu_s",
    "boundary_rate",
    "missing_root_rate",
    "delta_mae_vs_baseline",
    "paired_ci_low",
    "paired_ci_high",
]


def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _particle_seed(master_seed: int, case: str, replicate: int) -> int:
    rng = seeded_rng(master_seed, case, "replicate", replicate, "particle_filter")
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _estimate_score_root(
    case: CaseConfig, observed: np.ndarray, master_seed: int, replicate: int
) -> tuple[float, str, float, float]:
    signal = _make_signal(case)
    observation = _make_observation(case)
    score_surface = np.empty((case.theta_points, case.steps + 1))
    likelihood_final = np.empty(case.theta_points)
    seed = _filter_seed(master_seed, case.name, replicate)
    started = process_time()
    for index, theta in enumerate(case.theta_grid):
        result = run_augmented_filter(
            signal, observation, observed, float(theta), case.ensemble_size, case.dt, seed
        )
        score_surface[index] = result.score_mean
        likelihood_final[index] = result.log_likelihood[-1]
    path, status, _ = GridScoreRootEstimator().estimate_path(
        case.theta_grid, score_surface, initial_value=float(case.theta_grid.mean())
    )
    elapsed = process_time() - started
    pseudo_likelihood_theta = float(case.theta_grid[int(np.argmax(likelihood_final))])
    return float(path[-1]), str(status[-1]), elapsed, pseudo_likelihood_theta


def _row(
    case: CaseConfig,
    replicate: int,
    method: str,
    theta_hat: float,
    runtime: float,
    status: str,
    pseudo_likelihood_theta: float | str = "",
) -> dict[str, Any]:
    error = theta_hat - case.theta_true
    return {
        "case": case.name,
        "replicate": replicate,
        "method": method,
        "theta_true": case.theta_true,
        "theta_hat": theta_hat,
        "error": error,
        "absolute_error": abs(error),
        "cpu_s": runtime,
        "status": status,
        "enkf_loglik_argmax_theta": pseudo_likelihood_theta,
    }


def _run_record(
    case: CaseConfig,
    master_seed: int,
    replicate: int,
    particles: int,
    check_particles: int,
    do_check: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    _, observed = _simulate(case, master_seed, replicate)
    signal = _make_signal(case)
    if not isinstance(signal, OrnsteinUhlenbeckSignal):
        raise ValueError("the benchmark supports OU cases only")
    observation = _make_observation(case)
    theta, status, runtime, pseudo_theta = _estimate_score_root(
        case, observed, master_seed, replicate
    )
    rows = [_row(case, replicate, "score_root", theta, runtime, status, pseudo_theta)]

    started = process_time()
    if case.name == "ou_linear_gaussian":
        estimate = kalman_mle(
            observed,
            signal,
            float(case.observation["coefficient"]),
            float(case.observation["observation_std"]),
            case.dt,
            (case.theta_min, case.theta_max),
        )
        baseline = "kalman"
    else:
        estimate = particle_grid_mle(
            observed,
            case.theta_grid,
            signal,
            observation,
            case.dt,
            particles,
            _particle_seed(master_seed, case.name, replicate),
        )
        baseline = f"particle_{particles}"
    runtime = process_time() - started
    if not np.isfinite(estimate.theta):
        raise ValueError(f"{case.name} record {replicate}: baseline had no finite likelihood")
    rows.append(_row(case, replicate, baseline, estimate.theta, runtime, estimate.status))

    check = None
    if do_check and baseline.startswith("particle_"):
        started = process_time()
        higher = particle_grid_mle(
            observed,
            case.theta_grid,
            signal,
            observation,
            case.dt,
            check_particles,
            _particle_seed(master_seed, case.name, replicate),
        )
        check = {
            "case": case.name,
            "replicate": replicate,
            "primary_particles": particles,
            "check_particles": check_particles,
            "theta_primary": estimate.theta,
            "theta_check": higher.theta,
            "absolute_difference": abs(estimate.theta - higher.theta),
            "cpu_s": process_time() - started,
        }
    return rows, check


def _run_task(
    task: tuple[CaseConfig, int, int, int, int, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    return _run_record(*task)


def _paired_interval(differences: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(10_000, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_benchmark(
    output_dir: str | Path, config: ExperimentConfig, particles: int
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    records = _read_csv(output_dir / "per_record.csv")
    rows: list[dict[str, Any]] = []
    for case in config.cases:
        case_records = [item for item in records if item["case"] == case.name]
        methods = sorted({item["method"] for item in case_records})
        by_method = {
            method: [item for item in case_records if item["method"] == method]
            for method in methods
        }
        baseline = "kalman" if case.name == "ou_linear_gaussian" else f"particle_{particles}"
        if set(methods) != {"score_root", baseline}:
            raise ValueError(f"incomplete methods for {case.name}")
        by_replicate = {
            method: {int(item["replicate"]): item for item in values}
            for method, values in by_method.items()
        }
        expected = set(range(case.replicates))
        if any(set(values) != expected for values in by_replicate.values()):
            raise ValueError(f"incomplete replicates for {case.name}")
        differences = np.asarray(
            [
                float(by_replicate["score_root"][rep]["absolute_error"])
                - float(by_replicate[baseline][rep]["absolute_error"])
                for rep in range(case.replicates)
            ]
        )
        interval_seed = int(
            seeded_rng(config.master_seed, case.name, "paired_bootstrap").integers(
                0, np.iinfo(np.uint32).max, dtype=np.uint32
            )
        )
        low, high = _paired_interval(differences, interval_seed)
        for method in ["score_root", baseline]:
            values = by_method[method]
            errors = np.asarray([float(item["error"]) for item in values])
            runtimes = np.asarray([float(item["cpu_s"]) for item in values])
            estimates = np.asarray([float(item["theta_hat"]) for item in values])
            statuses = [item["status"] for item in values]
            rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "replicates": len(values),
                    "mae": float(np.abs(errors).mean()),
                    "bias": float(errors.mean()),
                    "median_cpu_s": float(np.median(runtimes)),
                    "boundary_rate": float(
                        np.mean(
                            np.isclose(estimates, case.theta_min, atol=1e-8)
                            | np.isclose(estimates, case.theta_max, atol=1e-8)
                        )
                    ),
                    "missing_root_rate": (
                        sum(value not in {"exact_root", "interpolated_root"} for value in statuses)
                        / len(values)
                        if method == "score_root"
                        else 0.0
                    ),
                    "delta_mae_vs_baseline": float(differences.mean())
                    if method == "score_root"
                    else "",
                    "paired_ci_low": low if method == "score_root" else "",
                    "paired_ci_high": high if method == "score_root" else "",
                }
            )
    _write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, rows)
    return rows


def run_benchmark(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    particles: int = 4096,
    check_particles: int = 8192,
    check_replicates: int = 5,
    resume: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if particles < 2 or check_particles <= particles or check_replicates < 0 or workers < 1:
        raise ValueError("invalid particle counts or check count")
    names = [case.name for case in config.cases]
    if names != ["ou_linear_gaussian", "ou_nonlinear_gaussian", "ou_poisson"]:
        raise ValueError("benchmark requires the three canonical OU cases in order")
    expected_observations = [
        ("gaussian", "linear", "right"),
        ("gaussian", "sigmoid", "right"),
        ("poisson", "positive_sigmoid", "left"),
    ]
    for case, expected in zip(config.cases, expected_observations, strict=True):
        actual = (
            case.observation["type"],
            case.observation["map"],
            case.observation.get("evaluation_rule", "right"),
        )
        if (
            case.signal["type"] != "ou"
            or case.signal["dimensions"] != 1
            or case.estimator != "grid"
            or actual != expected
        ):
            raise ValueError(f"case {case.name} does not match its benchmark model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "experiment": config.to_dict(),
        "particles": particles,
        "particle_convergence_check": {
            "particles": check_particles,
            "first_replicates": check_replicates,
        },
        "data_policy": "one simulated record per case and replicate, shared by both methods",
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    if resume:
        if read_json(manifest_path) != manifest:
            raise ValueError("saved manifest differs; use a new output directory")
    elif manifest_path.exists():
        raise FileExistsError("output directory already contains a benchmark; use --resume")
    else:
        write_json(manifest_path, manifest)

    records_path = output_dir / "per_record.csv"
    checks_path = output_dir / "particle_check.csv"
    existing = _read_csv(records_path)
    existing_checks = _read_csv(checks_path)
    for case in config.cases:
        tasks = []
        for replicate in range(case.replicates):
            baseline = "kalman" if case.name == "ou_linear_gaussian" else f"particle_{particles}"
            seen = {
                item["method"]
                for item in existing
                if item["case"] == case.name and int(item["replicate"]) == replicate
            }
            has_check = any(
                item["case"] == case.name and int(item["replicate"]) == replicate
                for item in existing_checks
            )
            needs_check = case.name != "ou_linear_gaussian" and replicate < check_replicates
            if seen == {"score_root", baseline} and (not needs_check or has_check):
                continue
            tasks.append(
                (case, config.master_seed, replicate, particles, check_particles, needs_check)
            )
        if workers == 1:
            results = map(_run_task, tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(_run_task, tasks)
        try:
            for task, (rows, check) in zip(tasks, results, strict=True):
                replicate = task[2]
                print(
                    f"{case.name}: completed record {replicate + 1}/{case.replicates}",
                    flush=True,
                )
                seen = {
                    item["method"]
                    for item in existing
                    if item["case"] == case.name and int(item["replicate"]) == replicate
                }
                has_check = any(
                    item["case"] == case.name and int(item["replicate"]) == replicate
                    for item in existing_checks
                )
                for row in rows:
                    if row["method"] not in seen:
                        _append_csv(records_path, MAIN_FIELDS, row)
                if check is not None and not has_check:
                    _append_csv(checks_path, CHECK_FIELDS, check)
                existing.extend(rows)
                if check is not None:
                    existing_checks.append(check)
        finally:
            if executor is not None:
                executor.shutdown()
    return summarize_benchmark(output_dir, config, particles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--check-particles", type=int, default=8192)
    parser.add_argument("--smoke", action="store_true", help="one record and 20 steps per case")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    if args.smoke:
        config = replace(
            config,
            name="benchmark_smoke",
            cases=[
                replace(case, steps=20, replicates=1, ensemble_size=24) for case in config.cases
            ],
        )
    primary_particles = 32 if args.smoke else args.particles
    check_particles = 64 if args.smoke else args.check_particles
    summary = run_benchmark(
        config,
        args.output,
        particles=primary_particles,
        check_particles=check_particles,
        check_replicates=0 if args.smoke else 5,
        resume=args.resume,
        workers=args.workers,
    )
    for row in summary:
        print(f"{row['case']} {row['method']}: MAE={row['mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
