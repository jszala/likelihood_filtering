"""Checkpoint comparison on paired records, with numerical sensitivity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np

from .baselines import kalman_mle, particle_grid_log_likelihoods
from .benchmark import _particle_seed, validate_benchmark_config
from .config import CaseConfig, ExperimentConfig
from .estimation import GridScoreRootEstimator
from .experiment import _filter_seed, _make_observation, _make_signal, _simulate
from .filtering import run_augmented_filter
from .persistence import read_json, write_json, write_metrics
from .randomness import seeded_rng


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _repeat_seed(seed: int, repeat: int) -> int:
    if repeat == 0:
        return seed
    return int(seeded_rng(seed, "comparison_repeat", repeat).integers(0, 2**32, dtype=np.uint32))


def _score_checkpoints(
    case: CaseConfig, observed: np.ndarray, seed: int, checkpoints: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.empty((case.theta_points, case.steps + 1))
    likelihoods = np.empty((case.theta_points, len(checkpoints)))
    for index, theta in enumerate(case.theta_grid):
        result = run_augmented_filter(
            _make_signal(case),
            _make_observation(case),
            observed,
            float(theta),
            case.ensemble_size,
            case.dt,
            seed,
        )
        scores[index] = result.score_mean
        likelihoods[index] = result.log_likelihood[list(checkpoints)]
    # Continuation uses all preceding roots, not just the reporting checkpoints.
    path, statuses, _ = GridScoreRootEstimator().estimate_path(
        case.theta_grid, scores, initial_value=float(case.theta_grid.mean())
    )
    return path[list(checkpoints)], statuses[list(checkpoints)], scores[:, checkpoints], likelihoods


def _run_record(task: dict[str, Any]) -> dict[str, Any]:
    case = task["case"]
    master_seed, replicate = task["master_seed"], task["replicate"]
    checkpoints = task["checkpoints"]
    _, observed = _simulate(case, master_seed, replicate)
    data_digest = hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()
    signal, observation = _make_signal(case), _make_observation(case)
    rows, costs, surfaces = [], [], []

    def collect(
        method: str,
        variant: str,
        repeat: int,
        seed: int | str,
        grid: np.ndarray,
        estimates: np.ndarray,
        statuses: Any,
        cpu: float,
        wall: float,
        values: np.ndarray | None = None,
        kind: str = "",
    ) -> None:
        costs.append(
            {
                "case": case.name,
                "replicate": replicate,
                "method": method,
                "variant": variant,
                "repeat": repeat,
                "cpu_s": cpu,
                "wall_s": wall,
            }
        )
        for index, step in enumerate(checkpoints):
            theta = float(estimates[index])
            if not np.isfinite(theta):
                raise FloatingPointError("nonfinite estimate; record not saved")
            rows.append(
                {
                    "case": case.name,
                    "replicate": replicate,
                    "method": method,
                    "variant": variant,
                    "repeat": repeat,
                    "filter_seed": seed,
                    "step": step,
                    "horizon": step * case.dt,
                    "theta_true": case.theta_true,
                    "theta_hat": theta,
                    "error": theta - case.theta_true,
                    "absolute_error": abs(theta - case.theta_true),
                    "status": str(statuses[index]),
                    "grid_points": len(grid),
                    "observation_sha256": data_digest,
                }
            )
        if values is not None:
            surfaces.append(
                {
                    "method": method,
                    "variant": variant,
                    "repeat": repeat,
                    "kind": kind,
                    "theta_grid": grid,
                    "values": values,
                }
            )

    do_checks = replicate < task["check_replicates"]
    repeats = task["filter_repeats"] if do_checks else 1
    for repeat in range(repeats):
        seed = _repeat_seed(_filter_seed(master_seed, case.name, replicate), repeat)
        cpu, wall = process_time(), perf_counter()
        estimates, statuses, scores, enkf_likelihoods = _score_checkpoints(
            case, observed, seed, checkpoints
        )
        collect(
            "score_root",
            "primary" if repeat == 0 else "seed_repeat",
            repeat,
            seed,
            case.theta_grid,
            estimates,
            statuses,
            process_time() - cpu,
            perf_counter() - wall,
            scores,
            "augmented_EnKF_score",
        )
        surfaces.append(
            {
                "method": "score_root",
                "variant": "primary" if repeat == 0 else "seed_repeat",
                "repeat": repeat,
                "kind": "EnKF_predictive_log_likelihood_diagnostic",
                "theta_grid": case.theta_grid,
                "values": enkf_likelihoods,
            }
        )

    if case.name == "ou_linear_gaussian":
        # Exact likelihood optimization at each checkpoint reads only that prefix.
        cpu, wall = process_time(), perf_counter()
        estimates = [
            kalman_mle(
                observed[:step],
                signal,
                float(case.observation["coefficient"]),
                float(case.observation["observation_std"]),
                case.dt,
                (case.theta_min, case.theta_max),
            )
            for step in checkpoints
        ]
        collect(
            "kalman",
            "primary",
            0,
            "",
            np.empty(0),
            np.array([item.theta for item in estimates]),
            [item.status for item in estimates],
            process_time() - cpu,
            perf_counter() - wall,
        )
    else:
        variants = [
            (
                "primary" if repeat == 0 else "seed_repeat",
                repeat,
                task["particles"],
                case.theta_grid,
            )
            for repeat in range(repeats)
        ]
        if do_checks:
            variants.append(("particle_count", 0, task["check_particles"], case.theta_grid))
            variants.append(
                (
                    "grid_resolution",
                    0,
                    task["particles"],
                    np.linspace(case.theta_min, case.theta_max, task["check_grid_points"]),
                )
            )
        for variant, repeat, particles, grid in variants:
            seed = _repeat_seed(_particle_seed(master_seed, case.name, replicate), repeat)
            cpu, wall = process_time(), perf_counter()
            likelihoods = particle_grid_log_likelihoods(
                observed, grid, signal, observation, case.dt, particles, seed, checkpoints
            )
            indices = np.argmax(likelihoods, axis=0)
            statuses = np.where((indices == 0) | (indices == len(grid) - 1), "boundary", "interior")
            collect(
                "particle",
                variant,
                repeat,
                seed,
                grid,
                grid[indices],
                statuses,
                process_time() - cpu,
                perf_counter() - wall,
                likelihoods,
                "particle_log_likelihood",
            )
            for row in rows[-len(checkpoints) :]:
                row["particles"] = particles
    # All rows share a uniform CSV schema.
    for row in rows:
        row.setdefault("particles", "")
    return {
        "case": case.name,
        "replicate": replicate,
        "checkpoints": checkpoints,
        "observation_sha256": data_digest,
        "rows": rows,
        "costs": costs,
        "surfaces": surfaces,
    }


def summarize_comparison(
    output: Path, config: ExperimentConfig, checkpoints: tuple[int, ...], bootstrap_samples: int
) -> list[dict[str, Any]]:
    artifacts = [
        read_json(output / "records" / case.name / f"{replicate:04d}.json")
        for case in config.cases
        for replicate in range(case.replicates)
    ]
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    keys = [
        (row["case"], row["replicate"], row["method"], row["variant"], row["repeat"], row["step"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate estimator/checkpoint rows")
    write_metrics(output / "estimates.csv", rows)
    write_metrics(
        output / "costs.csv", [row for artifact in artifacts for row in artifact["costs"]]
    )
    summary = []
    family_size = len(config.cases) * len(checkpoints)
    for case in config.cases:
        baseline = "kalman" if case.name == "ou_linear_gaussian" else "particle"
        for step in checkpoints:
            paired = {}
            for method in ("score_root", baseline):
                subset = [
                    row
                    for row in rows
                    if row["case"] == case.name
                    and row["step"] == step
                    and row["method"] == method
                    and row["variant"] == "primary"
                ]
                by_rep = {row["replicate"]: row for row in subset}
                if set(by_rep) != set(range(case.replicates)) or len(subset) != case.replicates:
                    raise ValueError("incomplete primary paired records")
                paired[method] = [by_rep[replicate] for replicate in range(case.replicates)]
            differences = np.array(
                [
                    a["absolute_error"] - b["absolute_error"]
                    for a, b in zip(paired["score_root"], paired[baseline], strict=True)
                ]
            )
            rng = seeded_rng(config.master_seed, case.name, "comparison_bootstrap", step)
            # Batches bound memory for large replicate counts; bootstrap unit is the record.
            means = np.concatenate(
                [
                    differences[
                        rng.integers(
                            0,
                            case.replicates,
                            size=(min(1000, bootstrap_samples - start), case.replicates),
                        )
                    ].mean(axis=1)
                    for start in range(0, bootstrap_samples, 1000)
                ]
            )
            low, high = np.quantile(means, [0.025, 0.975])
            family_low, family_high = np.quantile(
                means, [0.025 / family_size, 1 - 0.025 / family_size]
            )
            for method in ("score_root", baseline):
                values = paired[method]
                errors = np.array([row["error"] for row in values])
                estimates = np.array([row["theta_hat"] for row in values])
                score = method == "score_root"
                summary.append(
                    {
                        "case": case.name,
                        "method": method,
                        "step": step,
                        "horizon": step * case.dt,
                        "replicates": case.replicates,
                        "mae": float(np.abs(errors).mean()),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                        "bias": float(errors.mean()),
                        "estimate_sd": float(estimates.std(ddof=1)) if case.replicates > 1 else "",
                        "boundary_rate": float(
                            np.mean(
                                np.isclose(estimates, case.theta_min, rtol=0, atol=1e-8)
                                | np.isclose(estimates, case.theta_max, rtol=0, atol=1e-8)
                            )
                        ),
                        "missing_root_rate": float(
                            np.mean(
                                [
                                    row["status"] not in {"exact_root", "interpolated_root"}
                                    for row in values
                                ]
                            )
                        )
                        if score
                        else "",
                        "delta_mae_vs_baseline": float(differences.mean()) if score else "",
                        "paired_delta_mcse": float(
                            differences.std(ddof=1) / np.sqrt(case.replicates)
                        )
                        if score and case.replicates > 1
                        else "",
                        "paired_ci_low": float(low) if score else "",
                        "paired_ci_high": float(high) if score else "",
                        "family_ci_low": float(family_low) if score else "",
                        "family_ci_high": float(family_high) if score else "",
                    }
                )
    write_metrics(output / "summary.csv", summary)
    sensitivity = []
    primary = {
        (row["case"], row["replicate"], row["method"], row["step"]): row
        for row in rows
        if row["variant"] == "primary"
    }
    for row in rows:
        if row["variant"] != "primary":
            first = primary[(row["case"], row["replicate"], row["method"], row["step"])]
            sensitivity.append(
                {
                    **row,
                    "theta_primary": first["theta_hat"],
                    "absolute_change": abs(row["theta_hat"] - first["theta_hat"]),
                }
            )
    if sensitivity:
        write_metrics(output / "sensitivity.csv", sensitivity)
    return summary


def run_comparison(
    config: ExperimentConfig,
    output: str | Path,
    *,
    checkpoints: tuple[int, ...] = (25000, 50000, 100000),
    particles: int = 8192,
    check_particles: int = 16384,
    check_replicates: int = 5,
    check_grid_points: int = 37,
    filter_repeats: int = 2,
    workers: int = 1,
    resume: bool = False,
    bootstrap_samples: int = 50000,
) -> list[dict[str, Any]]:
    validate_benchmark_config(config)
    for case in config.cases:
        case.validate()
        if (
            not checkpoints
            or tuple(sorted(set(checkpoints))) != checkpoints
            or checkpoints[0] < 1
            or checkpoints[-1] != case.steps
        ):
            raise ValueError("checkpoints must increase and include the final configured step")
        if check_grid_points <= case.theta_points:
            raise ValueError("sensitivity grid must be finer than the primary grid")
    if (
        particles < 2
        or check_particles <= particles
        or check_replicates < 0
        or filter_repeats < 1
        or workers < 1
        or bootstrap_samples < 1000
    ):
        raise ValueError("invalid comparison settings")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": config.to_dict(),
        "checkpoint_steps": list(checkpoints),
        "particles": particles,
        "check_particles": check_particles,
        "check_replicates": check_replicates,
        "check_grid_points": check_grid_points,
        "filter_repeats": filter_repeats,
        "bootstrap_samples": bootstrap_samples,
        "source_sha256": _source_digest(),
        "software": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "PyYAML", "matplotlib", "likelihood-filtering")
            },
        },
        "design": {
            "primary_metric": "paired MAE difference at final horizon per model",
            "earlier_checkpoints": "secondary; identical record prefixes",
            "family_intervals": "Bonferroni percentile bootstrap across all model/checkpoint pairs",
            "bootstrap_unit": "independent paired observation record",
            "sensitivity_subset": "first check_replicates; never pooled with primary records",
            "timing": "total method workload for entire record; not per-checkpoint online latency",
        },
    }
    manifest_path = output / "manifest.json"
    if resume:
        if read_json(manifest_path) != manifest:
            raise ValueError("manifest/source/software changed; use a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError("output must be empty for a fresh run; use --resume")
    else:
        _atomic_json(manifest_path, manifest)
    tasks = []
    for case in config.cases:
        for replicate in range(case.replicates):
            path = output / "records" / case.name / f"{replicate:04d}.json"
            if path.exists():
                artifact = read_json(path)
                if (
                    artifact["case"] != case.name
                    or artifact["replicate"] != replicate
                    or artifact["checkpoints"] != list(checkpoints)
                ):
                    raise ValueError(f"record identity mismatch: {path}")
                continue
            tasks.append(
                {
                    "case": case,
                    "master_seed": config.master_seed,
                    "replicate": replicate,
                    "checkpoints": checkpoints,
                    "particles": particles,
                    "check_particles": check_particles,
                    "check_replicates": check_replicates,
                    "check_grid_points": check_grid_points,
                    "filter_repeats": filter_repeats,
                }
            )

    # Start the expensive particle sensitivity records first to reduce the final
    # queue tail. This changes only scheduling, never seeds or statistical inclusion.
    tasks.sort(
        key=lambda task: (
            task["case"].name == "ou_linear_gaussian",
            task["replicate"] >= check_replicates,
            task["replicate"],
            task["case"].name,
        )
    )

    def save(artifact: dict[str, Any]) -> None:
        _atomic_json(
            output / "records" / artifact["case"] / f"{artifact['replicate']:04d}.json", artifact
        )
        print(f"saved {artifact['case']} record {artifact['replicate'] + 1}", flush=True)

    print(f"{len(tasks)} pending records; {workers} workers", flush=True)
    if workers == 1:
        for task in tasks:
            save(_run_record(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_record, task) for task in tasks]
            for future in as_completed(futures):
                save(future.result())
    return summarize_comparison(output, config, checkpoints, bootstrap_samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/comparison.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--check-particles", type=int, default=16384)
    parser.add_argument("--check-replicates", type=int, default=5)
    parser.add_argument("--check-grid-points", type=int, default=37)
    parser.add_argument("--filter-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument(
        "--checkpoints",
        type=float,
        nargs="+",
        default=[250, 500, 1000],
        help="reporting times; must include the configured final horizon",
    )
    parser.add_argument("--replicates", type=int, help="override records per model for a pilot")
    parser.add_argument(
        "--master-seed", type=int, help="override simulation and filter master seed"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    if args.master_seed is not None:
        config = replace(config, master_seed=args.master_seed)
    if args.replicates is not None:
        config = replace(
            config, cases=[replace(case, replicates=args.replicates) for case in config.cases]
        )
    if args.smoke:
        config = replace(
            config,
            name="comparison_smoke",
            cases=[
                replace(case, steps=20, replicates=2, ensemble_size=12) for case in config.cases
            ],
        )
        checkpoints = (5, 10, 20)
        args.particles, args.check_particles = 32, 64
        args.check_replicates, args.bootstrap_samples = 1, 1000
    else:
        if len({case.dt for case in config.cases}) != 1:
            parser.error("CLI requires a shared time step")
        dt = config.cases[0].dt
        if any(
            not np.isclose(time / dt, round(time / dt), rtol=0, atol=1e-8)
            for time in args.checkpoints
        ):
            parser.error("checkpoints must be integer multiples of dt")
        checkpoints = tuple(round(time / dt) for time in args.checkpoints)
    summary = run_comparison(
        config,
        args.output,
        checkpoints=checkpoints,
        particles=args.particles,
        check_particles=args.check_particles,
        check_replicates=args.check_replicates,
        check_grid_points=args.check_grid_points,
        filter_repeats=args.filter_repeats,
        workers=args.workers,
        resume=args.resume,
        bootstrap_samples=args.bootstrap_samples,
    )
    for row in summary:
        print(f"{row['case']} T={row['horizon']:g} {row['method']}: MAE={row['mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
