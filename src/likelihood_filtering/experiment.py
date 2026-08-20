from __future__ import annotations

import importlib.metadata
import platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import CaseConfig, ExperimentConfig
from .estimation import GridScoreRootEstimator
from .experimental import RecursiveLouisEstimator
from .filtering import run_augmented_filter, run_recursive_filter
from .observations import (
    GaussianObservation,
    PoissonObservation,
    linear_observation,
    positive_sigmoid_intensity,
    sensor_sigmoid_observation,
    sigmoid_observation,
)
from .persistence import read_json, read_metrics, write_json, write_metrics
from .randomness import seeded_rng
from .signals import Heat1DFEMSignal, OrnsteinUhlenbeckSignal, ParametricSignal


def _make_signal(case: CaseConfig) -> ParametricSignal:
    settings = dict(case.signal)
    signal_type = settings.pop("type")
    if signal_type == "ou":
        return OrnsteinUhlenbeckSignal(**settings)
    return Heat1DFEMSignal(**settings)


def _make_observation(
    case: CaseConfig,
) -> GaussianObservation | PoissonObservation:
    settings = dict(case.observation)
    observation_type = settings.pop("type")
    map_name = settings.pop("map")
    evaluation_rule = settings.pop("evaluation_rule", "right")
    observation_std = settings.pop("observation_std", None)

    if map_name == "linear":
        mean_function = linear_observation(**settings)
    elif map_name == "sigmoid":
        mean_function = sigmoid_observation(**settings)
    elif map_name == "positive_sigmoid":
        mean_function = positive_sigmoid_intensity(**settings)
    elif map_name == "sensor_sigmoid":
        mean_function = sensor_sigmoid_observation(**settings)
    else:
        raise ValueError(f"case '{case.name}' has an unknown observation map")

    if observation_type == "gaussian":
        return GaussianObservation(
            mean_function,
            observation_std=float(observation_std),
            evaluation_rule=evaluation_rule,
        )
    if observation_std is not None:
        raise ValueError("Poisson observations do not use observation_std")
    return PoissonObservation(mean_function, evaluation_rule=evaluation_rule)


def _simulate(case: CaseConfig, master_seed: int, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    signal = _make_signal(case)
    observation = _make_observation(case)
    state = signal.sample_initial(
        1,
        seeded_rng(master_seed, case.name, "replicate", replicate, "signal_initial"),
    )
    truth = np.empty((case.steps + 1, *signal.state_shape))
    truth[0] = state[0]
    increments: list[np.ndarray] = []

    for index in range(case.steps):
        if observation.evaluation_rule == "left":
            observed_state = state
            observed_time = index * case.dt
        normal = seeded_rng(
            master_seed,
            case.name,
            "replicate",
            replicate,
            "signal",
            "timestep",
            index,
        ).standard_normal(state.shape)
        state = signal.step(state, case.theta_true, case.dt, normal).state
        truth[index + 1] = state[0]
        if observation.evaluation_rule == "right":
            observed_state = state
            observed_time = (index + 1) * case.dt

        observation_rng = seeded_rng(
            master_seed,
            case.name,
            "replicate",
            replicate,
            "observation",
            "timestep",
            index,
        )
        if isinstance(observation, GaussianObservation):
            observation_normal = observation_rng.standard_normal(
                observation.mean(observed_state, observed_time).shape
            )
            increment = observation.sample_increment(
                observed_state,
                observed_time,
                case.dt,
                observation_normal,
            )
        else:
            increment = observation.sample_increment(
                observed_state,
                observed_time,
                case.dt,
                observation_rng,
            )
        increments.append(increment[0])
    return truth, np.asarray(increments)


def _horizon_indices(steps: int, count: int = 201) -> np.ndarray:
    return np.unique(np.linspace(0, steps, min(count, steps + 1), dtype=int))


def _filter_seed(master_seed: int, case_name: str, replicate: int) -> int:
    rng = seeded_rng(master_seed, case_name, "replicate", replicate, "filter")
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _grid_replicate(
    case: CaseConfig,
    master_seed: int,
    replicate: int,
    output_dir: Path,
) -> dict[str, Any]:
    truth, observed = _simulate(case, master_seed, replicate)
    signal = _make_signal(case)
    observation = _make_observation(case)
    seed = _filter_seed(master_seed, case.name, replicate)
    theta_grid = case.theta_grid
    score_surface = np.empty((theta_grid.size, case.steps + 1))
    loglik_surface = np.empty_like(score_surface)

    for candidate, theta in enumerate(theta_grid):
        result = run_augmented_filter(
            signal,
            observation,
            observed,
            float(theta),
            case.ensemble_size,
            case.dt,
            seed,
        )
        score_surface[candidate] = result.score_mean
        loglik_surface[candidate] = result.log_likelihood

    theta_path, root_status, root_bracket = GridScoreRootEstimator().estimate_path(
        theta_grid,
        score_surface,
        initial_value=float(theta_grid.mean()),
    )
    reference = run_augmented_filter(
        signal,
        observation,
        observed,
        case.theta_true,
        case.ensemble_size,
        case.dt,
        seed,
    )
    horizons = _horizon_indices(case.steps)
    path = output_dir / "replicates" / f"replicate_{replicate:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=reference.time[horizons],
        theta_hat=theta_path[horizons],
        root_status=root_status[horizons],
        root_bracket=root_bracket[horizons],
        score_theta_true=reference.score_mean[horizons],
        observed_information=reference.observed_information[horizons],
        predictable_information=reference.predictable_information[horizons],
        theta_grid=theta_grid,
        score_surface=score_surface[:, horizons],
        log_likelihood_surface=loglik_surface[:, horizons],
        truth=truth if replicate == 0 else np.empty(0),
        observations=observed if replicate == 0 else np.empty(0),
    )
    final = float(theta_path[-1])
    info = float(reference.observed_information[-1])
    normalized = np.sqrt(max(info, 0.0)) * (final - case.theta_true)
    return {
        "case": case.name,
        "replicate": replicate,
        "theta_true": case.theta_true,
        "theta_final": final,
        "absolute_error": abs(final - case.theta_true),
        "observed_information": info,
        "predictable_information": float(reference.predictable_information[-1]),
        "normalized_error": normalized,
        "root_status": str(root_status[-1]),
    }


def _recursive_replicate(
    case: CaseConfig,
    master_seed: int,
    replicate: int,
    output_dir: Path,
) -> dict[str, Any]:
    truth, observed = _simulate(case, master_seed, replicate)
    signal = _make_signal(case)
    observation = _make_observation(case)
    if not isinstance(observation, GaussianObservation):
        raise ValueError("the recursive estimator currently requires Gaussian observations")
    estimator = RecursiveLouisEstimator(
        theta=float(case.initial_theta),
        lower=case.theta_min,
        upper=case.theta_max,
        step_size=case.recursive_step_size,
        information_floor=case.recursive_information_floor,
        step_clip=case.recursive_step_clip,
    )
    result = run_recursive_filter(
        signal,
        observation,
        observed,
        estimator,
        case.ensemble_size,
        case.dt,
        _filter_seed(master_seed, case.name, replicate),
    )
    horizons = _horizon_indices(case.steps)
    path = output_dir / "replicates" / f"replicate_{replicate:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=result.filter.time[horizons],
        theta_hat=result.theta[horizons],
        score_theta_true=result.filter.score_mean[horizons],
        observed_information=result.filter.observed_information[horizons],
        predictable_information=result.filter.predictable_information[horizons],
        truth=truth if replicate == 0 else np.empty(0),
        observations=observed if replicate == 0 else np.empty(0),
    )
    final = float(result.theta[-1])
    info = float(result.filter.observed_information[-1])
    normalized = np.sqrt(max(info, 0.0)) * (final - case.theta_true)
    return {
        "case": case.name,
        "replicate": replicate,
        "theta_true": case.theta_true,
        "theta_final": final,
        "absolute_error": abs(final - case.theta_true),
        "observed_information": info,
        "predictable_information": float(result.filter.predictable_information[-1]),
        "normalized_error": normalized,
        "root_status": "recursive",
    }


def _run_replicate(args: tuple[CaseConfig, int, int, str]) -> dict[str, Any]:
    case, master_seed, replicate, output = args
    output_dir = Path(output)
    if case.estimator == "recursive":
        return _recursive_replicate(case, master_seed, replicate, output_dir)
    return _grid_replicate(case, master_seed, replicate, output_dir)


def _observation_conventions(config: ExperimentConfig) -> dict[str, Any]:
    gaussian: dict[str, Any] = {}
    for case in config.cases:
        if case.observation["type"] != "gaussian":
            continue
        standard_deviation = float(case.observation["observation_std"])
        gaussian[case.name] = {
            "observation_std": standard_deviation,
            "observation_variance": standard_deviation**2,
            "increment_variance": standard_deviation**2 * case.dt,
        }
    return {
        "gaussian_model": "dY_t = h(X_t) dt + observation_std dV_t",
        "gaussian_cases": gaussian,
        "heat_state_noise": "state_noise_variance is a variance; its amplitude is its square root",
    }


def _software_versions() -> dict[str, str]:
    packages = ["numpy", "scipy", "PyYAML", "matplotlib", "likelihood-filtering"]
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source tree"
    return versions


def run_experiment(
    config: ExperimentConfig,
    output_dir: str | Path,
    workers: int = 1,
) -> Path:
    """Run all configured cases and write their artifacts."""
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": config.to_dict(),
        "conventions": _observation_conventions(config),
        "software": _software_versions(),
        "seed_policy": {
            "generator": "numpy.random.SeedSequence",
            "coordinates": [
                "case",
                "replicate",
                "signal or observation",
                "timestep",
                "ensemble member",
                "parameter candidate",
            ],
            "candidate_parameters": (
                "candidate identity is recorded, but noise coordinates omit it "
                "to give common random numbers"
            ),
            "parallelism": "independent replicates only",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    for case in config.cases:
        case_dir = output_dir / case.name
        tasks = [
            (case, config.master_seed, replicate, str(case_dir))
            for replicate in range(case.replicates)
        ]
        if workers == 1:
            rows = [_run_replicate(task) for task in tasks]
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(_run_replicate, tasks))
            except (NotImplementedError, PermissionError):
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(_run_replicate, tasks))
        write_metrics(case_dir / "metrics.csv", rows)
    summarize_experiment(output_dir)
    return output_dir


def summarize_experiment(output_dir: str | Path) -> list[dict[str, Any]]:
    """Build compact aggregate arrays and summary metrics."""
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / "manifest.json")
    summaries: list[dict[str, Any]] = []
    for case_data in manifest["experiment"]["cases"]:
        case_name = str(case_data["name"])
        case_dir = output_dir / case_name
        replicate_paths = sorted((case_dir / "replicates").glob("replicate_*.npz"))
        if not replicate_paths:
            raise FileNotFoundError(f"no replicate artifacts for {case_name}")
        loaded = [np.load(path, allow_pickle=False) for path in replicate_paths]
        time = loaded[0]["time"]
        theta = np.stack([item["theta_hat"] for item in loaded])
        observed = np.stack([item["observed_information"] for item in loaded])
        predictable = np.stack([item["predictable_information"] for item in loaded])
        np.savez_compressed(
            case_dir / "aggregates.npz",
            time=time,
            theta_mean=theta.mean(axis=0),
            theta_quantiles=np.quantile(theta, [0.05, 0.5, 0.95], axis=0),
            observed_information_mean=observed.mean(axis=0),
            predictable_information_mean=predictable.mean(axis=0),
            theta_final=theta[:, -1],
        )
        for item in loaded:
            item.close()

        metrics = read_metrics(case_dir / "metrics.csv")
        final_values = np.asarray([float(row["theta_final"]) for row in metrics])
        errors = np.asarray([float(row["absolute_error"]) for row in metrics])
        summaries.append(
            {
                "case": case_name,
                "replicates": len(metrics),
                "theta_true": float(metrics[0]["theta_true"]),
                "theta_mean": float(final_values.mean()),
                "theta_std": float(final_values.std()),
                "mean_absolute_error": float(errors.mean()),
            }
        )
    write_metrics(output_dir / "summary.csv", summaries)
    return summaries


def select_cases(config: ExperimentConfig, names: list[str]) -> ExperimentConfig:
    """Return a configuration containing the named cases."""
    selected = [replace(case) for case in config.cases if case.name in names]
    missing = set(names) - {case.name for case in selected}
    if missing:
        raise ValueError(f"unknown cases: {', '.join(sorted(missing))}")
    return ExperimentConfig(config.name, config.master_seed, selected)
