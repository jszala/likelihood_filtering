from dataclasses import replace

import numpy as np

from likelihood_filtering.config import ExperimentConfig
from likelihood_filtering.experiment import run_experiment
from likelihood_filtering.persistence import read_json
from likelihood_filtering.plotting import plot_experiment


def _tiny_config() -> ExperimentConfig:
    config = ExperimentConfig.from_yaml("configs/quick.yaml")
    case = replace(
        config.cases[0],
        steps=5,
        replicates=2,
        ensemble_size=6,
        theta_points=3,
    )
    return ExperimentConfig("tiny", 77, [case])


def _tiny_all_cases() -> ExperimentConfig:
    config = ExperimentConfig.from_yaml("configs/quick.yaml")
    cases = [
        replace(case, steps=2, replicates=1, ensemble_size=6, theta_points=3)
        for case in config.cases
    ]
    return ExperimentConfig("tiny-all", 78, cases)


def test_quick_experiment_artifacts_and_manifest(tmp_path) -> None:
    output = run_experiment(_tiny_config(), tmp_path / "run")
    manifest = read_json(output / "manifest.json")
    convention = manifest["conventions"]["gaussian_cases"]["ou_linear_gaussian"]
    standard_deviation = 0.068526
    assert convention["observation_std"] == standard_deviation
    assert convention["observation_variance"] == standard_deviation**2
    assert convention["increment_variance"] == standard_deviation**2 * 0.01
    assert (output / "summary.csv").is_file()
    assert (output / "ou_linear_gaussian" / "aggregates.npz").is_file()
    written = plot_experiment(output)
    assert written
    assert all(path.is_file() for path in written)


def test_worker_count_does_not_change_results(tmp_path) -> None:
    config = _tiny_config()
    first = run_experiment(config, tmp_path / "one", workers=1)
    second = run_experiment(config, tmp_path / "two", workers=2)
    for replicate in range(2):
        name = f"replicates/replicate_{replicate:04d}.npz"
        with (
            np.load(first / "ou_linear_gaussian" / name) as left,
            np.load(second / "ou_linear_gaussian" / name) as right,
        ):
            assert set(left.files) == set(right.files)
            for field in left.files:
                np.testing.assert_array_equal(left[field], right[field])


def test_all_quick_cases_run_end_to_end(tmp_path) -> None:
    config = _tiny_all_cases()
    output = run_experiment(config, tmp_path / "all")
    for case in config.cases:
        assert (output / case.name / "metrics.csv").is_file()
        assert (output / case.name / "aggregates.npz").is_file()
