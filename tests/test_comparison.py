import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from likelihood_filtering.baselines import particle_grid_log_likelihoods, particle_log_likelihood
from likelihood_filtering.comparison import _score_checkpoints, run_comparison, summarize_comparison
from likelihood_filtering.config import ExperimentConfig
from likelihood_filtering.experiment import _filter_seed, _make_observation, _make_signal, _simulate
from likelihood_filtering.persistence import read_json, write_json


def _config() -> ExperimentConfig:
    config = ExperimentConfig.from_yaml("configs/comparison.yaml")
    return replace(
        config,
        name="test_comparison",
        cases=[replace(case, steps=8, replicates=2, ensemble_size=8) for case in config.cases],
    )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("case_index", [1, 2])
def test_particle_checkpoints_equal_independent_prefix_filters(case_index: int) -> None:
    case = _config().cases[case_index]
    _, observations = _simulate(case, 141, 0)
    grid = np.array([0.2, 0.5, 0.8])
    steps = (2, 5, 8)
    snapshots = particle_grid_log_likelihoods(
        observations, grid, _make_signal(case), _make_observation(case), case.dt, 32, 97, steps
    )
    for index, step in enumerate(steps):
        expected = [
            particle_log_likelihood(
                observations[:step],
                theta,
                _make_signal(case),
                _make_observation(case),
                case.dt,
                32,
                97,
            )
            for theta in grid
        ]
        np.testing.assert_allclose(snapshots[:, index], expected, atol=1e-10, rtol=0)
    # Later observations cannot enter an earlier likelihood.
    changed = observations.copy()
    changed[5:] += 1
    other = particle_grid_log_likelihoods(
        changed, grid, _make_signal(case), _make_observation(case), case.dt, 32, 97, steps
    )
    np.testing.assert_array_equal(snapshots[:, :2], other[:, :2])


def test_running_score_continuation_has_no_future_data() -> None:
    case = _config().cases[2]
    _, observations = _simulate(case, 141, 0)
    seed = _filter_seed(141, case.name, 0)
    full = _score_checkpoints(case, observations, seed, (2, 5, 8))
    prefix = _score_checkpoints(replace(case, steps=5), observations[:5], seed, (2, 5))
    np.testing.assert_array_equal(full[0][:2], prefix[0])
    np.testing.assert_array_equal(full[1][:2], prefix[1])


def test_comparison_pairing_resume_sensitivity_and_worker_invariance(tmp_path: Path) -> None:
    config = _config()
    settings = dict(
        checkpoints=(2, 5, 8),
        particles=32,
        check_particles=64,
        check_replicates=1,
        bootstrap_samples=1000,
    )
    summary = run_comparison(config, tmp_path, **settings)
    records = _csv(tmp_path / "estimates.csv")
    primary = [row for row in records if row["variant"] == "primary"]
    assert len(primary) == 36
    assert len(summary) == 18
    assert len(_csv(tmp_path / "sensitivity.csv")) == 27
    for case in config.cases:
        for replicate in range(2):
            paired = [
                row
                for row in records
                if row["case"] == case.name and int(row["replicate"]) == replicate
            ]
            assert len({row["observation_sha256"] for row in paired}) == 1
    assert run_comparison(config, tmp_path, resume=True, workers=2, **settings) == summary
    assert _csv(tmp_path / "estimates.csv") == records
    with pytest.raises(ValueError, match="changed"):
        run_comparison(config, tmp_path, resume=True, **{**settings, "particles": 48})
    parallel_path = tmp_path / "parallel"
    assert run_comparison(config, parallel_path, workers=2, **settings) == summary
    assert _csv(parallel_path / "estimates.csv") == records
    # Resume recomputes a missing record; completed artifacts are left intact.
    path = tmp_path / "records" / config.cases[1].name / "0000.json"
    path.unlink()
    assert run_comparison(config, tmp_path, resume=True, **settings) == summary
    assert _csv(tmp_path / "estimates.csv") == records
    # The summary agrees with independently computed paired errors.
    for row in summary:
        if row["method"] == "score_root":
            subset = [
                item
                for item in primary
                if item["case"] == row["case"] and int(item["step"]) == row["step"]
            ]
            score_errors = [
                float(item["absolute_error"]) for item in subset if item["method"] == "score_root"
            ]
            baseline_errors = [
                float(item["absolute_error"]) for item in subset if item["method"] != "score_root"
            ]
            assert row["delta_mae_vs_baseline"] == pytest.approx(
                np.mean(score_errors) - np.mean(baseline_errors)
            )
            assert row["family_ci_low"] <= row["paired_ci_low"]
            assert row["family_ci_high"] >= row["paired_ci_high"]
    artifact = read_json(path)
    artifact["rows"].append(artifact["rows"][0])
    write_json(path, artifact)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_comparison(tmp_path, config, (2, 5, 8), 1000)


@pytest.mark.parametrize("checkpoints", [(), (5, 2, 8), (2, 5), (0, 8)])
def test_rejects_invalid_checkpoint_design(tmp_path: Path, checkpoints: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="checkpoints"):
        run_comparison(_config(), tmp_path, checkpoints=checkpoints)
