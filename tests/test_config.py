from pathlib import Path

import pytest
import yaml

from likelihood_filtering.config import ExperimentConfig


def _write_config(path: Path, observation: dict[str, object]) -> None:
    data = {
        "name": "test",
        "master_seed": 4,
        "cases": [
            {
                "name": "case",
                "signal": {"type": "ou"},
                "observation": observation,
                "time": {"dt": 0.01, "steps": 2},
                "replicates": 1,
                "filter": {"ensemble_size": 4},
                "estimation": {
                    "theta_true": 0.37,
                    "theta_min": 0.1,
                    "theta_max": 1.0,
                    "theta_points": 3,
                },
            }
        ],
    }
    path.write_text(yaml.safe_dump(data))


@pytest.mark.parametrize(
    "field",
    [
        "gamma",
        "noise_coefficient",
        "observation_variance",
        "observation_covariance",
        "obs_std",
        "standard_deviation",
    ],
)
def test_config_rejects_ambiguous_gaussian_fields(tmp_path: Path, field: str) -> None:
    observation = {
        "type": "gaussian",
        "map": "linear",
        "coefficient": 2.0,
        "observation_std": 0.068526,
        field: 0.1,
    }
    path = tmp_path / "bad.yaml"
    _write_config(path, observation)
    with pytest.raises(ValueError, match="ambiguous Gaussian fields"):
        ExperimentConfig.from_yaml(path)


def test_thesis_standard_deviations() -> None:
    root = Path(__file__).parents[1]
    config = ExperimentConfig.from_yaml(root / "configs" / "thesis.yaml")
    cases = {case.name: case for case in config.cases}
    assert cases["ou_linear_gaussian"].observation["observation_std"] == 0.068526
    assert cases["ou_nonlinear_gaussian"].observation["observation_std"] == 0.0117758
    assert cases["heat_nonlinear_gaussian"].signal["state_noise_variance"] == 0.015
