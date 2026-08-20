from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class CaseConfig:
    name: str
    signal: dict[str, Any]
    observation: dict[str, Any]
    dt: float
    steps: int
    replicates: int
    ensemble_size: int
    theta_true: float
    theta_min: float
    theta_max: float
    theta_points: int
    estimator: str = "grid"
    initial_theta: float | None = None
    recursive_step_size: float = 1.0
    recursive_information_floor: float = 1e-8
    recursive_step_clip: float | None = None

    @property
    def theta_grid(self) -> np.ndarray:
        return np.linspace(self.theta_min, self.theta_max, self.theta_points)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("case name must not be empty")
        if self.dt <= 0 or self.steps < 1 or self.replicates < 1:
            raise ValueError(f"case '{self.name}' has invalid time or replicate settings")
        if self.ensemble_size < 2:
            raise ValueError(f"case '{self.name}' needs at least two ensemble members")
        if self.theta_points < 2 or self.theta_min >= self.theta_max:
            raise ValueError(f"case '{self.name}' has an invalid theta grid")
        if not self.theta_min <= self.theta_true <= self.theta_max:
            raise ValueError(f"case '{self.name}' has theta_true outside the grid bounds")
        if self.estimator not in {"grid", "recursive"}:
            raise ValueError(f"case '{self.name}' has an unknown estimator")
        if self.estimator == "recursive" and self.initial_theta is None:
            raise ValueError(f"case '{self.name}' needs initial_theta")
        signal_type = self.signal.get("type")
        if signal_type not in {"ou", "heat_1d_fem"}:
            raise ValueError(f"case '{self.name}' has an unknown signal type")
        signal_fields = {
            "ou": {"type", "dimensions", "mu", "state_diffusion", "initial_mean", "initial_std"},
            "heat_1d_fem": {
                "type",
                "dimensions",
                "domain_length",
                "state_noise_variance",
                "initial_mean",
                "initial_std",
            },
        }
        unknown_signal = set(self.signal) - signal_fields[signal_type]
        if unknown_signal:
            fields = ", ".join(sorted(unknown_signal))
            raise ValueError(f"case '{self.name}' has unknown signal fields: {fields}")

        observation_type = self.observation.get("type")
        if observation_type not in {"gaussian", "poisson"}:
            raise ValueError(f"case '{self.name}' has an unknown observation type")
        map_name = self.observation.get("map")
        map_fields = {
            "linear": {"coefficient"},
            "sigmoid": {"coefficient", "scale", "shift"},
            "positive_sigmoid": {"coefficient", "scale", "shift", "floor"},
            "sensor_sigmoid": {
                "sensor_locations",
                "dimensions",
                "domain_length",
                "coefficient",
                "scale",
                "shift",
            },
        }
        if map_name not in map_fields:
            raise ValueError(f"case '{self.name}' has an unknown observation map")
        forbidden = {
            "gamma",
            "noise",
            "noise_coefficient",
            "observation_covariance",
            "observation_variance",
            "observation_noise_std",
            "obs_std",
            "sigma_obs",
            "standard_deviation",
            "std",
            "variance",
        }
        if observation_type == "gaussian":
            ambiguous = forbidden.intersection(self.observation)
            if ambiguous:
                fields = ", ".join(sorted(ambiguous))
                raise ValueError(f"case '{self.name}' uses ambiguous Gaussian fields: {fields}")
            if "observation_std" not in self.observation:
                raise ValueError(f"case '{self.name}' needs observation_std")
            if float(self.observation["observation_std"]) <= 0:
                raise ValueError(f"case '{self.name}' has nonpositive observation_std")
        elif ({"observation_std"} | forbidden).intersection(self.observation):
            raise ValueError(f"case '{self.name}' mixes Gaussian and Poisson settings")

        common_fields = {"type", "map", "evaluation_rule"}
        noise_fields = {"observation_std"} if observation_type == "gaussian" else set()
        unknown_observation = (
            set(self.observation) - common_fields - noise_fields - map_fields[map_name]
        )
        if unknown_observation:
            fields = ", ".join(sorted(unknown_observation))
            raise ValueError(f"case '{self.name}' has unknown observation fields: {fields}")
        if observation_type == "poisson" and map_name != "positive_sigmoid":
            raise ValueError(f"case '{self.name}' needs a positive Poisson intensity map")
        if self.estimator == "recursive":
            if observation_type != "gaussian":
                raise ValueError(f"case '{self.name}' needs Gaussian observations for recursion")
            if self.observation.get("evaluation_rule", "right") != "right":
                raise ValueError(f"case '{self.name}' needs right-endpoint recursion")


@dataclass
class ExperimentConfig:
    name: str
    master_seed: int
    cases: list[CaseConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("experiment configuration must be a mapping")
        if set(raw) - {"name", "master_seed", "cases"}:
            unknown = ", ".join(sorted(set(raw) - {"name", "master_seed", "cases"}))
            raise ValueError(f"unknown experiment fields: {unknown}")
        if not isinstance(raw.get("cases"), list) or not raw["cases"]:
            raise ValueError("experiment needs at least one case")

        cases: list[CaseConfig] = []
        for item in raw["cases"]:
            if not isinstance(item, dict):
                raise ValueError("each case must be a mapping")
            case = cls._parse_case(item)
            case.validate()
            cases.append(case)
        names = [case.name for case in cases]
        if len(names) != len(set(names)):
            raise ValueError("case names must be unique")
        return cls(str(raw.get("name", path.stem)), int(raw.get("master_seed", 0)), cases)

    @staticmethod
    def _parse_case(raw: dict[str, Any]) -> CaseConfig:
        allowed = {
            "name",
            "signal",
            "observation",
            "time",
            "replicates",
            "filter",
            "estimation",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown case fields: {', '.join(sorted(unknown))}")
        time = raw.get("time", {})
        filtering = raw.get("filter", {})
        estimation = raw.get("estimation", {})
        return CaseConfig(
            name=str(raw["name"]),
            signal=dict(raw["signal"]),
            observation=dict(raw["observation"]),
            dt=float(time["dt"]),
            steps=int(time["steps"]),
            replicates=int(raw["replicates"]),
            ensemble_size=int(filtering["ensemble_size"]),
            theta_true=float(estimation["theta_true"]),
            theta_min=float(estimation["theta_min"]),
            theta_max=float(estimation["theta_max"]),
            theta_points=int(estimation["theta_points"]),
            estimator=str(estimation.get("method", "grid")),
            initial_theta=(
                float(estimation["initial_theta"]) if "initial_theta" in estimation else None
            ),
            recursive_step_size=float(estimation.get("step_size", 1.0)),
            recursive_information_floor=float(estimation.get("information_floor", 1e-8)),
            recursive_step_clip=(
                float(estimation["step_clip"]) if "step_clip" in estimation else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "master_seed": self.master_seed,
            "cases": [asdict(case) for case in self.cases],
        }
