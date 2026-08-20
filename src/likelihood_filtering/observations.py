from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.special import expit

Array = NDArray[np.float64]
ObservationFunction = Callable[[Array, float], Array]


def linear_observation(coefficient: float) -> ObservationFunction:
    def evaluate(state: Array, time: float) -> Array:
        del time
        return coefficient * np.asarray(state, dtype=float)

    return evaluate


def sigmoid_observation(coefficient: float, scale: float, shift: float) -> ObservationFunction:
    def evaluate(state: Array, time: float) -> Array:
        del time
        return coefficient * expit(scale * (np.asarray(state, dtype=float) - shift))

    return evaluate


def positive_sigmoid_intensity(
    coefficient: float,
    scale: float,
    shift: float,
    floor: float,
) -> ObservationFunction:
    if floor <= 0:
        raise ValueError("Poisson intensity floor must be positive")

    def evaluate(state: Array, time: float) -> Array:
        del time
        return floor + coefficient * expit(scale * (np.asarray(state, dtype=float) - shift))

    return evaluate


def sensor_matrix_1d_fem(
    sensor_locations: list[float],
    dimensions: int,
    domain_length: float,
) -> csr_matrix:
    if dimensions < 1 or domain_length <= 0:
        raise ValueError("invalid finite-element domain")
    if not sensor_locations:
        raise ValueError("at least one sensor is required")

    h = domain_length / (dimensions + 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, location in enumerate(sensor_locations):
        if not 0 < location < domain_length:
            raise ValueError("sensor locations must lie inside the domain")
        scaled = location / h
        left = int(np.floor(scaled))
        right = left + 1
        right_weight = scaled - left
        left_weight = 1.0 - right_weight
        if 1 <= left <= dimensions:
            rows.append(row)
            cols.append(left - 1)
            data.append(left_weight)
        if 1 <= right <= dimensions:
            rows.append(row)
            cols.append(right - 1)
            data.append(right_weight)
    return csr_matrix((data, (rows, cols)), shape=(len(sensor_locations), dimensions))


def sensor_sigmoid_observation(
    sensor_locations: list[float],
    dimensions: int,
    domain_length: float,
    coefficient: float,
    scale: float,
    shift: float,
) -> ObservationFunction:
    matrix = sensor_matrix_1d_fem(sensor_locations, dimensions, domain_length)

    def evaluate(state: Array, time: float) -> Array:
        del time
        sensor_values = np.asarray(state, dtype=float) @ matrix.T
        return coefficient * expit(scale * (sensor_values - shift))

    return evaluate


def _as_observation_array(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("observation functions must return (ensemble, observation_dim)")
    return values


@dataclass
class GaussianObservation:
    mean_function: ObservationFunction
    observation_std: float | None = None
    observation_covariance: Array | None = None
    evaluation_rule: str = "right"

    def __post_init__(self) -> None:
        if (self.observation_std is None) == (self.observation_covariance is None):
            raise ValueError("set exactly one of observation_std or observation_covariance")
        if self.observation_std is not None and self.observation_std <= 0:
            raise ValueError("observation_std must be positive")
        if self.evaluation_rule not in {"left", "right"}:
            raise ValueError("evaluation_rule must be 'left' or 'right'")
        if self.observation_covariance is not None:
            covariance = np.asarray(self.observation_covariance, dtype=float)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("observation_covariance must be square")
            if not np.allclose(covariance, covariance.T):
                raise ValueError("observation_covariance must be symmetric")
            np.linalg.cholesky(covariance)
            self.observation_covariance = covariance

    def mean(self, state: Array, time: float) -> Array:
        return _as_observation_array(self.mean_function(state, time))

    def covariance(self, observation_dim: int) -> Array:
        if self.observation_std is not None:
            return self.observation_std**2 * np.eye(observation_dim)
        covariance = np.asarray(self.observation_covariance, dtype=float)
        if covariance.shape != (observation_dim, observation_dim):
            raise ValueError("observation_covariance has the wrong dimension")
        return covariance

    def increment_covariance(self, dt: float, observation_dim: int) -> Array:
        return dt * self.covariance(observation_dim)

    def sample_increment(self, state: Array, time: float, dt: float, normal: Array) -> Array:
        mean = self.mean(state, time) * dt
        normal = _as_observation_array(normal)
        if normal.shape != mean.shape:
            raise ValueError("normal has the wrong observation shape")
        chol = np.linalg.cholesky(self.increment_covariance(dt, mean.shape[1]))
        return mean + normal @ chol.T


@dataclass
class PoissonObservation:
    intensity_function: ObservationFunction
    evaluation_rule: str = "left"

    def __post_init__(self) -> None:
        if self.evaluation_rule not in {"left", "right"}:
            raise ValueError("evaluation_rule must be 'left' or 'right'")

    def intensity(self, state: Array, time: float) -> Array:
        intensity = _as_observation_array(self.intensity_function(state, time))
        if np.any(~np.isfinite(intensity)) or np.any(intensity <= 0):
            raise ValueError("Poisson intensity must be finite and positive")
        return intensity

    def sample_increment(
        self,
        state: Array,
        time: float,
        dt: float,
        rng: np.random.Generator,
    ) -> Array:
        return rng.poisson(self.intensity(state, time) * dt).astype(float)
