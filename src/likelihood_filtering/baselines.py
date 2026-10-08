"""Independent likelihood benchmarks for the scalar OU examples.

These estimators use the observation density directly. They do not use the
augmented EnKF score or its Gaussian approximation for Poisson observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from .observations import GaussianObservation, PoissonObservation
from .randomness import seeded_rng
from .signals import OrnsteinUhlenbeckSignal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LikelihoodEstimate:
    theta: float
    log_likelihood: float
    status: str


def _increments(values: Array) -> Array:
    observed = np.asarray(values, dtype=float)
    if observed.ndim == 1:
        observed = observed[:, None]
    if observed.ndim != 2 or observed.shape[0] == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("observations must be a nonempty finite (time, dimension) array")
    return observed


def kalman_log_likelihood(
    observed_increments: Array,
    theta: float,
    signal: OrnsteinUhlenbeckSignal,
    observation_coefficient: float,
    observation_std: float,
    dt: float,
) -> float:
    """Exact Gaussian likelihood for the repository's Euler/right-endpoint OU model."""
    observed = _increments(observed_increments)
    if signal.dimensions != 1 or observed.shape[1] != 1:
        raise ValueError("the Kalman benchmark requires a scalar signal and observation")
    if dt <= 0 or observation_std <= 0 or theta <= 0:
        raise ValueError("dt, observation_std, and theta must be positive")

    transition = 1.0 - theta * signal.mu * dt
    process_variance = signal.state_diffusion**2 * dt
    measurement = observation_coefficient * dt
    measurement_variance = observation_std**2 * dt
    mean = signal.initial_mean
    variance = signal.initial_std**2
    log_likelihood = 0.0

    for value in observed[:, 0]:
        mean = transition * mean
        variance = transition**2 * variance + process_variance
        innovation_variance = measurement**2 * variance + measurement_variance
        innovation = value - measurement * mean
        log_likelihood -= 0.5 * (
            np.log(2.0 * np.pi * innovation_variance) + innovation**2 / innovation_variance
        )
        gain = variance * measurement / innovation_variance
        mean += gain * innovation
        variance *= 1.0 - gain * measurement
    return float(log_likelihood)


def kalman_mle(
    observed_increments: Array,
    signal: OrnsteinUhlenbeckSignal,
    observation_coefficient: float,
    observation_std: float,
    dt: float,
    bounds: tuple[float, float],
) -> LikelihoodEstimate:
    """Maximize the exact discrete-model likelihood on the stated parameter interval."""
    lower, upper = bounds
    if not 0 < lower < upper:
        raise ValueError("bounds must be positive and increasing")

    def objective(theta: float) -> float:
        return -kalman_log_likelihood(
            observed_increments, theta, signal, observation_coefficient, observation_std, dt
        )

    optimum = minimize_scalar(objective, method="bounded", bounds=bounds, options={"xatol": 1e-5})
    candidates = [(lower, objective(lower)), (upper, objective(upper))]
    if optimum.success and np.isfinite(optimum.fun):
        candidates.append((float(optimum.x), float(optimum.fun)))
    theta, negative_log_likelihood = min(candidates, key=lambda item: item[1])
    status = "boundary" if theta in bounds else "interior"
    return LikelihoodEstimate(float(theta), -float(negative_log_likelihood), status)


def _log_observation_density(
    observed: Array,
    state: Array,
    time: float,
    dt: float,
    observation: GaussianObservation | PoissonObservation,
) -> Array:
    if isinstance(observation, GaussianObservation):
        predicted = observation.mean(state, time) * dt
        covariance = observation.increment_covariance(dt, observed.size)
        residual = predicted - observed[None, :]
        solved = np.linalg.solve(covariance, residual.T).T
        _, logdet = np.linalg.slogdet(covariance)
        return -0.5 * (
            observed.size * np.log(2.0 * np.pi) + logdet + np.sum(residual * solved, axis=1)
        )
    mean_count = observation.intensity(state, time) * dt
    return np.sum(
        observed[None, :] * np.log(mean_count) - mean_count - gammaln(observed[None, :] + 1.0),
        axis=1,
    )


def systematic_resample(weights: Array, offset: float) -> NDArray[np.intp]:
    """Return systematic ancestor indices; offset is uniform on [0, 1)."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0 or not 0 <= offset < 1:
        raise ValueError("invalid weights or offset")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    points = (offset + np.arange(weights.size)) / weights.size
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, points, side="right")


def particle_log_likelihood(
    observed_increments: Array,
    theta: float,
    signal: OrnsteinUhlenbeckSignal,
    observation: GaussianObservation | PoissonObservation,
    dt: float,
    particles: int,
    seed: int,
) -> float:
    """Bootstrap filter likelihood with exact per-particle observation densities."""
    observed = _increments(observed_increments)
    if signal.dimensions != 1 or observed.shape[1] != 1:
        raise ValueError("the particle benchmark requires a scalar signal and observation")
    if theta <= 0 or dt <= 0 or particles < 2:
        raise ValueError("theta and dt must be positive; particles must be at least two")
    if isinstance(observation, PoissonObservation) and (
        np.any(observed < 0) or np.any(observed != np.floor(observed))
    ):
        raise ValueError("Poisson observations must be nonnegative integer counts")

    state = signal.sample_initial(particles, seeded_rng(seed, "particle_initial"))
    log_weights = np.full(particles, -np.log(particles))
    total = 0.0
    for index, value in enumerate(observed):
        if observation.evaluation_rule == "right":
            normal = seeded_rng(seed, "particle_forecast", index).standard_normal(state.shape)
            state = signal.step(state, theta, dt, normal).state
            time = (index + 1) * dt
        else:
            time = index * dt

        log_weights += _log_observation_density(value, state, time, dt, observation)
        increment = float(logsumexp(log_weights))
        if not np.isfinite(increment):
            return float("-inf")
        total += increment
        log_weights -= increment
        weights = np.exp(log_weights)
        effective_size = 1.0 / np.sum(weights**2)
        if effective_size < particles / 2.0:
            offset = float(seeded_rng(seed, "particle_resample", index).random())
            state = state[systematic_resample(weights, offset)]
            log_weights.fill(-np.log(particles))

        if observation.evaluation_rule == "left":
            normal = seeded_rng(seed, "particle_forecast", index).standard_normal(state.shape)
            state = signal.step(state, theta, dt, normal).state
    return total


def particle_grid_mle(
    observed_increments: Array,
    theta_grid: Array,
    signal: OrnsteinUhlenbeckSignal,
    observation: GaussianObservation | PoissonObservation,
    dt: float,
    particles: int,
    seed: int,
) -> LikelihoodEstimate:
    grid = np.asarray(theta_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("theta grid must be strictly increasing")
    observed = _increments(observed_increments)
    if signal.dimensions != 1 or observed.shape[1] != 1:
        raise ValueError("the particle benchmark requires a scalar signal and observation")
    if grid[0] <= 0 or dt <= 0 or particles < 2:
        raise ValueError("theta, dt, and particles must be positive")
    if isinstance(observation, PoissonObservation) and (
        np.any(observed < 0) or np.any(observed != np.floor(observed))
    ):
        raise ValueError("Poisson observations must be nonnegative integer counts")

    # All parameter candidates share forecast noise and resampling offsets.
    # Advancing the grid together avoids one Python filter loop per candidate.
    state = signal.sample_initial(particles, seeded_rng(seed, "particle_initial"))[:, 0]
    state = np.broadcast_to(state, (grid.size, particles)).copy()
    log_weights = np.full((grid.size, particles), -np.log(particles))
    totals = np.zeros(grid.size)
    noise_scale = signal.state_diffusion * np.sqrt(dt)
    gaussian_variance = (
        float(observation.increment_covariance(dt, 1)[0, 0])
        if isinstance(observation, GaussianObservation)
        else None
    )

    for index, value in enumerate(observed[:, 0]):
        if observation.evaluation_rule == "right":
            normal = seeded_rng(seed, "particle_forecast", index).standard_normal(particles)
            state = state - grid[:, None] * signal.mu * state * dt + noise_scale * normal
            time = (index + 1) * dt
        else:
            time = index * dt

        if isinstance(observation, GaussianObservation):
            predicted = observation.mean(state.reshape(-1, 1), time).reshape(state.shape) * dt
            residual = predicted - value
            log_density = -0.5 * (
                np.log(2.0 * np.pi * gaussian_variance) + residual**2 / gaussian_variance
            )
        else:
            predicted = observation.intensity(state.reshape(-1, 1), time).reshape(state.shape) * dt
            log_density = value * np.log(predicted) - predicted - gammaln(value + 1.0)

        log_weights += log_density
        increments = logsumexp(log_weights, axis=1)
        if not np.all(np.isfinite(increments)):
            raise FloatingPointError("particle likelihood became nonfinite")
        totals += increments
        log_weights -= increments[:, None]
        weights = np.exp(log_weights)
        effective_size = 1.0 / np.sum(weights**2, axis=1)
        resample_rows = np.flatnonzero(effective_size < particles / 2.0)
        if resample_rows.size:
            offset = float(seeded_rng(seed, "particle_resample", index).random())
            for row in resample_rows:
                state[row] = state[row, systematic_resample(weights[row], offset)]
                log_weights[row].fill(-np.log(particles))

        if observation.evaluation_rule == "left":
            normal = seeded_rng(seed, "particle_forecast", index).standard_normal(particles)
            state = state - grid[:, None] * signal.mu * state * dt + noise_scale * normal

    values = totals
    if not np.any(np.isfinite(values)):
        return LikelihoodEstimate(float("nan"), float("-inf"), "no_finite_likelihood")
    index = int(np.argmax(values))
    status = "boundary" if index in {0, grid.size - 1} else "interior"
    return LikelihoodEstimate(float(grid[index]), float(values[index]), status)
