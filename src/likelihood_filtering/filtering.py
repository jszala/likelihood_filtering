from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from .diagnostics import fisher_louis_information, running_quadratic_variation
from .enkf import ensemble_update, poisson_ensemble_update
from .experimental import RecursiveLouisEstimator
from .observations import GaussianObservation, PoissonObservation
from .randomness import seeded_rng
from .signals import ParametricSignal

Array = NDArray[np.float64]


@dataclass
class FilterResult:
    time: Array
    state_mean: Array
    state_variance: Array
    score_mean: Array
    score_variance: Array
    complete_information_mean: Array
    observed_information: Array
    predictable_information: Array
    log_likelihood: Array


@dataclass
class RecursiveFilterResult:
    filter: FilterResult
    theta: Array


def _join(state: Array, score: Array, information: Array) -> Array:
    return np.column_stack((state.reshape(state.shape[0], -1), score, information))


def _split(augmented: Array, state_shape: tuple[int, ...]) -> tuple[Array, Array, Array]:
    state_size = int(np.prod(state_shape))
    state = augmented[:, :state_size].reshape((augmented.shape[0], *state_shape))
    return state, augmented[:, state_size], augmented[:, state_size + 1]


def _gaussian_log_likelihood(observed: Array, predicted: Array, covariance: Array) -> float:
    mean = predicted.mean(axis=0)
    anomaly = predicted - mean
    if predicted.shape[0] > 1:
        covariance = covariance + anomaly.T @ anomaly / (predicted.shape[0] - 1)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("nan")
    residual = observed - mean
    quadratic = residual @ np.linalg.solve(covariance, residual)
    return float(-0.5 * (observed.size * np.log(2 * np.pi) + logdet + quadratic))


def _poisson_log_likelihood(observed: Array, predicted_count: Array) -> float:
    mean = np.maximum(predicted_count.mean(axis=0), 1e-12)
    return float(np.sum(observed * np.log(mean) - mean - gammaln(observed + 1)))


def run_augmented_filter(
    signal: ParametricSignal,
    observation: GaussianObservation | PoissonObservation,
    observed_increments: Array,
    theta: float,
    ensemble_size: int,
    dt: float,
    seed: int,
) -> FilterResult:
    """Run one augmented EnKF at a fixed parameter value."""
    observed_increments = np.asarray(observed_increments, dtype=float)
    if observed_increments.ndim == 1:
        observed_increments = observed_increments[:, None]
    if observed_increments.ndim != 2:
        raise ValueError("observed_increments must have shape (time, observation_dim)")
    if ensemble_size < 2 or dt <= 0:
        raise ValueError("invalid ensemble size or timestep")

    state = signal.sample_initial(ensemble_size, seeded_rng(seed, "filter_initial"))
    score = np.zeros(ensemble_size)
    information = np.zeros(ensemble_size)
    steps, observation_dim = observed_increments.shape

    state_mean = np.empty((steps + 1, *signal.state_shape))
    state_variance = np.empty_like(state_mean)
    score_mean = np.empty(steps + 1)
    score_variance = np.empty(steps + 1)
    information_mean = np.empty(steps + 1)
    observed_information = np.empty(steps + 1)
    log_likelihood = np.zeros(steps + 1)

    def record(index: int) -> None:
        state_mean[index] = state.mean(axis=0)
        state_variance[index] = state.var(axis=0)
        score_mean[index] = score.mean()
        score_variance[index] = score.var()
        information_mean[index] = information.mean()
        observed_information[index] = fisher_louis_information(information, score)

    def analyse(index: int, time: float) -> None:
        nonlocal state, score, information
        augmented = _join(state, score, information)
        normal = seeded_rng(seed, "analysis", index).standard_normal(
            (ensemble_size, observation_dim)
        )
        if isinstance(observation, GaussianObservation):
            predicted = observation.mean(state, time) * dt
            covariance = observation.increment_covariance(dt, observation_dim)
            log_likelihood[index + 1] = log_likelihood[index] + _gaussian_log_likelihood(
                observed_increments[index], predicted, covariance
            )
            augmented = ensemble_update(
                augmented,
                predicted,
                observed_increments[index],
                covariance,
                normal,
            )
        else:
            predicted = observation.intensity(state, time) * dt
            log_likelihood[index + 1] = log_likelihood[index] + _poisson_log_likelihood(
                observed_increments[index], predicted
            )
            augmented = poisson_ensemble_update(
                augmented,
                predicted,
                observed_increments[index],
                normal,
            )
        state, score, information = _split(augmented, signal.state_shape)

    record(0)
    for index in range(steps):
        forecast_normal = seeded_rng(seed, "forecast", index).standard_normal(state.shape)
        if observation.evaluation_rule == "left":
            analyse(index, index * dt)
            step = signal.step(state, theta, dt, forecast_normal)
            state = step.state
            score += step.score_increment
            information += step.information_increment
        else:
            step = signal.step(state, theta, dt, forecast_normal)
            state = step.state
            score += step.score_increment
            information += step.information_increment
            analyse(index, (index + 1) * dt)
        record(index + 1)

    return FilterResult(
        time=np.arange(steps + 1) * dt,
        state_mean=state_mean,
        state_variance=state_variance,
        score_mean=score_mean,
        score_variance=score_variance,
        complete_information_mean=information_mean,
        observed_information=observed_information,
        predictable_information=running_quadratic_variation(score_mean),
        log_likelihood=log_likelihood,
    )


def run_recursive_filter(
    signal: ParametricSignal,
    observation: GaussianObservation,
    observed_increments: Array,
    estimator: RecursiveLouisEstimator,
    ensemble_size: int,
    dt: float,
    seed: int,
) -> RecursiveFilterResult:
    """Run the experimental single-filter recursive estimator."""
    observed_increments = np.asarray(observed_increments, dtype=float)
    if observed_increments.ndim == 1:
        observed_increments = observed_increments[:, None]
    if observation.evaluation_rule != "right":
        raise ValueError("the recursive implementation uses right-endpoint observations")

    state = signal.sample_initial(ensemble_size, seeded_rng(seed, "filter_initial"))
    score = np.zeros(ensemble_size)
    information = np.zeros(ensemble_size)
    steps, observation_dim = observed_increments.shape
    state_mean = np.empty((steps + 1, *signal.state_shape))
    state_variance = np.empty_like(state_mean)
    score_mean = np.empty(steps + 1)
    score_variance = np.empty(steps + 1)
    information_mean = np.empty(steps + 1)
    observed_information = np.empty(steps + 1)
    log_likelihood = np.zeros(steps + 1)
    theta_path = np.empty(steps + 1)

    def record(index: int) -> None:
        state_mean[index] = state.mean(axis=0)
        state_variance[index] = state.var(axis=0)
        score_mean[index] = score.mean()
        score_variance[index] = score.var()
        information_mean[index] = information.mean()
        observed_information[index] = fisher_louis_information(information, score)
        theta_path[index] = estimator.theta

    record(0)
    for index in range(steps):
        forecast_normal = seeded_rng(seed, "forecast", index).standard_normal(state.shape)
        step = signal.step(state, estimator.theta, dt, forecast_normal)
        state = step.state
        score += step.score_increment
        information += step.information_increment

        predicted = observation.mean(state, (index + 1) * dt) * dt
        covariance = observation.increment_covariance(dt, observation_dim)
        log_likelihood[index + 1] = log_likelihood[index] + _gaussian_log_likelihood(
            observed_increments[index], predicted, covariance
        )
        normal = seeded_rng(seed, "analysis", index).standard_normal(
            (ensemble_size, observation_dim)
        )
        augmented = ensemble_update(
            _join(state, score, information),
            predicted,
            observed_increments[index],
            covariance,
            normal,
        )
        state, score, information = _split(augmented, signal.state_shape)

        observed_info = fisher_louis_information(information, score)
        _, change = estimator.update(float(score.mean()), observed_info)
        score = estimator.transport_score(score, information, change)
        record(index + 1)

    result = FilterResult(
        time=np.arange(steps + 1) * dt,
        state_mean=state_mean,
        state_variance=state_variance,
        score_mean=score_mean,
        score_variance=score_variance,
        complete_information_mean=information_mean,
        observed_information=observed_information,
        predictable_information=running_quadratic_variation(score_mean),
        log_likelihood=log_likelihood,
    )
    return RecursiveFilterResult(result, theta_path)
