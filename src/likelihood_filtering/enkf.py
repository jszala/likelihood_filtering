from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def ensemble_update(
    augmented: Array,
    predicted_increment: Array,
    observed_increment: Array,
    observation_covariance: Array,
    perturbation_normal: Array,
) -> Array:
    """Apply a stochastic ensemble Kalman update."""
    augmented = np.asarray(augmented, dtype=float)
    predicted_increment = np.asarray(predicted_increment, dtype=float)
    observed_increment = np.asarray(observed_increment, dtype=float).reshape(-1)
    covariance = np.asarray(observation_covariance, dtype=float)
    perturbation_normal = np.asarray(perturbation_normal, dtype=float)

    ensemble_size = augmented.shape[0]
    if ensemble_size < 2:
        raise ValueError("the EnKF needs at least two ensemble members")
    if predicted_increment.shape[0] != ensemble_size:
        raise ValueError("predicted observations and ensemble size do not match")
    observation_dim = predicted_increment.shape[1]
    if observed_increment.shape != (observation_dim,):
        raise ValueError("observed increment has the wrong shape")
    if covariance.shape != (observation_dim, observation_dim):
        raise ValueError("observation covariance has the wrong shape")
    if perturbation_normal.shape != predicted_increment.shape:
        raise ValueError("perturbation_normal has the wrong shape")

    augmented_anomaly = augmented - augmented.mean(axis=0)
    observation_anomaly = predicted_increment - predicted_increment.mean(axis=0)
    cross_covariance = augmented_anomaly.T @ observation_anomaly / (ensemble_size - 1)
    innovation_covariance = (
        observation_anomaly.T @ observation_anomaly / (ensemble_size - 1) + covariance
    )
    gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    chol = np.linalg.cholesky(covariance)
    perturbation = perturbation_normal @ chol.T
    innovation = observed_increment + perturbation - predicted_increment
    return augmented + innovation @ gain.T


def poisson_ensemble_update(
    augmented: Array,
    predicted_count: Array,
    observed_count: Array,
    perturbation_normal: Array,
    variance_floor: float = 1e-9,
) -> Array:
    """Apply the Gaussian EnKF approximation used for Poisson counts."""
    predicted_count = np.asarray(predicted_count, dtype=float)
    mean_count = np.maximum(predicted_count.mean(axis=0), variance_floor)
    covariance = np.diag(mean_count)
    return ensemble_update(
        augmented,
        predicted_count,
        observed_count,
        covariance,
        perturbation_normal,
    )
