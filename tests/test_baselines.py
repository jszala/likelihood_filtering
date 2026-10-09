from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, poisson

from likelihood_filtering.baselines import (
    kalman_log_likelihood,
    kalman_mle,
    particle_grid_mle,
    particle_log_likelihood,
    systematic_resample,
)
from likelihood_filtering.observations import GaussianObservation, PoissonObservation
from likelihood_filtering.signals import OrnsteinUhlenbeckSignal, SignalStep


def test_kalman_likelihood_matches_joint_gaussian_density() -> None:
    signal = OrnsteinUhlenbeckSignal(initial_mean=0.0, initial_std=0.2)
    theta, dt, coefficient, observation_std = 0.37, 0.1, 2.0, 0.07
    observed = np.array([0.02, -0.01, 0.03])
    transition = 1.0 - theta * signal.mu * dt
    process_variance = signal.state_diffusion**2 * dt
    state_from_noise = np.array(
        [
            [transition, 1.0, 0.0, 0.0],
            [transition**2, transition, 1.0, 0.0],
            [transition**3, transition**2, transition, 1.0],
        ]
    )
    state_covariance = (
        state_from_noise
        @ np.diag([signal.initial_std**2, process_variance, process_variance, process_variance])
        @ state_from_noise.T
    )
    covariance = (coefficient * dt) ** 2 * state_covariance + observation_std**2 * dt * np.eye(3)
    expected = multivariate_normal.logpdf(observed, mean=np.zeros(3), cov=covariance)
    actual = kalman_log_likelihood(observed, theta, signal, coefficient, observation_std, dt)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_kalman_mle_respects_bounds() -> None:
    signal = OrnsteinUhlenbeckSignal()
    result = kalman_mle(np.zeros(3), signal, 2.0, 0.1, 0.01, (0.1, 1.0))
    assert 0.1 <= result.theta <= 1.0
    assert np.isfinite(result.log_likelihood)


@dataclass
class DeterministicSignal:
    dimensions: int = 1
    shift: float = 0.0

    def sample_initial(self, ensemble_size: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        assert ensemble_size == 2
        return np.array([[0.0], [1.0]])

    def step(self, state: np.ndarray, theta: float, dt: float, normal: np.ndarray) -> SignalStep:
        del theta, dt, normal
        return SignalStep(state + self.shift, np.zeros(state.shape[0]), np.zeros(state.shape[0]))


def test_particle_likelihood_respects_gaussian_endpoint_timing() -> None:
    signal = DeterministicSignal(shift=1.0)
    values = np.array([1.0])
    for rule, states in [("left", [0.0, 1.0]), ("right", [1.0, 2.0])]:
        observation = GaussianObservation(lambda x, t: x, observation_std=1.0, evaluation_rule=rule)
        actual = particle_log_likelihood(values, 0.3, signal, observation, 1.0, 2, 7)
        expected = logsumexp(norm.logpdf(1.0, loc=states)) - np.log(2.0)
        assert actual == pytest.approx(expected)


def test_particle_likelihood_accumulates_weights_without_resampling() -> None:
    signal = DeterministicSignal()
    observation = GaussianObservation(lambda x, t: x, observation_std=1.0)
    values = np.array([0.0, 0.5])
    actual = particle_log_likelihood(values, 0.3, signal, observation, 1.0, 2, 7)
    expected = logsumexp(
        [norm.logpdf(0.0, loc=state) + norm.logpdf(0.5, loc=state) for state in [0.0, 1.0]]
    ) - np.log(2.0)
    assert actual == pytest.approx(expected)


def test_particle_likelihood_uses_poisson_density_at_left_endpoint() -> None:
    signal = DeterministicSignal(shift=1.0)
    observation = PoissonObservation(lambda x, t: x + 1.0, evaluation_rule="left")
    actual = particle_log_likelihood(np.array([1]), 0.3, signal, observation, 1.0, 2, 7)
    expected = logsumexp(poisson.logpmf(1, [1.0, 2.0])) - np.log(2.0)
    assert actual == pytest.approx(expected)


def test_resampling_and_particle_grid_are_deterministic() -> None:
    np.testing.assert_array_equal(systematic_resample(np.array([0.1, 0.2, 0.7]), 0.25), [0, 2, 2])
    signal = OrnsteinUhlenbeckSignal()
    observation = PoissonObservation(lambda x, t: 2.0 + np.exp(x))
    grid = np.array([0.2, 0.4, 0.6])
    observed = np.array([0, 1, 0, 2])
    first = particle_grid_mle(observed, grid, signal, observation, 0.01, 32, 17)
    second = particle_grid_mle(observed, grid, signal, observation, 0.01, 32, 17)
    assert first == second
    scalar_values = np.array(
        [
            particle_log_likelihood(observed, theta, signal, observation, 0.01, 32, 17)
            for theta in grid
        ]
    )
    assert first.theta == grid[int(np.argmax(scalar_values))]
    assert first.log_likelihood == pytest.approx(float(np.max(scalar_values)), abs=1e-10)


def test_vectorized_gaussian_grid_matches_scalar_particle_likelihood() -> None:
    signal = OrnsteinUhlenbeckSignal()
    observation = GaussianObservation(lambda x, t: x, observation_std=0.1)
    grid = np.array([0.2, 0.4, 0.6])
    observed = np.array([0.02, -0.01, 0.0, 0.01, -0.02])
    result = particle_grid_mle(observed, grid, signal, observation, 0.01, 32, 17)
    scalar_values = np.array(
        [
            particle_log_likelihood(observed, theta, signal, observation, 0.01, 32, 17)
            for theta in grid
        ]
    )
    assert result.theta == grid[int(np.argmax(scalar_values))]
    assert result.log_likelihood == pytest.approx(float(np.max(scalar_values)), abs=1e-10)


def test_particle_filter_rejects_fractional_poisson_counts() -> None:
    signal = OrnsteinUhlenbeckSignal()
    observation = PoissonObservation(lambda x, t: 1.0 + np.exp(x))
    with pytest.raises(ValueError, match="integer"):
        particle_log_likelihood(np.array([0.5]), 0.3, signal, observation, 0.01, 32, 0)
