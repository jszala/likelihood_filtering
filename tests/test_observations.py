import numpy as np
import pytest

from likelihood_filtering.observations import (
    GaussianObservation,
    PoissonObservation,
    linear_observation,
    positive_sigmoid_intensity,
    sensor_matrix_1d_fem,
)


def test_gaussian_increment_uses_standard_deviation() -> None:
    observation = GaussianObservation(linear_observation(2.0), observation_std=0.3)
    state = np.array([[1.0], [-1.0]])
    normal = np.array([[0.5], [-2.0]])
    dt = 0.04
    increment = observation.sample_increment(state, 0.0, dt, normal)
    expected = 2.0 * state * dt + 0.3 * np.sqrt(dt) * normal
    np.testing.assert_array_equal(increment, expected)
    np.testing.assert_array_equal(observation.increment_covariance(dt, 1), [[0.3**2 * dt]])


def test_gaussian_covariance_python_api() -> None:
    covariance = np.array([[0.2, 0.05], [0.05, 0.1]])
    observation = GaussianObservation(lambda state, time: state, observation_covariance=covariance)
    np.testing.assert_array_equal(observation.covariance(2), covariance)
    with pytest.raises(ValueError, match="exactly one"):
        GaussianObservation(
            lambda state, time: state,
            observation_std=0.1,
            observation_covariance=covariance,
        )


def test_poisson_increment_mean() -> None:
    observation = PoissonObservation(positive_sigmoid_intensity(4.0, 0.0, 0.0, 0.25))
    state = np.zeros((3, 1))
    np.testing.assert_allclose(observation.intensity(state, 0.0) * 0.1, 0.225)


def test_fem_sensor_interpolation() -> None:
    matrix = sensor_matrix_1d_fem([0.1, 0.15], dimensions=3, domain_length=0.4)
    np.testing.assert_allclose(matrix.toarray(), [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
