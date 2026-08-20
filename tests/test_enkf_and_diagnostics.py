import numpy as np

from likelihood_filtering.diagnostics import fisher_louis_information
from likelihood_filtering.enkf import ensemble_update


def test_louis_information() -> None:
    score = np.array([1.0, 2.0, 3.0])
    complete_information = np.array([5.0, 6.0, 7.0])
    expected = complete_information.mean() - score.var()
    assert fisher_louis_information(complete_information, score) == expected


def test_enkf_uses_supplied_increment_covariance() -> None:
    augmented = np.array([[0.0], [2.0]])
    predicted = np.array([[0.0], [1.0]])
    observed = np.array([0.5])
    covariance = np.array([[0.3**2 * 0.04]])
    normal = np.zeros((2, 1))
    updated = ensemble_update(augmented, predicted, observed, covariance, normal)

    cross_covariance = 1.0
    innovation_covariance = 0.5 + covariance[0, 0]
    gain = cross_covariance / innovation_covariance
    expected = augmented[:, 0] + gain * (observed[0] - predicted[:, 0])
    np.testing.assert_allclose(updated[:, 0], expected)
