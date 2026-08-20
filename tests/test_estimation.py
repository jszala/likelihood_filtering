import numpy as np
import pytest

from likelihood_filtering.estimation import GridScoreRootEstimator
from likelihood_filtering.experimental import RecursiveLouisEstimator


def test_root_interpolation_and_nearest_root() -> None:
    estimator = GridScoreRootEstimator()
    theta = np.array([0.0, 1.0, 2.0, 3.0])
    score = np.array([-1.0, 1.0, -1.0, 1.0])
    root = estimator.find_root(theta, score, reference=2.2)
    assert root.value == 2.5
    assert root.status == "interpolated_root"
    assert root.bracket_count == 3


def test_root_fallbacks() -> None:
    theta = np.array([0.0, 1.0, 2.0])
    score = np.array([3.0, 2.0, 1.0])
    assert GridScoreRootEstimator().find_root(theta, score, 0.4).value == 0.4
    minimum = GridScoreRootEstimator("minimum_score").find_root(theta, score, 0.4)
    assert (minimum.value, minimum.status) == (2.0, "minimum_score")
    boundary = GridScoreRootEstimator("boundary").find_root(theta, score, 0.4)
    assert (boundary.value, boundary.status) == (2.0, "boundary")


def test_recursive_update_and_score_transport() -> None:
    estimator = RecursiveLouisEstimator(0.5, 0.1, 1.0, step_size=0.5)
    theta, change = estimator.update(score=0.2, information=0.5)
    assert theta == 0.7
    assert change == pytest.approx(0.2)
    transported = estimator.transport_score(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        change,
    )
    np.testing.assert_allclose(transported, [0.4, 1.2])
