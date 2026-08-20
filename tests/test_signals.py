import numpy as np

from likelihood_filtering.signals import Heat1DFEMSignal, OrnsteinUhlenbeckSignal


def test_ou_step_and_derivatives() -> None:
    signal = OrnsteinUhlenbeckSignal(dimensions=1, mu=2.0, state_diffusion=0.5)
    state = np.array([[1.0], [-2.0]])
    normal = np.array([[0.3], [-0.4]])
    result = signal.step(state, theta=0.25, dt=0.04, normal=normal)

    d_w = 0.2 * normal
    expected_state = state - 0.25 * 2.0 * state * 0.04 + 0.5 * d_w
    expected_score = -(2.0 / 0.5) * np.sum(state * d_w, axis=1)
    expected_information = (2.0 / 0.5) ** 2 * np.sum(state**2, axis=1) * 0.04
    np.testing.assert_allclose(result.state, expected_state)
    np.testing.assert_allclose(result.score_increment, expected_score)
    np.testing.assert_allclose(result.information_increment, expected_information)


def test_heat_fem_matrices() -> None:
    signal = Heat1DFEMSignal(dimensions=3, domain_length=0.4)
    np.testing.assert_allclose(signal.mass_diag, [1 / 15, 1 / 15, 1 / 15])
    np.testing.assert_allclose(signal.mass_off, [1 / 60, 1 / 60])
    np.testing.assert_allclose(signal.stiffness_diag, [20.0, 20.0, 20.0])
    np.testing.assert_allclose(signal.stiffness_off, [-10.0, -10.0])


def test_heat_zero_noise_step_solves_implicit_system() -> None:
    signal = Heat1DFEMSignal(dimensions=3, domain_length=0.4)
    state = np.array([[1.0, 0.5, -0.25]])
    result = signal.step(state, theta=0.025, dt=0.01, normal=np.zeros_like(state))

    mass = np.diag(signal.mass_diag) + np.diag(signal.mass_off, 1) + np.diag(signal.mass_off, -1)
    stiffness = (
        np.diag(signal.stiffness_diag)
        + np.diag(signal.stiffness_off, 1)
        + np.diag(signal.stiffness_off, -1)
    )
    expected = np.linalg.solve(mass + 0.025 * 0.01 * stiffness, mass @ state[0])
    np.testing.assert_allclose(result.state[0], expected)
    np.testing.assert_allclose(result.score_increment, 0.0)
    assert np.all(result.information_increment >= 0.0)


def test_heat_state_noise_uses_variance() -> None:
    signal = Heat1DFEMSignal(dimensions=2, state_noise_variance=0.015)
    assert np.sqrt(signal.state_noise_variance) == np.sqrt(0.015)
