from dataclasses import dataclass

import numpy as np

from likelihood_filtering.filtering import run_augmented_filter
from likelihood_filtering.observations import GaussianObservation
from likelihood_filtering.signals import SignalStep


@dataclass
class ArraySignal:
    state_shape = (2, 2)

    def sample_initial(self, ensemble_size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(ensemble_size, *self.state_shape))

    def step(
        self,
        state: np.ndarray,
        theta: float,
        dt: float,
        normal: np.ndarray,
    ) -> SignalStep:
        next_state = state - theta * np.tanh(state) * dt + 0.1 * np.sqrt(dt) * normal
        score = -np.sum(np.tanh(state) * normal, axis=(1, 2)) * np.sqrt(dt) / 0.1
        information = np.sum(np.tanh(state) ** 2, axis=(1, 2)) * dt / 0.1**2
        return SignalStep(next_state, score, information)


def test_custom_spatial_array_signal_and_determinism() -> None:
    signal = ArraySignal()
    observation = GaussianObservation(
        lambda state, time: state.mean(axis=(1, 2))[:, None],
        observation_std=0.2,
    )
    increments = np.array([[0.01], [-0.02], [0.03]])
    first = run_augmented_filter(signal, observation, increments, 0.4, 8, 0.01, seed=12)
    second = run_augmented_filter(signal, observation, increments, 0.4, 8, 0.01, seed=12)

    assert first.state_mean.shape == (4, 2, 2)
    np.testing.assert_array_equal(first.state_mean, second.state_mean)
    np.testing.assert_array_equal(first.score_mean, second.score_mean)
    np.testing.assert_array_equal(first.observed_information, second.observed_information)
