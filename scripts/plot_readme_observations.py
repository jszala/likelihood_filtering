"""Plot one OU signal and three observation-increment streams for the README."""

from pathlib import Path

import matplotlib
import numpy as np

from likelihood_filtering.observations import (
    GaussianObservation,
    PoissonObservation,
    linear_observation,
    positive_sigmoid_intensity,
    sigmoid_observation,
)
from likelihood_filtering.signals import OrnsteinUhlenbeckSignal

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    dt = 0.01
    steps = 500
    theta_true = 0.37
    time = np.arange(steps + 1) * dt
    signal = OrnsteinUhlenbeckSignal(
        dimensions=1,
        mu=1.0,
        state_diffusion=0.15,
        initial_mean=0.0,
        initial_std=0.2,
    )
    rng_signal = np.random.default_rng(20260820)
    state = signal.sample_initial(1, rng_signal)
    truth = np.empty(steps + 1)
    truth[0] = state[0, 0]
    for index in range(steps):
        normal = rng_signal.standard_normal(state.shape)
        state = signal.step(state, theta_true, dt, normal).state
        truth[index + 1] = state[0, 0]

    linear = GaussianObservation(
        linear_observation(coefficient=2.0),
        observation_std=0.068526,
        evaluation_rule="right",
    )
    nonlinear = GaussianObservation(
        sigmoid_observation(coefficient=1.0, scale=17.2047, shift=0.15),
        observation_std=0.0117758,
        evaluation_rule="right",
    )
    poisson = PoissonObservation(
        positive_sigmoid_intensity(
            coefficient=3394.91,
            scale=8.0,
            shift=0.04,
            floor=0.25,
        ),
        evaluation_rule="left",
    )
    observation_models = [linear, nonlinear, poisson]
    observation_rngs = [np.random.default_rng(seed) for seed in (1101, 1102, 1103)]
    increments = np.empty((3, steps))
    means = np.empty((3, steps))
    for index in range(steps):
        for case, (observation, rng) in enumerate(
            zip(observation_models, observation_rngs, strict=True)
        ):
            observed_index = index if observation.evaluation_rule == "left" else index + 1
            observed_state = np.array([[truth[observed_index]]])
            observed_time = time[observed_index]
            if isinstance(observation, GaussianObservation):
                means[case, index] = observation.mean(observed_state, observed_time)[0, 0] * dt
                normal = rng.standard_normal((1, 1))
                increment = observation.sample_increment(observed_state, observed_time, dt, normal)
            else:
                means[case, index] = observation.intensity(observed_state, observed_time)[0, 0] * dt
                increment = observation.sample_increment(observed_state, observed_time, dt, rng)
            increments[case, index] = increment[0, 0]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.3), sharex=True, layout="constrained")
    figure.suptitle("One hidden signal, three observed increment streams", fontsize=15)
    axes[0, 0].plot(time, truth, color="#284b8f", linewidth=1.4)
    axes[0, 0].set(title="Hidden OU signal", ylabel=r"$X_t$")
    axes[0, 0].axhline(0, color="#aaaaaa", linewidth=0.7)

    titles = ("Linear Gaussian", "Nonlinear Gaussian", "Poisson counts")
    positions = ((0, 1), (1, 0), (1, 1))
    colors = ("#3b79a7", "#7c58a5", "#ce713d")
    for case, (row, column) in enumerate(positions):
        axis = axes[row, column]
        axis.plot(
            time[1:],
            increments[case],
            color=colors[case],
            linewidth=0.8,
            alpha=0.65,
            label="observed increment",
        )
        axis.plot(
            time[1:],
            means[case],
            color="#202936",
            linewidth=1.5,
            label="conditional mean",
        )
        axis.set(title=titles[case], ylabel=r"$\Delta Y_n$")
        axis.grid(axis="y", alpha=0.18)
    axes[0, 1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("time")
    axes[1, 1].set_xlabel("time")
    axes[0, 0].set_xlim(0, time[-1])

    output = Path(__file__).resolve().parents[1] / "docs/assets/signal_and_observations.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
