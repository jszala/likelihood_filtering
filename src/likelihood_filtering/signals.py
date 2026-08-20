from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky_banded, solveh_banded

Array = NDArray[np.float64]


@dataclass
class SignalStep:
    state: Array
    score_increment: Array
    information_increment: Array


class ParametricSignal(Protocol):
    """Interface used by the augmented filter."""

    state_shape: tuple[int, ...]

    def sample_initial(self, ensemble_size: int, rng: np.random.Generator) -> Array: ...

    def step(self, state: Array, theta: float, dt: float, normal: Array) -> SignalStep: ...


@dataclass
class OrnsteinUhlenbeckSignal:
    dimensions: int = 1
    mu: float = 1.0
    state_diffusion: float = 0.15
    initial_mean: float = 0.0
    initial_std: float = 0.2

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        if self.state_diffusion <= 0:
            raise ValueError("state_diffusion must be positive")
        if self.initial_std < 0:
            raise ValueError("initial_std must be non-negative")
        self.state_shape = (self.dimensions,)

    def sample_initial(self, ensemble_size: int, rng: np.random.Generator) -> Array:
        return rng.normal(
            self.initial_mean,
            self.initial_std,
            size=(ensemble_size, self.dimensions),
        )

    def step(self, state: Array, theta: float, dt: float, normal: Array) -> SignalStep:
        state = np.asarray(state, dtype=float)
        normal = np.asarray(normal, dtype=float)
        if state.shape != normal.shape or state.shape[1:] != self.state_shape:
            raise ValueError("state and normal must have shape (ensemble, dimensions)")

        d_w = np.sqrt(dt) * normal
        factor = self.mu / self.state_diffusion
        score = -factor * np.sum(state * d_w, axis=1)
        information = factor**2 * np.sum(state**2, axis=1) * dt
        next_state = state - theta * self.mu * state * dt + self.state_diffusion * d_w
        return SignalStep(next_state, score, information)


class Heat1DFEMSignal:
    """Finite-element heat equation with implicit drift stepping."""

    def __init__(
        self,
        dimensions: int = 20,
        domain_length: float = 0.1,
        state_noise_variance: float = 0.015,
        initial_mean: float = 0.0,
        initial_std: float = 0.05,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if domain_length <= 0:
            raise ValueError("domain_length must be positive")
        if state_noise_variance <= 0:
            raise ValueError("state_noise_variance must be positive")
        if initial_std < 0:
            raise ValueError("initial_std must be non-negative")

        self.dimensions = dimensions
        self.domain_length = domain_length
        self.state_noise_variance = state_noise_variance
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.state_shape = (dimensions,)

        h = domain_length / (dimensions + 1)
        self.mass_diag = np.full(dimensions, 2.0 * h / 3.0)
        self.mass_off = np.full(max(dimensions - 1, 0), h / 6.0)
        self.stiffness_diag = np.full(dimensions, 2.0 / h)
        self.stiffness_off = np.full(max(dimensions - 1, 0), -1.0 / h)
        self.mass_banded = self._banded(self.mass_diag, self.mass_off)
        self.mass_cholesky = cholesky_banded(self.mass_banded, lower=True)

    @staticmethod
    def _banded(diag: Array, off: Array) -> Array:
        banded = np.zeros((2, diag.size))
        banded[0] = diag
        if diag.size > 1:
            banded[1, :-1] = off
        return banded

    @staticmethod
    def _tridiagonal_product(diag: Array, off: Array, values: Array) -> Array:
        out = values * diag
        if values.shape[1] > 1:
            out[:, 1:] += values[:, :-1] * off
            out[:, :-1] += values[:, 1:] * off
        return out

    def _mass_cholesky_product(self, normal: Array) -> Array:
        out = normal * self.mass_cholesky[0]
        if self.dimensions > 1:
            out[:, 1:] += normal[:, :-1] * self.mass_cholesky[1, :-1]
        return out

    def _solve_mass_cholesky(self, values: Array) -> Array:
        out = np.empty_like(values)
        diagonal = self.mass_cholesky[0]
        out[:, 0] = values[:, 0] / diagonal[0]
        if self.dimensions > 1:
            sub = self.mass_cholesky[1, :-1]
            for i in range(1, self.dimensions):
                out[:, i] = (values[:, i] - sub[i - 1] * out[:, i - 1]) / diagonal[i]
        return out

    def sample_initial(self, ensemble_size: int, rng: np.random.Generator) -> Array:
        return rng.normal(
            self.initial_mean,
            self.initial_std,
            size=(ensemble_size, self.dimensions),
        )

    def step(self, state: Array, theta: float, dt: float, normal: Array) -> SignalStep:
        state = np.asarray(state, dtype=float)
        normal = np.asarray(normal, dtype=float)
        if state.shape != normal.shape or state.shape[1:] != self.state_shape:
            raise ValueError("state and normal must have shape (ensemble, dimensions)")

        mass_state = self._tridiagonal_product(self.mass_diag, self.mass_off, state)
        mass_noise = np.sqrt(dt) * self._mass_cholesky_product(normal)
        rhs = mass_state + np.sqrt(self.state_noise_variance) * mass_noise

        system = self._banded(
            self.mass_diag + theta * dt * self.stiffness_diag,
            self.mass_off + theta * dt * self.stiffness_off,
        )
        next_state = solveh_banded(system, rhs.T, lower=True, check_finite=False).T

        stiffness_state = self._tridiagonal_product(
            self.stiffness_diag,
            self.stiffness_off,
            state,
        )
        whitened = self._solve_mass_cholesky(stiffness_state)
        score = -np.sqrt(dt / self.state_noise_variance) * np.sum(whitened * normal, axis=1)
        information = dt / self.state_noise_variance * np.sum(whitened**2, axis=1)
        return SignalStep(next_state, score, information)
