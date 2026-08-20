from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RootResult:
    value: float
    status: str
    bracket_count: int


class GridScoreRootEstimator:
    """Find a score root by linear interpolation on a parameter grid."""

    def __init__(self, no_bracket: str = "hold_previous", tolerance: float = 1e-8) -> None:
        if no_bracket not in {"hold_previous", "minimum_score", "boundary"}:
            raise ValueError("invalid no-bracket policy")
        self.no_bracket = no_bracket
        self.tolerance = tolerance

    def find_root(self, theta: Array, score: Array, reference: float) -> RootResult:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        score = np.asarray(score, dtype=float).reshape(-1)
        if theta.shape != score.shape or theta.size < 2:
            raise ValueError("theta and score must be matching one-dimensional arrays")
        if np.any(np.diff(theta) <= 0):
            raise ValueError("theta must be strictly increasing")

        finite = np.isfinite(score)
        if not finite.any():
            return RootResult(reference, "all_non_finite", 0)

        exact = np.flatnonzero(finite & (np.abs(score) <= self.tolerance))
        if exact.size:
            candidates = theta[exact]
            value = candidates[np.argmin(np.abs(candidates - reference))]
            return RootResult(float(value), "exact_root", int(exact.size))

        roots: list[float] = []
        for i in range(theta.size - 1):
            left = score[i]
            right = score[i + 1]
            if not (np.isfinite(left) and np.isfinite(right)) or left * right >= 0:
                continue
            value = theta[i] - left * (theta[i + 1] - theta[i]) / (right - left)
            roots.append(float(np.clip(value, theta[i], theta[i + 1])))

        if roots:
            candidates = np.asarray(roots)
            value = candidates[np.argmin(np.abs(candidates - reference))]
            return RootResult(float(value), "interpolated_root", len(roots))

        valid = np.flatnonzero(finite)
        if self.no_bracket == "minimum_score":
            index = valid[np.argmin(np.abs(score[valid]))]
            return RootResult(float(theta[index]), "minimum_score", 0)
        if self.no_bracket == "boundary":
            boundary = theta[0] if score[valid[0]] < 0 else theta[-1]
            return RootResult(float(boundary), "boundary", 0)
        return RootResult(float(reference), "hold_previous", 0)

    def estimate_path(
        self,
        theta: Array,
        score_surface: Array,
        initial_value: float,
    ) -> tuple[Array, NDArray[np.str_], NDArray[np.int32]]:
        score_surface = np.asarray(score_surface, dtype=float)
        if score_surface.shape[0] != np.asarray(theta).size:
            raise ValueError("score_surface must have shape (theta, time)")

        path = np.empty(score_surface.shape[1])
        status = np.empty(score_surface.shape[1], dtype="U20")
        brackets = np.empty(score_surface.shape[1], dtype=np.int32)
        reference = float(initial_value)
        for time_index in range(score_surface.shape[1]):
            result = self.find_root(theta, score_surface[:, time_index], reference)
            reference = result.value
            path[time_index] = result.value
            status[time_index] = result.status
            brackets[time_index] = result.bracket_count
        return path, status, brackets
