from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RecursiveLouisEstimator:
    """Single-filter recursive score-root approximation."""

    theta: float
    lower: float
    upper: float
    step_size: float = 1.0
    information_floor: float = 1e-8
    step_clip: float | None = None

    def update(self, score: float, information: float) -> tuple[float, float]:
        regularized = max(float(information), self.information_floor)
        change = self.step_size * float(score) / regularized
        if self.step_clip is not None:
            change = float(np.clip(change, -self.step_clip, self.step_clip))
        previous = self.theta
        self.theta = float(np.clip(self.theta + change, self.lower, self.upper))
        return self.theta, self.theta - previous

    @staticmethod
    def transport_score(score_ensemble: Array, information_ensemble: Array, change: float) -> Array:
        score_ensemble = np.asarray(score_ensemble, dtype=float)
        information_ensemble = np.asarray(information_ensemble, dtype=float)
        if score_ensemble.shape != information_ensemble.shape:
            raise ValueError("score and information ensembles must have the same shape")
        return score_ensemble - change * information_ensemble
