from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def fisher_louis_information(information: Array, score: Array) -> float:
    """Estimate observed information with the Louis identity."""
    information = np.asarray(information, dtype=float).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if information.shape != score.shape:
        raise ValueError("information and score ensembles must have the same shape")
    return float(information.mean() - score.var())


def running_quadratic_variation(values: Array) -> Array:
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros_like(values)
    if values.size > 1:
        out[1:] = np.cumsum(np.diff(values) ** 2)
    return out


def quantile_band(values: Array) -> Array:
    """Return the 10%, 50%, and 90% pointwise quantiles."""
    return np.quantile(np.asarray(values, dtype=float), [0.1, 0.5, 0.9], axis=0)
