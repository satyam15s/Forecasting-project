"""Sample extremogram estimator (Davis–Mikosch)."""

from __future__ import annotations

import numpy as np


def estimate_extremogram(
    series: np.ndarray,
    lags: int = 30,
    quantile: float = 0.98,
    absolute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate rho(h) = P(|X_{t+h}| > a | |X_t| > a) with a = empirical quantile.

    Returns (lag_grid, estimates) for h = 1..lags.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    z = np.abs(x) if absolute else x
    a = float(np.quantile(z, quantile))
    extreme = z > a
    n = len(z)
    denom = float(np.sum(extreme))
    if denom < 1:
        return np.arange(1, lags + 1), np.full(lags, np.nan)

    estimates = np.empty(lags)
    for h in range(1, lags + 1):
        joint = np.sum(extreme[: n - h] & extreme[h:])
        estimates[h - 1] = joint / denom
    return np.arange(1, lags + 1), estimates
