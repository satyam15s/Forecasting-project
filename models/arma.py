"""Homoskedastic ARMA(1,1) models with Gaussian or Student-t residuals."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _arma_residuals(y: np.ndarray, const: float, ar: float, ma: float) -> np.ndarray:
    n = len(y)
    eps = np.zeros(n)
    y_prev = 0.0
    e_prev = 0.0
    for t in range(n):
        eps[t] = y[t] - const - ar * y_prev - ma * e_prev
        y_prev = y[t]
        e_prev = eps[t]
    return eps


def _fit_arma_gaussian(y: np.ndarray) -> tuple[dict, np.ndarray, float]:
    def nll(theta: np.ndarray) -> float:
        const, ar, ma, log_sigma = theta
        if abs(ar) >= 0.999 or abs(ma) >= 0.999:
            return 1e12
        sigma = float(np.exp(log_sigma))
        eps = _arma_residuals(y, const, ar, ma)
        return float(0.5 * len(y) * np.log(2 * np.pi * sigma**2) + 0.5 * np.sum(eps**2) / sigma**2)

    x0 = np.array([float(np.mean(y)), 0.05, 0.05, float(np.log(max(np.std(y), 1e-3)))])
    result = optimize.minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 80, "ftol": 1e-6})
    const, ar, ma, log_sigma = result.x
    ar = float(np.clip(ar, -0.99, 0.99))
    ma = float(np.clip(ma, -0.99, 0.99))
    sigma = float(np.exp(log_sigma))
    eps = _arma_residuals(y, const, ar, ma)
    ll = -float(nll(np.array([const, ar, ma, log_sigma])))
    aic = 2 * 4 - 2 * ll
    params = {"const": float(const), "ar": ar, "ma": ma, "scale": sigma, "nu": None}
    return params, eps, aic


def _fit_arma_student(y: np.ndarray) -> tuple[dict, np.ndarray, float]:
    def nll(theta: np.ndarray) -> float:
        const, ar, ma, log_scale, log_nu_m2 = theta
        if abs(ar) >= 0.999 or abs(ma) >= 0.999:
            return 1e12
        scale = float(np.exp(log_scale))
        nu = 2.0 + float(np.exp(log_nu_m2))
        eps = _arma_residuals(y, const, ar, ma)
        return -float(np.sum(stats.t.logpdf(eps, nu, loc=0.0, scale=scale)))

    x0 = np.array(
        [
            float(np.mean(y)),
            0.05,
            0.05,
            float(np.log(max(np.std(y), 1e-3))),
            np.log(4.0),
        ]
    )
    result = optimize.minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-6})
    const, ar, ma, log_scale, log_nu_m2 = result.x
    ar = float(np.clip(ar, -0.99, 0.99))
    ma = float(np.clip(ma, -0.99, 0.99))
    scale = float(np.exp(log_scale))
    nu = 2.0 + float(np.exp(log_nu_m2))
    eps = _arma_residuals(y, const, ar, ma)
    ll = -float(nll(np.array([const, ar, ma, log_scale, log_nu_m2])))
    aic = 2 * 5 - 2 * ll
    params = {"const": float(const), "ar": ar, "ma": ma, "scale": scale, "nu": float(nu)}
    return params, eps, aic


def fit_arma(returns, distribution: str = "normal") -> dict:
    """Fit ARMA(1,1) and return one-step 1% VaR, AIC, and a simulator."""
    if distribution not in {"normal", "t"}:
        raise ValueError(f"Unknown distribution: {distribution}")

    y = np.asarray(returns, dtype=float)
    if distribution == "normal":
        params, eps, aic = _fit_arma_gaussian(y)
        q_eps = stats.norm.ppf(0.01)
    else:
        params, eps, aic = _fit_arma_student(y)
        q_eps = stats.t.ppf(0.01, params["nu"])

    mean = params["const"] + params["ar"] * float(y[-1]) + params["ma"] * float(eps[-1])
    var_1pct = mean + params["scale"] * q_eps

    def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
        if distribution == "normal":
            shocks = rng.normal(0.0, params["scale"], size=n)
        else:
            shocks = rng.standard_t(params["nu"], size=n) * params["scale"]
        out = np.empty(n)
        y_prev = float(y[-1])
        e_prev = float(eps[-1])
        for t in range(n):
            out[t] = params["const"] + params["ar"] * y_prev + params["ma"] * e_prev + shocks[t]
            e_prev = shocks[t]
            y_prev = out[t]
        return out

    label = "ARMA-N" if distribution == "normal" else "ARMA-t"
    return {
        "model": label,
        "aic": float(aic),
        "var_1pct": float(var_1pct),
        "simulate": simulate,
        "params": params,
    }
