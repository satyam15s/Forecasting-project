"""Simplified score-driven AR(1) with Student-t innovations (Delle Monache–Petrella style)."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _student_logpdf(eps: float, sigma: float, nu: float) -> float:
    z = eps / sigma
    return float(stats.t.logpdf(z, nu) - np.log(sigma))


def _filter(y: np.ndarray, theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Static params theta = (phi0, phi1, log_sigma0, log_k_phi, log_k_sigma, log_nu_m2).
    """
    phi0, phi1, log_sigma0, log_k_phi, log_k_sigma, log_nu_m2 = theta
    k_phi = np.exp(log_k_phi)
    k_sigma = np.exp(log_k_sigma)
    nu = 2.0 + np.exp(log_nu_m2)
    eta = 1.0 / nu

    n = len(y)
    sigma = np.exp(log_sigma0)
    phi1 = float(np.clip(phi1, -0.99, 0.99))
    ll = 0.0
    means = np.empty(n)
    sigmas = np.empty(n)
    y_prev = float(y[0])
    for t in range(n):
        mean_t = phi0 + phi1 * y_prev
        means[t] = mean_t
        sigmas[t] = sigma
        eps = float(y[t] - mean_t)
        ll += _student_logpdf(eps, sigma, nu)
        zeta = eps / sigma
        w = (1.0 + eta) / (1.0 - 2.0 * eta + eta * zeta**2)
        phi0 = phi0 + k_phi * w * eps
        phi1 = float(np.clip(phi1 + k_phi * w * eps * y_prev, -0.99, 0.99))
        dlog_sigma = 0.5 * k_sigma * (w * zeta**2 - 1.0)
        sigma = float(np.clip(np.exp(np.log(sigma) + dlog_sigma), 1e-4, 50.0))
        y_prev = float(y[t])
    return ll, means, sigmas


def fit_ar_sd_t(returns) -> dict:
    """Estimate a simplified AR-SD-t model by maximizing the predictive log-likelihood."""
    y = np.asarray(returns, dtype=float)
    x0 = np.array(
        [
            float(np.mean(y)),
            0.05,
            float(np.log(max(np.std(y), 1e-3))),
            np.log(0.01),
            np.log(0.05),
            np.log(4.0),
        ]
    )

    def objective(theta: np.ndarray) -> float:
        try:
            ll, _, _ = _filter(y, theta)
            return -ll
        except Exception:
            return 1e12

    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 40, "ftol": 1e-5},
    )
    theta_hat = result.x
    ll, _, _ = _filter(y, theta_hat)
    k = len(theta_hat)
    aic = 2 * k - 2 * ll

    phi0, phi1, log_sigma0, log_k_phi, log_k_sigma, log_nu_m2 = theta_hat
    k_phi = float(np.exp(log_k_phi))
    k_sigma = float(np.exp(log_k_sigma))
    nu = 2.0 + float(np.exp(log_nu_m2))
    eta = 1.0 / nu
    sigma = float(np.exp(log_sigma0))
    phi1 = float(np.clip(phi1, -0.99, 0.99))
    y_prev = float(y[0])
    for t in range(len(y)):
        mean_t = phi0 + phi1 * y_prev
        eps = float(y[t] - mean_t)
        zeta = eps / sigma
        w = (1.0 + eta) / (1.0 - 2.0 * eta + eta * zeta**2)
        phi0 = phi0 + k_phi * w * eps
        phi1 = float(np.clip(phi1 + k_phi * w * eps * y_prev, -0.99, 0.99))
        sigma = float(
            np.clip(np.exp(np.log(sigma) + 0.5 * k_sigma * (w * zeta**2 - 1.0)), 1e-4, 50.0)
        )
        y_prev = float(y[t])
    mean_next = phi0 + phi1 * y_prev
    var_1pct = mean_next + sigma * stats.t.ppf(0.01, nu)

    def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
        p0, p1 = float(phi0), float(phi1)
        s = float(sigma)
        kp, ks = k_phi, k_sigma
        out = np.empty(n)
        y_p = float(y[-1])
        for t in range(n):
            m = p0 + p1 * y_p
            eps = rng.standard_t(nu) * s
            out[t] = m + eps
            zeta = eps / s
            w = (1.0 + eta) / (1.0 - 2.0 * eta + eta * zeta**2)
            p0 = p0 + kp * w * eps
            p1 = float(np.clip(p1 + kp * w * eps * y_p, -0.99, 0.99))
            s = float(np.clip(np.exp(np.log(s) + 0.5 * ks * (w * zeta**2 - 1.0)), 1e-4, 50.0))
            y_p = out[t]
        return out

    return {
        "model": "AR-SD-t",
        "aic": float(aic),
        "var_1pct": float(var_1pct),
        "simulate": simulate,
        "params": {
            "phi0": float(phi0),
            "phi1": float(phi1),
            "sigma": float(sigma),
            "nu": float(nu),
            "k_phi": float(k_phi),
            "k_sigma": float(k_sigma),
            "ll": float(ll),
            "converged": bool(result.success),
        },
    }
