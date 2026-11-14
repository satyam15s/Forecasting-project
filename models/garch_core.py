"""Pure NumPy/SciPy AR(1)–GARCH(1,1) filters (Normal or Student-t)."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _garch_path(
    y: np.ndarray,
    const: float,
    ar: float,
    omega: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    mean = np.empty(n)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    unc = omega / max(1.0 - alpha - beta, 1e-4)
    sigma2[0] = max(unc, 1e-8)
    mean[0] = const
    eps[0] = y[0] - mean[0]
    for t in range(1, n):
        mean[t] = const + ar * y[t - 1]
        eps[t] = y[t] - mean[t]
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] < 1e-8:
            sigma2[t] = 1e-8
    return mean, sigma2, eps


def _unpack(theta: np.ndarray, with_nu: bool) -> tuple:
    const = float(theta[0])
    ar = float(np.clip(theta[1], -0.99, 0.99))
    omega = float(np.exp(theta[2]))
    # alpha, beta > 0 and alpha+beta < 0.999 via stick-breaking on (0,1).
    a_raw = 1.0 / (1.0 + np.exp(-theta[3]))
    b_raw = 1.0 / (1.0 + np.exp(-theta[4]))
    alpha = float(0.999 * a_raw * (1.0 - b_raw))
    beta = float(0.999 * b_raw)
    if with_nu:
        nu = 2.1 + float(np.exp(theta[5]))
        return const, ar, omega, alpha, beta, nu
    return const, ar, omega, alpha, beta, None


def _nll(theta: np.ndarray, y: np.ndarray, distribution: str) -> float:
    with_nu = distribution == "t"
    const, ar, omega, alpha, beta, nu = _unpack(theta, with_nu)
    mean, sigma2, eps = _garch_path(y, const, ar, omega, alpha, beta)
    if distribution == "normal":
        return float(0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2))
    scale = np.sqrt(sigma2 * (nu - 2.0) / nu)
    return -float(np.sum(stats.t.logpdf(eps, nu, loc=0.0, scale=scale)))


def fit_ar_garch(y: np.ndarray, distribution: str = "t") -> dict:
    """Return params, AIC, last mean/vol, std residuals, and forecast helpers."""
    y = np.asarray(y, dtype=float)
    var0 = float(max(np.var(y), 1e-6))
    # Target alpha≈0.05, beta≈0.90 → b_raw≈0.90/0.999, a_raw≈0.05/(0.999*(1-b_raw))
    x0 = np.array(
        [
            float(np.mean(y) * 0.2),
            0.05,
            np.log(max(var0 * 0.05, 1e-6)),
            -0.5,  # a_raw
            2.2,  # b_raw ~ high persistence
        ]
    )
    if distribution == "t":
        x0 = np.append(x0, np.log(6.0))

    result = optimize.minimize(
        _nll,
        x0,
        args=(y, distribution),
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-7},
    )
    if not np.isfinite(result.fun) or result.fun >= 1e11:
        # Fallback grid-ish restart from unconditional GARCH-ish values.
        x0[2] = np.log(max(var0 * 0.02, 1e-6))
        x0[3], x0[4] = -1.0, 2.5
        result = optimize.minimize(
            _nll,
            x0,
            args=(y, distribution),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-5},
        )

    with_nu = distribution == "t"
    const, ar, omega, alpha, beta, nu = _unpack(result.x, with_nu)
    mean, sigma2, eps = _garch_path(y, const, ar, omega, alpha, beta)
    std_resid = eps / np.sqrt(sigma2)
    nll = float(_nll(result.x, y, distribution))
    k = 6 if with_nu else 5
    aic = 2 * k + 2 * nll

    mean_f = float(const + ar * y[-1])
    sigma2_f = float(omega + alpha * eps[-1] ** 2 + beta * sigma2[-1])
    vol_f = float(np.sqrt(max(sigma2_f, 1e-12)))
    if distribution == "normal":
        z_q = float(stats.norm.ppf(0.01))
    else:
        z_q = float(stats.t.ppf(0.01, nu) * np.sqrt((nu - 2.0) / nu))
    var_1pct = mean_f + vol_f * z_q

    params = {
        "const": float(const),
        "ar": float(ar),
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu,
        "mean_f": mean_f,
        "vol_f": vol_f,
        "last_eps": float(eps[-1]),
        "last_sigma2": float(sigma2[-1]),
        "last_y": float(y[-1]),
    }
    return {
        "params": params,
        "aic": float(aic),
        "var_1pct": float(var_1pct),
        "std_resid": std_resid,
        "mean": mean,
        "sigma2": sigma2,
        "eps": eps,
        "distribution": distribution,
    }


def simulate_ar_garch(
    params: dict,
    n: int,
    rng: np.random.Generator,
    distribution: str,
    shocks: np.ndarray | None = None,
) -> np.ndarray:
    const = params["const"]
    ar = params["ar"]
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    nu = params["nu"]
    if shocks is None:
        if distribution == "normal":
            z = rng.normal(size=n)
        else:
            z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    else:
        z = shocks
    out = np.empty(n)
    y_prev = params["last_y"]
    sigma2 = max(
        float(omega + alpha * params["last_eps"] ** 2 + beta * params["last_sigma2"]),
        1e-8,
    )
    for t in range(n):
        eps = np.sqrt(sigma2) * z[t]
        out[t] = const + ar * y_prev + eps
        sigma2 = max(omega + alpha * eps**2 + beta * sigma2, 1e-8)
        y_prev = out[t]
    return out
