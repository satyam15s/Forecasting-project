"""McNeil–Frey GARCH filter + GPD tail on standardized residuals."""

from __future__ import annotations

import numpy as np
from scipy import stats

from models.garch_core import fit_ar_garch, simulate_ar_garch

try:
    from arch import arch_model

    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False


def _fit_gaussian_filter(y: np.ndarray) -> dict:
    if _HAS_ARCH:
        model = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
        fitted = model.fit(disp="off", show_warning=False)
        if fitted.convergence_flag == 0:
            forecast = fitted.forecast(horizon=1)
            mean = float(forecast.mean.iloc[-1, 0])
            vol = float(np.sqrt(forecast.variance.iloc[-1, 0]))
            std_resid = np.asarray(fitted.std_resid, dtype=float)
            std_resid = std_resid[np.isfinite(std_resid)]
            const = float(fitted.params.get("Const", fitted.params.get("mu", 0.0)))
            ar = float(fitted.params.get("y[1]", 0.0))
            omega = float(fitted.params["omega"])
            alpha = float(fitted.params["alpha[1]"])
            beta = float(fitted.params["beta[1]"])
            cond_vol = np.asarray(fitted.conditional_volatility, dtype=float)
            params = {
                "const": const,
                "ar": ar,
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "nu": None,
                "mean_f": mean,
                "vol_f": vol,
                "last_eps": float(y[-1] - (const + ar * y[-2])),
                "last_sigma2": float(cond_vol[-1] ** 2),
                "last_y": float(y[-1]),
            }
            return {"params": params, "std_resid": std_resid}
    fitted = fit_ar_garch(y, distribution="normal")
    return {"params": fitted["params"], "std_resid": fitted["std_resid"]}


def fit_evt_mf(returns, residual_quantile: float = 0.90) -> dict:
    """
    Fit AR(1)–GARCH(1,1) by Gaussian PML, then GPD on the lower residual tail.

    AIC is intentionally omitted: the GPD likelihood is not comparable to
    full-sample ARMA/GARCH likelihoods.
    """
    y = np.asarray(returns, dtype=float)
    filtered = _fit_gaussian_filter(y)
    params = filtered["params"]
    std_resid = filtered["std_resid"]

    losses = -std_resid
    u = float(np.quantile(losses, residual_quantile))
    exceedances = losses[losses > u] - u
    if len(exceedances) < 20:
        u = float(np.quantile(losses, 0.85))
        exceedances = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(exceedances, floc=0.0)
    xi = float(xi)
    beta = float(max(beta, 1e-8))
    p_u = float(np.mean(losses > u))

    alpha = 0.01
    if alpha >= p_u:
        z_q = float(np.quantile(std_resid, alpha))
    else:
        if abs(xi) < 1e-6:
            y_ext = -beta * np.log(alpha / p_u)
        else:
            y_ext = (beta / xi) * ((alpha / p_u) ** (-xi) - 1.0)
        z_q = -(u + float(y_ext))

    var_1pct = params["mean_f"] + params["vol_f"] * z_q

    def _sample_innovation(rng: np.random.Generator, n: int) -> np.ndarray:
        z = np.empty(n)
        body = std_resid[losses <= u]
        if len(body) == 0:
            body = std_resid
        for i in range(n):
            if rng.random() < p_u:
                excess = stats.genpareto.rvs(c=xi, loc=0.0, scale=beta, random_state=rng)
                z[i] = -(u + excess)
            else:
                z[i] = float(rng.choice(body))
        return z

    def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
        shocks = _sample_innovation(rng, n)
        return simulate_ar_garch(params, n, rng, distribution="normal", shocks=shocks)

    return {
        "model": "EVT (MF)",
        "aic": np.nan,
        "var_1pct": float(var_1pct),
        "simulate": simulate,
        "std_resid": std_resid,
        "params": {
            **params,
            "u": u,
            "xi": xi,
            "gpd_beta": beta,
            "p_u": p_u,
        },
    }
