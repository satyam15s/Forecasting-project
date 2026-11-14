"""AR(1)–GARCH(1,1) with Student-t innovations."""

from __future__ import annotations

import numpy as np

from models.garch_core import fit_ar_garch, simulate_ar_garch

try:
    from arch import arch_model

    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False


def fit_ar_garch_t(returns) -> dict:
    """Fit AR(1)–GARCH(1,1)–t and return VaR, AIC, residuals, and simulator."""
    y = np.asarray(returns, dtype=float)

    if _HAS_ARCH:
        model = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
        fitted = model.fit(disp="off", show_warning=False)
        if fitted.convergence_flag != 0:
            fitted_core = fit_ar_garch(y, distribution="t")
            params = fitted_core["params"]

            def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
                return simulate_ar_garch(params, n, rng, distribution="t")

            return {
                "model": "ARMA-GARCH-t",
                "aic": fitted_core["aic"],
                "var_1pct": fitted_core["var_1pct"],
                "simulate": simulate,
                "std_resid": fitted_core["std_resid"],
                "params": params,
            }

        forecast = fitted.forecast(horizon=1)
        mean = float(forecast.mean.iloc[-1, 0])
        vol = float(np.sqrt(forecast.variance.iloc[-1, 0]))
        innovation = model.distribution
        shape = fitted.params.reindex(innovation.parameter_names()).to_numpy()
        var_1pct = mean + vol * float(innovation.ppf(0.01, shape))
        std_resid = np.asarray(fitted.std_resid, dtype=float)
        std_resid = std_resid[np.isfinite(std_resid)]
        const = float(fitted.params.get("Const", fitted.params.get("mu", 0.0)))
        ar = float(fitted.params.get("y[1]", 0.0))
        omega = float(fitted.params["omega"])
        alpha = float(fitted.params["alpha[1]"])
        beta = float(fitted.params["beta[1]"])
        nu = float(fitted.params["nu"])
        cond_vol = np.asarray(fitted.conditional_volatility, dtype=float)
        params = {
            "const": const,
            "ar": ar,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "nu": nu,
            "mean_f": mean,
            "vol_f": vol,
            "last_eps": float(y[-1] - (const + ar * y[-2])),
            "last_sigma2": float(cond_vol[-1] ** 2),
            "last_y": float(y[-1]),
        }

        def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
            return simulate_ar_garch(params, n, rng, distribution="t")

        return {
            "model": "ARMA-GARCH-t",
            "aic": float(fitted.aic),
            "var_1pct": float(var_1pct),
            "simulate": simulate,
            "std_resid": std_resid,
            "params": params,
        }

    fitted = fit_ar_garch(y, distribution="t")
    params = fitted["params"]

    def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
        return simulate_ar_garch(params, n, rng, distribution="t")

    return {
        "model": "ARMA-GARCH-t",
        "aic": fitted["aic"],
        "var_1pct": fitted["var_1pct"],
        "simulate": simulate,
        "std_resid": fitted["std_resid"],
        "params": params,
    }
