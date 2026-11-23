"""AR(1)–GARCH(1,1) with Gaussian or Student-t innovations."""

from __future__ import annotations

import numpy as np

from models.garch_core import fit_ar_garch, simulate_ar_garch

try:
    from arch import arch_model

    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

_LABELS = {"normal": "ARMA-GARCH-N", "t": "ARMA-GARCH-t"}
_ARCH_DIST = {"normal": "normal", "t": "t"}


def _fit_ar_garch(returns, distribution: str) -> dict:
    """Fit AR(1)–GARCH(1,1) and return VaR, AIC, residuals, and simulator."""
    if distribution not in _LABELS:
        raise ValueError(f"Unknown distribution: {distribution}")
    label = _LABELS[distribution]
    y = np.asarray(returns, dtype=float)

    if _HAS_ARCH:
        model = arch_model(
            y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist=_ARCH_DIST[distribution]
        )
        fitted = model.fit(disp="off", show_warning=False)
        if fitted.convergence_flag != 0:
            fitted_core = fit_ar_garch(y, distribution=distribution)
            params = fitted_core["params"]

            def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
                return simulate_ar_garch(params, n, rng, distribution=distribution)

            return {
                "model": label,
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
        if distribution == "t":
            shape = fitted.params.reindex(innovation.parameter_names()).to_numpy()
            var_1pct = mean + vol * float(innovation.ppf(0.01, shape))
            nu = float(fitted.params["nu"])
        else:
            var_1pct = mean + vol * float(innovation.ppf(0.01))
            nu = None
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
            "nu": nu,
            "mean_f": mean,
            "vol_f": vol,
            "last_eps": float(y[-1] - (const + ar * y[-2])),
            "last_sigma2": float(cond_vol[-1] ** 2),
            "last_y": float(y[-1]),
        }

        def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
            return simulate_ar_garch(params, n, rng, distribution=distribution)

        return {
            "model": label,
            "aic": float(fitted.aic),
            "var_1pct": float(var_1pct),
            "simulate": simulate,
            "std_resid": std_resid,
            "params": params,
        }

    fitted = fit_ar_garch(y, distribution=distribution)
    params = fitted["params"]

    def simulate(n: int, rng: np.random.Generator) -> np.ndarray:
        return simulate_ar_garch(params, n, rng, distribution=distribution)

    return {
        "model": label,
        "aic": fitted["aic"],
        "var_1pct": fitted["var_1pct"],
        "simulate": simulate,
        "std_resid": fitted["std_resid"],
        "params": params,
    }


def fit_ar_garch_n(returns) -> dict:
    """Fit AR(1)–GARCH(1,1)–Normal."""
    return _fit_ar_garch(returns, "normal")


def fit_ar_garch_t(returns) -> dict:
    """Fit AR(1)–GARCH(1,1)–t and return VaR, AIC, residuals, and simulator."""
    return _fit_ar_garch(returns, "t")
