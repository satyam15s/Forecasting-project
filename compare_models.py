"""Rolling-window model comparison figures (half-slide friendly)."""

from __future__ import annotations

import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, YearLocator, date2num

from download_data import BLACK_MONDAY, DATA_PATH, FRIDAY
from models import (
    estimate_extremogram,
    fit_ar_garch_n,
    fit_ar_garch_t,
    fit_ar_sd_t,
    fit_arma,
    fit_evt_mf,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
WINDOW = 1000
# Re-estimate every ORIGIN_STEP sessions (still a rolling window); keeps CI tractable.
ORIGIN_STEP = 5
ALPHA = 0.01
# Rolling averages over this many backtest points (~ORIGIN_STEP trading days each).
HIT_ROLL = 50
PINBALL_ROLL = 50
EXTREMOGRAM_LAGS = 30
SIM_LENGTH = 5000
FIGSIZE = (6.5, 3.2)
DPI = 200

MODEL_ORDER = ["ARMA-N", "ARMA-t", "ARMA-GARCH-N", "ARMA-GARCH-t", "EVT (MF)", "AR-SD-t"]
COLORS = {
    "ARMA-N": "#6a8395",
    "ARMA-t": "#385268",
    "ARMA-GARCH-N": "#d4a017",
    "ARMA-GARCH-t": "#b05b23",
    "EVT (MF)": "#a52c25",
    "AR-SD-t": "#2f6b4f",
    "Data": "#1a1a1a",
}


def load_returns() -> tuple[np.ndarray, np.ndarray]:
    """Load percent returns and their dates (numpy only, no pandas)."""
    dates: list[date] = []
    closes: list[float] = []
    with DATA_PATH.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dates.append(date.fromisoformat(row["date"]))
            closes.append(float(row["close"]))
    closes_arr = np.asarray(closes, dtype=float)
    rets = 100.0 * (closes_arr[1:] / closes_arr[:-1] - 1.0)
    ret_dates = np.asarray(dates[1:], dtype=object)
    return ret_dates, rets


def pinball_loss(y: float, q: float, alpha: float = ALPHA) -> float:
    return (y - q) * (alpha - float(y < q))


def fit_all(window: np.ndarray) -> list[dict]:
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.append(fit_arma(window, "normal"))
        results.append(fit_arma(window, "t"))
        results.append(fit_ar_garch_n(window))
        results.append(fit_ar_garch_t(window))
        results.append(fit_evt_mf(window))
        results.append(fit_ar_sd_t(window))
    return results


def _eval_origin(payload: tuple) -> list[dict]:
    day_iso, window, realized = payload
    try:
        fits = fit_all(window)
    except Exception as exc:
        print(f"Skip {day_iso}: {exc}")
        return []
    rows = []
    for fit in fits:
        q = float(fit["var_1pct"])
        rows.append(
            {
                "date": day_iso,
                "model": fit["model"],
                "aic": fit["aic"],
                "var_1pct": q,
                "realized": realized,
                "hit": float(realized < q),
                "pinball": pinball_loss(realized, q),
            }
        )
    return rows


def run_backtest(dates: np.ndarray, returns: np.ndarray) -> list[dict]:
    n = len(returns)
    origins = list(range(WINDOW, n, ORIGIN_STEP))
    if origins[-1] != n - 1:
        origins.append(n - 1)
    payloads = [
        (
            dates[end].isoformat(),
            np.asarray(returns[end - WINDOW : end], dtype=float),
            float(returns[end]),
        )
        for end in origins
    ]
    total = len(payloads)
    rows: list[dict] = []
    workers = max(1, min(4, os.cpu_count() or 1))
    print(f"Using {workers} workers for {total} origins.")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_eval_origin, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            if done % 20 == 0 or done == total:
                print(f"Backtest {done}/{total}")
    rows.sort(key=lambda row: (row["date"], MODEL_ORDER.index(row["model"])))
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["date", "model", "aic", "var_1pct", "realized", "hit", "pinball"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _series_for_model(rows: list[dict], model: str, field: str) -> tuple[list[datetime], np.ndarray]:
    xs: list[datetime] = []
    ys: list[float] = []
    for row in rows:
        if row["model"] != model:
            continue
        xs.append(datetime.fromisoformat(row["date"]))
        value = row[field]
        ys.append(float(value) if value not in {"", "nan", "NaN"} else np.nan)
    return xs, np.asarray(ys, dtype=float)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    min_periods = max(50, window // 5)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        count = i + 1 - start
        if count >= min_periods:
            out[i] = (cumsum[i + 1] - cumsum[start]) / count
    return out


def _style_half_slide(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dedede", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(
        date2num(datetime.combine(BLACK_MONDAY, datetime.min.time())),
        color="#606060",
        linestyle=":",
        linewidth=1.0,
        label="Black Monday",
    )
    ax.legend(loc="best", fontsize=7, frameon=False, ncols=2)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))


def plot_aic(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for model in MODEL_ORDER:
        if model == "EVT (MF)":
            continue
        xs, ys = _series_for_model(rows, model, "aic")
        ax.plot(xs, ys, color=COLORS[model], linewidth=1.1, label=model)
    _style_half_slide(ax, "Rolling-window AIC (1000 days)", "AIC")
    fig.savefig(OUTPUT_DIR / "aic_rolling.png", dpi=DPI)
    plt.close(fig)


def plot_exceedance(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for model in MODEL_ORDER:
        xs, hits = _series_for_model(rows, model, "hit")
        rate = _rolling_mean(hits, HIT_ROLL)
        ax.plot(xs, rate, color=COLORS[model], linewidth=1.1, label=model)
    ax.axhline(ALPHA, color="#303030", linestyle="--", linewidth=0.9, label="Nominal 1%")
    span = HIT_ROLL * ORIGIN_STEP
    _style_half_slide(ax, f"Rolling ~{span}-day VaR exceedance rate", "Hit rate")
    fig.savefig(OUTPUT_DIR / "exceedance_rate.png", dpi=DPI)
    plt.close(fig)


def plot_pinball(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    n_forecasts = 0
    for model in MODEL_ORDER:
        xs, pinball = _series_for_model(rows, model, "pinball")
        _, hits = _series_for_model(rows, model, "hit")
        roll = _rolling_mean(pinball, PINBALL_ROLL)
        avg = float(np.nanmean(pinball))
        n_forecasts = int(np.sum(np.isfinite(pinball)))
        n_breaches = int(np.nansum(hits))
        color = COLORS[model]
        ax.plot(
            xs,
            roll,
            color=color,
            linewidth=1.1,
            label=f"{model}: avg {avg:.3f}, {n_breaches}*",
        )
        ax.axhline(avg, color=color, linestyle=":", linewidth=1.0)
    span = PINBALL_ROLL * ORIGIN_STEP
    _style_half_slide(ax, f"Rolling ~{span}-day pinball loss (1% quantile)", "Pinball")
    ax.text(
        0.0,
        -0.18,
        f"* breaches / {n_forecasts} forecasts",
        transform=ax.transAxes,
        fontsize=7,
        color="#404040",
        ha="left",
        va="top",
    )
    fig.savefig(OUTPUT_DIR / "pinball_loss.png", dpi=DPI)
    plt.close(fig)


def plot_extremogram(dates: np.ndarray, returns: np.ndarray) -> None:
    friday_idx = int(np.where(dates == FRIDAY)[0][0])
    start = friday_idx - WINDOW + 1
    window = returns[start : friday_idx + 1]
    lags, data_rho = estimate_extremogram(window, lags=EXTREMOGRAM_LAGS, quantile=0.98)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = fit_all(window)

    rng = np.random.default_rng(1987)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(
        lags,
        data_rho,
        color=COLORS["Data"],
        linewidth=1.6,
        marker="o",
        markersize=3,
        label="Data",
    )
    for fit in fits:
        sim = fit["simulate"](SIM_LENGTH, rng)
        _, rho = estimate_extremogram(sim, lags=EXTREMOGRAM_LAGS, quantile=0.98)
        ax.plot(lags, rho, color=COLORS[fit["model"]], linewidth=1.1, label=fit["model"])
    ax.set_title("Extremogram of |returns| (q=0.98), pre-crash window", fontsize=11)
    ax.set_xlabel("Lag (days)", fontsize=9)
    ax.set_ylabel(r"$\hat{\rho}(h)$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dedede", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=7, frameon=False, ncols=2)
    fig.savefig(OUTPUT_DIR / "extremogram.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Run python download_data.py first.")

    plt.switch_backend("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)

    dates, returns = load_returns()
    n_origins = len(list(range(WINDOW, len(returns), ORIGIN_STEP)))
    print(
        f"Rolling backtest on ~{n_origins:,} origins "
        f"({dates[WINDOW]} → {dates[-1]}), window={WINDOW}, step={ORIGIN_STEP}."
    )
    rows = run_backtest(dates, returns)
    csv_path = OUTPUT_DIR / "rolling_backtest.csv"
    save_csv(rows, csv_path)

    plot_aic(rows)
    plot_exceedance(rows)
    plot_pinball(rows)
    plot_extremogram(dates, returns)
    print(f"Saved rolling CSV and four comparison figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
