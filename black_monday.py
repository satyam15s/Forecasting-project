"""Compare pre-crash Gaussian and Student-t GARCH(1,1) forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import PercentFormatter

from download_data import BLACK_MONDAY, DATA_PATH, FRIDAY

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def plot_results(returns, observed, results):
    plt.switch_backend("Agg")
    fig, events = plt.subplots(figsize=(8, 4.2), layout="constrained")
    october = returns.loc["1987-10-01":BLACK_MONDAY.isoformat()]
    crash_day = pd.Timestamp(BLACK_MONDAY)
    friday = pd.Timestamp(FRIDAY)
    colors = [
        "#a52c25" if day.date() == BLACK_MONDAY else "#6a8395"
        for day in october.index
    ]
    events.bar(october.index, october, color=colors)
    events.axhline(0, color="#303030", linewidth=0.8)
    events.axvline(
        friday + (crash_day - friday) / 2,
        color="#606060",
        linestyle=":",
        label="Forecast made after Friday’s close",
    )
    events.annotate(
        f"Friday  {october.loc[friday]:.2f}%",
        xy=(friday, october.loc[friday]),
        xytext=(-8, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#385268",
    )
    events.annotate(
        f"Oct 19  {observed:.2f}%",
        xy=(crash_day - pd.Timedelta(days=2), observed),
        xytext=(-8, 5),
        textcoords="offset points",
        ha="right",
        color="#a52c25",
    )
    events.set(
        title="S&P 500 returns before and on Black Monday",
        ylabel="Close-to-close return",
    )
    events.legend(loc="lower left", frameon=False)
    events.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    events.xaxis.set_major_locator(DayLocator(bymonthday=[1, 7, 13, 16, 19]))
    events.xaxis.set_major_formatter(DateFormatter("%b %d"))

    events.spines[["top", "right"]].set_visible(False)
    events.grid(axis="y", color="#dedede", linewidth=0.6)
    events.set_axisbelow(True)
    fig.savefig(OUTPUT_DIR / "forecast_timeline.png", dpi=200)
    plt.close(fig)

    fig, tails = plt.subplots(figsize=(8, 3.8), layout="constrained")
    probabilities = results["probability_at_or_below_observed"]
    for position, (probability, color) in enumerate(
        zip(probabilities, ["#385268", "#b05b23"])
    ):
        tails.scatter(probability, position, color=color, s=95, zorder=3)
        coefficient, exponent = f"{probability:.2e}".split("e")
        tails.annotate(
            rf"${coefficient} \times 10^{{{int(exponent)}}}$",
            (probability, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontweight="bold",
        )
    tails.set_xscale("log")
    tails.set_yticks(range(len(results)), results["model"])
    tails.invert_yaxis()
    tails.set(
        title=f"Friday’s forecast of a Monday return of {observed:.2f}% or worse",
        xlabel="Forecast probability (log scale)",
        xlim=(probabilities.min() / 100, min(1, probabilities.max() * 100_000)),
        ylim=(len(results) - 0.6, -0.4),
    )
    tails.spines[["top", "right", "left"]].set_visible(False)
    tails.tick_params(axis="y", length=0)
    tails.grid(axis="x", color="#dedede", linewidth=0.6)
    fig.savefig(OUTPUT_DIR / "black_monday.png", dpi=200)
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Run python download_data.py first.")

    closes = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")["close"]
    returns = (100 * closes.pct_change(fill_method=None)).iloc[1:]
    training = returns.loc[:FRIDAY.isoformat()]
    observed = returns.loc[BLACK_MONDAY.isoformat()]
    rows = []

    for label, distribution in (("Gaussian GARCH", "normal"), ("Student-t GARCH", "t")):
        # Both models use the same returns and variance recursion.
        model = arch_model(
            training, mean="Constant", vol="GARCH", p=1, q=1, dist=distribution
        )
        fitted = model.fit(disp="off")
        if fitted.convergence_flag != 0:
            raise RuntimeError(f"{label} fit failed: {fitted.optimization_result.message}")

        forecast = fitted.forecast(horizon=1)
        mean = forecast.mean.iloc[-1, 0]
        volatility = np.sqrt(forecast.variance.iloc[-1, 0])
        shock = (observed - mean) / volatility

        # The arch Student-t distribution already has unit variance.
        innovation = model.distribution
        shape = fitted.params.reindex(innovation.parameter_names()).to_numpy()
        tail_probability = float(innovation.cdf(shock, shape))
        lower_quantile = mean + volatility * float(innovation.ppf(0.01, shape))
        rows.append(
            {
                "model": label,
                "mean_pct": mean,
                "volatility_pct": volatility,
                "observed_return_pct": observed,
                "observed_standardized_shock": shock,
                "probability_at_or_below_observed": tail_probability,
                "one_pct_return_quantile_pct": lower_quantile,
                "aic_in_sample": fitted.aic,
                "omega": fitted.params["omega"],
                "alpha": fitted.params["alpha[1]"],
                "beta": fitted.params["beta[1]"],
                "nu": fitted.params.get("nu", np.nan),
            }
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    plot_results(returns, observed, results)

    print(
        f"Fitted on {len(training):,} returns from {training.index[0].date()} "
        f"through {training.index[-1].date()}."
    )
    print(f"Observed return on {BLACK_MONDAY}: {observed:.4f}%\n")
    columns = [
        "model",
        "volatility_pct",
        "one_pct_return_quantile_pct",
        "probability_at_or_below_observed",
    ]
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"\nSaved comparison CSV and both figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
