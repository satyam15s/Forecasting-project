# Black Monday 1987

How much downside risk did volatility and tail models imply for Monday, October 19, 1987, using information available at previous closes?

## Data and models

The downloader saves 2,024 daily S&P 500 price-index closes from December 31, 1979 through December 31, 1987. The source is a [fixed GitHub snapshot of Yahoo Finance `^GSPC`](https://github.com/fja05680/dow-sp500-100-years/blob/52a1ddd481fc947e511f709896d4f2b3fb49fb0e/SP500.csv). The October 16 and 19 closes match 282.70 and 224.84 in [Schwert’s study of the crash](https://www.nber.org/system/files/working_papers/w2954/w2954.pdf).

1. Compute simple returns in percent as `100 * (close_today / close_previous_session - 1)`.
2. **Event study** (`black_monday.py`): fit Gaussian and Student-t GARCH(1,1) on returns through Friday, October 16, 1987, then evaluate the probability of the observed Monday crash.
3. **Rolling comparison** (`compare_models.py`): 1000-day rolling windows through end-1987, re-estimated every 5 sessions (`ORIGIN_STEP`), comparing six models at the 1% lower quantile. ARMA(1,1) is fit in pure NumPy/SciPy; GARCH uses `arch` when available and a NumPy fallback otherwise.

### Models in the rolling comparison

| Label | Specification |
| --- | --- |
| ARMA-N | ARMA(1,1) + Gaussian residuals, constant variance |
| ARMA-t | ARMA(1,1) + Student-t residuals, constant variance |
| ARMA-GARCH-N | AR(1)–GARCH(1,1)–Gaussian |
| ARMA-GARCH-t | AR(1)–GARCH(1,1)–Student-t |
| EVT (MF) | McNeil–Frey: Gaussian AR(1)–GARCH filter + GPD on residual lower tail |
| AR-SD-t | Simplified score-driven AR(1) with Student-t innovations (Delle Monache–Petrella style) |

The two homoskedastic ARMA models are intentional weak baselines for extremes.

## First-run event study

The S&P 500 fell **20.47%** on Black Monday.

| Model | Forecast daily volatility | 1% return quantile | Model probability of a return at or below −20.47% |
| --- | ---: | ---: | ---: |
| Gaussian GARCH(1,1) | 1.66% | −3.81% | 2.51 × 10⁻³⁵ |
| Student-t GARCH(1,1) | 1.56% | −3.87% | 1.48 × 10⁻⁷ |

The Student-t fit estimated about 8.17 degrees of freedom and assigned far more probability to the crash despite a slightly lower volatility forecast. Both models still assigned very small probabilities.

## Rolling comparison figures

Half-slide figures written to `outputs/`:

1. `aic_rolling.png` — window AIC over time for ARMA-N, ARMA-t, ARMA-GARCH-N, ARMA-GARCH-t, AR-SD-t. **EVT is excluded**: its GPD likelihood is only over threshold exceedances and is not comparable to full-sample ARMA/GARCH AIC.
2. `exceedance_rate.png` — rolling VaR hit rate at 1% versus the nominal 1% line (all six models).
3. `pinball_loss.png` — rolling pinball loss at the 1% quantile (main ranking metric for quantile forecasts).
4. `extremogram.png` — Davis–Mikosch extremogram of absolute returns (quantile 0.98) for the pre-crash 1000-day window, plus long simulations from each fitted model.

Intermediate metrics are saved in `outputs/rolling_backtest.csv`.

### How to read the comparisons

- **Pinball and exceedances** are valid out-of-sample quantile diagnostics and are the right place to rank models for downside VaR.
- **AIC** is in-sample fit within each window, not a tail-forecast score.
- **Extremogram** diagnoses extreme-event clustering (lag dependence), not VaR accuracy.
- Large gaps between ARMA and GARCH models mostly reflect missing volatility dynamics, not fine differences in tail shape.

## Reproduce

```bash
python download_data.py
python black_monday.py
python compare_models.py
```

Once the workflow file is on the default branch, open **Actions → Run Black Monday analysis → Run workflow**. GitHub downloads the data and installs packages on its own runner. Open the completed run and download **black-monday-results** under **Artifacts** for the CSV and figures. [Running a GitHub workflow manually](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
