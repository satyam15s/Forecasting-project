# Black Monday 1987

How much downside risk did two GARCH models imply for Monday, October 19, 1987, using the information available at the previous close?

## Data and models

The downloader saves 2,024 daily S&P 500 price-index closes from December 31, 1979 through December 31, 1987. The source is a [fixed GitHub snapshot of Yahoo Finance `^GSPC`](https://github.com/fja05680/dow-sp500-100-years/blob/52a1ddd481fc947e511f709896d4f2b3fb49fb0e/SP500.csv). The October 16 and 19 closes match 282.70 and 224.84 in [Schwert’s study of the crash](https://www.nber.org/system/files/working_papers/w2954/w2954.pdf).

1. Compute simple returns in percent as `100 * (close_today / close_previous_session - 1)`. This measures the same close-to-close decline as the project slide.
2. Fit both models on 1,971 returns from January 2, 1980 through Friday, October 16, 1987. The first close supplies the starting price. Monday's close enters only when evaluating the forecast.
3. Forecast one trading session ahead. Both models use a constant mean and the same GARCH(1,1) variance equation. They estimate their parameters separately. Gaussian GARCH uses Normal shocks and Student-t GARCH uses unit-variance Student-t shocks with fitted degrees of freedom. [One-step forecasting in `arch`](https://arch.readthedocs.io/en/stable/univariate/forecasting.html)
4. Evaluate each forecast distribution at the observed return. Compute the standardized shock as `(observed_return - forecast_mean) / forecast_volatility` and apply the fitted innovation CDF. The CSV also reports the 1% return quantile and the fitted parameters.

## First run

The S&P 500 fell **20.47%** on Black Monday.

| Model | Forecast daily volatility | 1% return quantile | Model probability of a return at or below −20.47% |
| --- | ---: | ---: | ---: |
| Gaussian GARCH(1,1) | 1.66% | −3.81% | 2.51 × 10⁻³⁵ |
| Student-t GARCH(1,1) | 1.56% | −3.87% | 1.48 × 10⁻⁷ |

The Student-t fit estimated 8.17 degrees of freedom and assigned far more probability to the crash despite a slightly lower volatility forecast. Both models still assigned very small probabilities. These probabilities are extrapolations from fitted models for one session. The event study alone cannot establish which model produces better forecasts across many days.

```

Once the workflow file is on the default branch, open **Actions → Run Black Monday analysis → Run workflow**. GitHub downloads the data and installs packages on its own runner. Open the completed run and download **black-monday-results** under **Artifacts** for the CSV and figures. [Running a GitHub workflow manually](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
