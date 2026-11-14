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

## Run on your Mac with Codex

Open this folder in Codex or Terminal. The analysis was verified with Python 3.10. Once the daily CSV is present, the analysis can run offline.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_data.py
.venv/bin/python black_monday.py
```

The packages live inside the project’s `.venv` folder. The data goes to `data/sp500_daily.csv`. The model comparison is in `outputs/model_comparison.csv`.

### Figures for the modelling slides

1. **Model setup.** Use `outputs/forecast_timeline.png` to show the Friday cutoff and the held-out Monday return. Both models were fit on the full pre-crash training window described above. The figure zooms in on October.
2. **Forecast comparison.** Use `outputs/black_monday.png` to show the two estimated probabilities of Monday’s loss or worse. The probabilities use a log axis because they span many orders of magnitude. Both models treat the crash as extremely unlikely, even with the market moves observed through Friday.

## Share and run through GitHub

If your teammate already has a repository, clone it, create a branch, and copy these project files into it. Add the Python scripts, requirements, README, and workflow file. Merge the ignore rules into any existing `.gitignore`. Commit, push the branch, and open a pull request.

If you are starting a new team repository, create an empty private repository on GitHub. From this folder, create a local commit if needed and then push it. Replace `OWNER/REPO` with the path to your new repository. Invite your teammate from the repository settings. [GitHub’s steps for publishing local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=mac)

```sh
# First commit, if this folder does not already have one
git init -b main
git add .gitignore .github README.md requirements.txt download_data.py black_monday.py
git commit -m "Compare GARCH models for Black Monday"

# Connect the new empty GitHub repo
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

Once the workflow file is on the default branch, open **Actions → Run Black Monday analysis → Run workflow**. GitHub downloads the data and installs packages on its own runner. Open the completed run and download **black-monday-results** under **Artifacts** for the CSV and figures. [Running a GitHub workflow manually](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
