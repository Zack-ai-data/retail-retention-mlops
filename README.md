# End-to-End Consumer Retention MLE & MLOps pipeline

[![CI](https://github.com/Zack-ai-data/retail-retention-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Zack-ai-data/retail-retention-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**consumer retail retention analytics on [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**

## Project Overview

 - Dataset: ~1.07M transactions from a UK online gift retailer (Dec 2009 - Dec 2011), spread over two Excel sheets (CC BY 4.0, UCI Machine Learning Repository).
 - Goal: help a retailer keep its customers. Turn raw transactions into customer segments (RFM), churn and lifetime-value predictions, then decide where to spend a retention budget, served through an API and a dashboard.
 - Stack: PostgreSQL, Docker, MLflow, scikit-learn, FastAPI, Streamlit, and (planned) Airflow, Prometheus + Grafana, GitHub Actions.
 - Status: **steps 1-4 done** - Postgres in Docker, raw load, cleaned staging layer, customer feature/label snapshots with automated data quality checks, and three models (churn, 90-day customer value, RFM segments) tracked and registered in MLflow with a promotion quality gate. Lint, strict type checking, a coverage threshold and explicit error handling are in place (see Code quality). Batch scoring, an outreach optimizer with a random holdout, a FastAPI service and a Streamlit dashboard sit on top of the models. Orchestration and monitoring are planned (see Roadmap).

Repo Structure:
```
src/
  config.py       settings and constants (DB connection, paths, column mapping)
  database.py     PostgreSQL access: connect, run SQL files, bulk COPY
  extractor.py    read the Excel sheets and normalise them for the raw layer
  processor.py    build staging and marts from raw, run data quality checks
  features.py     the feature contract: which columns models use, shared by training and the future API
  model.py        trains the churn and value models, applies the gate, orchestrates a training run
  model_search.py candidate model families and the time-ordered cross-validation that chooses between them
  measures.py     metrics for churn scores and spend forecasts (pure functions)
  backtesting.py  evaluation at several time origins, and the BG/NBD baseline scores
  segmentation.py RFM customer segments
  reporting.py    what a run logs to MLflow: figures, importances, parameters, gate decision, datasets
  transforms.py   feature transforms stored inside saved models (numpy only, so a model loads without the training stack)
  lineage.py      the code, data and library versions recorded on every training run
  evaluation.py   paired bootstrap: is a model's gain over a baseline real or noise?
  gate.py         the promotion rules (which evidence a new model needs), as pure functions
  calibration.py  Platt calibration of churn probabilities, applied only when it helps
  baselines.py    BG/NBD, the statistical churn baseline
  hurdle.py       the two-part value model: P(buy) x E[spend | buy]
  tracking.py     MLflow connection, loading the champion so it can be re-scored, recording decisions
  errors.py       explainable failure types (missing file, database down, empty table, ...)
  main.py         entry point / orchestration (--stage load | transform | train | all)
  scoring.py      the one place that turns features into churn probability, spend and margin at risk; batch scoring into ml.predictions
  optimizer.py    who to contact within a fixed budget, with a random holdout group, and how the list changes with the assumptions
  api.py          FastAPI service: stored and live scores, what-if scoring, the outreach plan
  dashboard_client.py  what the dashboard knows about the API (HTTP client and page calculations), tested without Streamlit
sql/
  init/           runs once when the Postgres volume is first created (extra databases)
  schema/         idempotent DDL (schemas, raw table, segment output table), applied on every run
  transform/      SQL that rebuilds staging and marts from raw (the cleaning and feature logic)
services/         one folder per supporting service's container setup
  mlflow/         MLflow server image and entrypoint (Airflow, Prometheus and Grafana will join it here)
  dashboard/      the Streamlit dashboard: app, its own image and its own pinned dependencies
tests/            pytest suite: unit, database, end-to-end pipeline, error-path and real-warehouse
                  regression tests (database tests use a throwaway Postgres database,
                  model tests a throwaway SQLite MLflow store)
data/             source Excel file (UCI Online Retail II, CC BY 4.0, included in the repository)
pyproject.toml    dependencies plus the lint, format, type-check and coverage configuration
CONTRIBUTING.md   engineering workflow: definition of done, what the quality gate proves, where tests go
```

Database layers (PostgreSQL, database `retail_dw`):
 - `raw`: untouched copy of the source (`raw.online_retail`), with `source_sheet` / `source_row` lineage.
 - `staging`: `staging.transactions`, cleaned and deduplicated, every row tagged with a `line_type`.
 - `marts`: `marts.customer_snapshots`, one row per customer per snapshot date with features and churn labels (plus `marts.snapshot_dates`), and `marts.customer_segments`, the latest RFM segment of every customer.
 - `ml`: predictions and monitoring data (planned). Experiments and the model registry live in MLflow, backed by their own `mlflow` database in the same Postgres.

Quickstart (needs Docker Desktop only, no local Python):
```
# 1. the dataset is already in data/online_retail_II.xlsx; create your env file
cp .env.example .env            # then change POSTGRES_PASSWORD
#    optional: record the commit on training runs (the image has no .git folder)
export GIT_SHA=$(git rev-parse HEAD)

# 2. start Postgres and the MLflow server
docker compose up -d postgres mlflow

# 3. run the pipeline: load Excel into raw (a few minutes), build staging + marts (~20 s),
#    train and register the models (~1-2 min)
docker compose run --rm app
#    or one stage at a time:
docker compose run --rm app python -m src.main --stage load
docker compose run --rm app python -m src.main --stage transform
docker compose run --rm app python -m src.main --stage train
docker compose run --rm app python -m src.main --stage score   # live models -> ml.predictions
docker compose run --rm app python -m src.main --stage plan    # who to contact -> ml.outreach_*

# 4. start the API and the dashboard (both bound to localhost only; there is no login)
docker compose up -d api dashboard
#    API docs: http://localhost:8000/docs      Dashboard: http://localhost:8501

# 5. run the tests (test databases are throwaway and never touch your data)
docker compose run --rm quality python -m pytest
#    or the full quality gate (about 15 minutes): lint, format check, strict types, tests +
#    coverage threshold
docker compose run --rm quality
#    the dashboard's own tests (they need Streamlit, which only its image has)
docker compose run --rm dashboard-tests
```
Connect DBeaver to `localhost:5432` with the user, password and database from `.env`.
Browse experiments, metrics, plots and the model registry at http://localhost:5000.

Code quality (all configured in `pyproject.toml`, all run by `docker compose run --rm quality`, which exits non-zero if anything fails):
 - **Lint and format:** `ruff` (bug-finding, import order, modern syntax, simplifications) and `ruff format`.
 - **Types:** `mypy --strict` on `src/`.
 - **Tests:** unit tests; database-backed SQL tests on a throwaway Postgres; end-to-end tests that run the whole pipeline (load, transform, train) on a small synthetic workbook with the real file's two overlapping sheets; error-path tests; and regression tests that copy the loaded raw table into a scratch database, run the current cleaning and feature SQL on it, and check the known counts, line types, snapshot sizes and churn rates, and that re-training reproduces the recorded model results (they skip themselves if the real data is not loaded, or fail with `--require-warehouse`; update the expected numbers in the same commit as any intentional change to the cleaning or feature SQL). Marks: `-m e2e`, `-m warehouse`.
 - **Coverage:** branch coverage with a 90% minimum (98% when this was written). Mutation testing (a one-off, about 10 minutes, not part of the gate) killed 91% of the mutants made in 18 pure functions, using the faster tests only; the training, orchestration, database and SQL code were not mutated, so there is no mutation evidence for them either way.
 - **Error handling:** expected failures (missing or corrupt Excel file, database or MLflow unreachable, running a stage before its input exists, failed data quality checks) raise a specific error from `src/errors.py` with what happened and what to do; the command prints it without a traceback and exits with status 1. Unexpected errors keep their traceback and exit with status 2. The raw load is all-or-nothing, so a failed load leaves the previous data untouched. The database password is never printed.
 - **Workflow:** `CONTRIBUTING.md` defines what "done" means for any change (tests, a green `quality` run, updated regression values when results change on purpose) and what a green run does and does not prove.
 - **Continuous integration:** `.github/workflows/ci.yml` runs on GitHub Actions. On every push and pull request, `quality` (lint, format check, strict types, the fast tests with the 90% coverage floor) and `dashboard` (the Streamlit tests) must pass before a change can merge to `main`, which is protected. After a merge, `full-training` repeats the tests with the real hyperparameter grids and `regression` loads the real dataset and checks that the pinned results reproduce. Training tests use one hyperparameter combination and 100 bootstrap resamples by default (about 11 minutes instead of 23); see CONTRIBUTING.md. The workflow ran green on its first run on GitHub.
 - **Reproducibility:** exact library versions, with hashes, are pinned in `requirements.lock` (runtime) and `requirements-dev.lock` (plus test tools); the Docker images and CI install from them, and a test checks the installed versions match. Every training run is tagged with the git commit, the checksum of the Excel file that was loaded, a checksum of the transform SQL, the lock file checksum and the library versions, and logs the datasets it used, so a model in the registry can be traced to what produced it.
 - **Packaging:** the `app` image is runtime-only and runs as a non-root user; tests and dev tools are in the separate `quality` image. A saved model loads without importing the training code, matplotlib or the database driver (tested in a fresh interpreter), which is what lets the API image stay small.
 - Not set up yet: pre-commit hooks and vulnerability scanning of the dependencies.

Data notes and cleaning decisions (`sql/transform/010_staging_transactions.sql`):
 - **Sheet overlap:** the two sheets share 2010-12-01 to 2010-12-09 as an exact copy (22,523 rows, 1,088 invoices, identical line counts). Each invoice is kept from the earliest sheet only. A quality check proves every dropped row has an identical kept twin.
 - **Repeated lines inside one sheet** (about 12k rows) look like genuine repeat scans, e.g. 20 lines of quantity 1 on one invoice. They are kept and flagged `is_repeated_line`, not deleted.
 - **Nothing is dropped for being odd.** Each row gets a `line_type` so downstream code chooses what to use and counts always reconcile with raw:
   `sale` (quantity and price > 0), `return` (invoice starts with `C`), `non_product` (postage, fees, discounts, manual entries, vouchers, test items), `stock_adjustment` (write-offs and zero-price lines, nearly all with no customer), `adjustment` (invoice starts with `A`; in this data those rows use the bad-debt code `B`, so they land in `non_product`).
 - About 22% of sale lines have no customer id, so customer-level features cover the ~78% of lines where the customer is known.
 - Prices can be negative or zero and customers can span several countries; features use the most recent country.

Retention modelling setup (`sql/transform/020_marts_customer_snapshots.sql`):
 - **Money is net of returns.** A cancelled order must not count as a purchase: one customer's £77,183.60 order was cancelled the same day, which had made them the best "retained" customer in the data. `monetary` is sales minus returns to date (`gross_sales` keeps the gross figure), `spend_last_90d` is net, and `return_rate` is the share of gross sales returned.
 - **Churn definition:** a customer is churned at a snapshot date if they have no net-positive spend (sales minus returns, never below zero) in the following 90 days. Only customers with at least one prior purchase are included.
 - **No leakage:** features use only purchases strictly before the snapshot date; `label_*` columns use only the 90 days after it.
 - **Time-based split:** eight snapshots exactly 90 days apart. Six train (2010-03-18 to 2011-06-11), one tests (2011-09-09), so a model is always tested on a later period than it trained on. The earlier snapshots matter because they include the previous year's pre-Christmas period (2010-09-14). A final `score` snapshot (2011-12-10) describes every customer as of the end of the data; it has no labels (its future is unknown) and is what deployed models score.
 - **Fixed-window features:** model inputs describe the 365 days before each snapshot (`*_365d`), so every snapshot is measured the same way even though the data holds only 3 months of history at the first snapshot and 21 at the test snapshot. `history_days_available` (capped at 365) says how much of the window exists, and `season_sin` / `season_cos` encode the time of year. Lifetime columns (`frequency`, `monetary`, ...) are kept for segmentation.
 - **Actionable population:** the churn and value models train, are gated and report on customers who bought within the last 180 days (`is_actionable`; 53% of customers at the test snapshot). Customers silent for longer are lapsed, a reactivation problem where churn is close to certain (77-90% in every snapshot), and including them flattered every headline number: they are easy to rank by recency alone. The 180-day threshold is two horizons and was chosen from the campaign logic, not tuned on the test snapshot (though I had already seen full-population test results when I chose it). Scoring a lapsed customer with these models is outside what they were trained for.
 - **Features:** recency, frequency, net monetary value, tenure, average order value, average days between orders, distinct products, recent order trend, returns, country.
 - **Worth knowing:** churn runs 40-70% across snapshots (40% in March 2010, 53% in June, 43% in September, 70% in December 2010, 65%, 68%, then 57% in the test snapshot) because the pre-Christmas buying peak shifts who buys in the following 90 days. That distribution shift is a useful monitoring case later.

Data quality checks (run after every transform, the pipeline fails if any fails): dropped rows have identical twins, each invoice comes from one sheet, every row has a line type, features never use data on or after the snapshot date, histories are consistent, only the score snapshot is unlabelled, no customer is labelled retained without positive net spend in the window, and every label window ends inside the available data. Transform, checks and publication happen in one transaction: if a check fails, the previous staging and marts tables are left unchanged.

Modeling (`src/model.py`, tracked in MLflow experiment `retail-retention`):
 - **Selection without peeking:** candidate algorithms and hyperparameters are compared by expanding-window cross-validation over the training snapshots only (train on earlier snapshots, validate on the next one). The test snapshot is not used to choose models or hyperparameters; it is used to report results and by the promotion gate, and the value model's objective was changed once after its first test result had been seen (the change was motivated by cross-validation on the training snapshots).
 - **Churn classifier** (`retail-churn-classifier`): chosen between logistic regression and gradient boosting by cross-validated PR-AUC; gradient boosting is currently selected. On the actionable test customers (churn rate 0.39): ROC-AUC 0.772, PR-AUC 0.652 (1.68x the no-skill rate) against 0.657 / 0.535 for ranking customers by recency alone. The PR-AUC gain is +0.118 with a 95% bootstrap interval of +0.089 to +0.144, and it is significant at all four backtest origins (2010-12-13 to 2011-09-09, each trained only on snapshots whose label window had ended). Among the 20% of customers scored highest, 70.6% churn, against 60.2% when ranking by recency, and the model finds 36.5% of all churners against 31.1%. By band, PR-AUC is 0.554 against 0.391 among customers who bought in the last 90 days, and 0.734 against 0.611 among those who last bought 91-180 days ago. The BG/NBD statistical baseline (probability a customer is still active, fitted before the test snapshot) scores PR-AUC 0.439: it treats every one-time buyer as active, so it does not rank them. The same model scored 0.821 / 0.799 on the whole population before the population was restricted, which was mostly the lapsed customers being easy to rank.
 - **Calibration:** on the actionable test customers the model expects 47% to churn against 39% observed (calibration error 0.087, Brier 0.195). A Platt calibrator is fitted on out-of-fold predictions and kept only if it improves log loss on the latest cross-validation snapshot; here it did not, so the model is uncalibrated. A first version applied it unconditionally and made the test result worse (error 0.148), because churn depends on the season and a mapping learned from other periods does not fit the pre-Christmas window. Probabilities still run high, which is one reason retraining and drift monitoring are in the roadmap.
 - **90-day value model** (`retail-value-regressor`): a two-part (hurdle) model, expected spend = P(buy) x E[spend | buy]. P(buy) is one minus the churn model's churn probability, so churn and value cannot disagree about who is likely to buy. E[spend | buy] is a gamma-loss regressor trained only on customers who bought, selected by cross-validated gamma deviance. A single-stage Poisson model is trained alongside for comparison only. Lessons learned: a first version trained on log(spend) underestimated total spend by 78% in cross-validation, and absolute-error loss collapses to predicting zero for the many non-buyers. The two-part model ranks customers exactly as well as the single-stage one (Spearman 0.592 vs 0.592) and is slightly more biased (-31.8% vs -30.9%), so it adds explanation, not accuracy. It shows where the bias comes from: in the test window the model expects 53% of customers to buy (61% did) and buyers to spend 1,068 (they spent 1,423), a seasonal level shift that the previous year's snapshots only partly capture.
 - **Quality gate** (`src/gate.py`, rules in `docs/problem_statement.md`): a new model version becomes the `champion` (what the API will serve) only if every check passes, and the decision with all its evidence is stored as a tag on the model version and the run (and as `gate.json`). The current champion is loaded and re-scored on the same data as the candidate, rather than trusting the metric it logged when it was trained; a champion trained on different feature columns is reported as not comparable. **Churn** must beat ranking by recency with a bootstrap interval that excludes 0, on the test snapshot and on most backtest origins, and must not be worse than the champion. **Value** must rank customers better than the naive "next 90 days = last 90 days" forecast (Spearman interval excludes 0), capture at least as much spend in the top decile, have a total bias within 40% of actual spend, and not be significantly worse on RMSE with spend capped at the 99th percentile of training spend (so a few wholesale accounts cannot decide it). On the current data both pass. The 40% bias bound is a sanity limit, confirmed by the project owner. It was set **after** seeing that the value model is 31.8% low in the pre-Christmas test window (the naive forecast is 27.0% low); an earlier rule, "no more biased than the naive forecast", rejected the model by 4.8 points. The bound is an assumption to revisit as more seasons of data arrive. **Registry after the retrain on 2026-10-19:** churn v2 is the champion (v1 was trained on different feature columns, so it could not be re-scored and was recorded as not comparable); value v2 is the first value champion (v1 stays in the registry as the rejected attempt); segments v2 is the champion.
 - **RFM segments** (`retail-rfm-segments`): KMeans on log-scaled recency, frequency and monetary value as of the score snapshot, k chosen by silhouette (k=4). Written to `marts.customer_segments`. The largest group, "High value / Active", is 20% of customers and 74% of revenue; "High value / Lapsed" (25% of customers, about 230 days since their last order) is the win-back target.
 - Models are saved with cloudpickle, so only load models from this project's own MLflow registry. A pickle needs the library versions that produced it, which is why they are pinned and recorded on each run. (MLflow's safer `skops` format was not adopted: the custom model classes would each need an explicit trust list.)

Serving and decisions (step 4):
 - **Margin at risk** (`src/scoring.py`): `P(churn) x spend if the customer buys x 30% margin`. "Spend if they buy" is the second stage of the value model, not the 90-day forecast, which already includes the chance of not buying and would count it twice. It is an upper bound on what contacting a customer could save: the project has no campaign data, so it ranks who is at risk and does not claim who can be saved.
 - **Only customers the models were trained for are scored** (bought within the last 180 days). Asking about a lapsed customer is an explained error that points to the win-back list, not a meaningless number.
 - **Stored and live scores, deliberately both.** Batch scoring writes `ml.predictions` (every run is kept, with the model versions): reliable, auditable, and the list the outreach plan is made from. The API can also score a customer live with the champion models, or score features you send for a what-if question. Both call the same function, and a test checks that for the same model versions the table and the API give the same numbers. Live scoring does not make a customer's inputs fresher: features are as of the latest snapshot.
 - **Outreach plan** (`src/optimizer.py`, tables `ml.outreach_plans` and `ml.outreach_assignments`): rank by margin at risk, take the top 500 whose margin at risk exceeds the £5 contact cost, and hold out a random 10% (fixed seed, stored) who are deliberately not contacted, so that impact could be measured later. The 500 and the £5 and 30% are stated assumptions (`docs/problem_statement.md`).
 - **On the real data** (3,471 customers scored as of 2011-12-10): the plan contacts 450 customers, holds out 50, costs £2,250 and covers £83,271 of margin at risk. **Every one of the 3,471 customers clears the £5 contact cost** (margin at risk averages about £83, £286,419 in total), so the cost never limits the list and the fixed number of customers decides it. The Assumptions page of the dashboard shows this.
 - **API** (`src/api.py`, FastAPI, interactive docs at `/docs`): `GET /health`, `GET /customers/{id}/score` (live), `GET /customers/{id}/prediction` (stored), `GET /customers/{id}/features`, `POST /score` (what-if), `GET /outreach/latest`, `GET /outreach/sensitivity`, `POST /models/reload`. Inputs are validated, the customer id is a bound query parameter, errors say what to do, and a database failure never includes the password. Models are loaded once at startup (a failure shows as a clear 503, not a crash) and can be reloaded without a restart. Every answer carries the model versions.
 - **Dashboard** (`services/dashboard/`, Streamlit): who to contact (with a CSV download of the contact group only, never the holdout), customer lookup (live next to stored), what-if, and the assumptions. It talks only to the API, runs in its own small image with its own pinned dependencies, and none of the training stack. Its logic is in `src/dashboard_client.py` and tested without Streamlit; the pages are tested with Streamlit's test runner against a fake API, and were also driven against the live API and real data.
 - **Not yet:** authentication (everything is bound to localhost), scheduled scoring (Airflow, step 5), and monitoring of the stored predictions (step 6).

Roadmap:
 1. Postgres + raw load (done)
 2. Staging cleaning, customer features and churn labels (done)
 3. Model training tracked in MLflow: RFM segmentation, churn, 90-day value (done)
 4. Batch scoring, outreach optimizer, FastAPI + Streamlit (done)
 5. Airflow DAGs for ETL and retraining
 6. Prometheus + Grafana monitoring (including drift checks), GitHub Actions CI
 7. (optional) deploy the API to local Kubernetes

## License

The code is released under the MIT License (see `LICENSE`). The dataset in `data/` is the UCI Online Retail II dataset, which keeps its own licence (CC BY 4.0); credit the UCI Machine Learning Repository if you reuse it.
