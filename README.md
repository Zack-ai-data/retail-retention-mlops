# End-to-End Consumer Retention MLE & MLOps pipeline

[![CI](https://github.com/Zack-ai-data/retail-retention-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Zack-ai-data/retail-retention-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**consumer retail retention analytics on [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**

## Problem statement

A retailer can only afford to contact a limited number of customers, so it needs to know which ones are about to stop buying and how much margin is at risk if they do. A customer counts as churned if their net spend (sales minus returns) is zero over the next 90 days. The task is to predict that, estimate the margin at risk, and choose who to contact within a fixed budget. The full definition and the rules a model must pass are in [docs/problem_statement.md](docs/problem_statement.md).

## Results

 - On a period later than anything it trained on, the churn model ranks customers better than a "least recent buyer first" rule (PR-AUC 0.652 against 0.535), and the same held at four time points between December 2010 and September 2011.
 - The probabilities themselves are less reliable: the predicted churn rate was too low in some seasons and too high in others, and recalibrating did not fix it. The ranking is more trustworthy than the pound figures.

## Architecture

```
Excel -> Postgres (raw -> staging -> marts) -> MLflow (train, gate, registry)
      -> batch scores + outreach plan -> FastAPI -> Streamlit
```

Airflow runs the ETL and retraining. Prometheus and Grafana monitor the API, the data and the models, and alerts go to Slack. GitHub Actions checks every change.

## Limitations

 - The data is a fixed historical file, so nothing shows whether contacted customers actually stayed.
 - The API has no login and everything runs locally in Docker.

## Project overview

 - Dataset: about 1.07M transactions from a UK online gift retailer (Dec 2009 - Dec 2011), spread over two Excel sheets (CC BY 4.0, UCI Machine Learning Repository).
 - Stack: PostgreSQL, Docker, MLflow, Airflow, scikit-learn, FastAPI, Streamlit, Prometheus, Grafana, GitHub Actions.
 - Status: **steps 1-6 done**. Postgres in Docker, a raw load, a cleaned staging layer, customer snapshots with churn labels and automated data quality checks; three models (churn, 90-day customer value, RFM segments) tracked in MLflow behind a promotion gate; batch scoring, an outreach plan with a random holdout, a FastAPI service and a Streamlit dashboard; Airflow DAGs; Prometheus and Grafana monitoring with alerts to Slack; and GitHub Actions checks on every change. Step 7 (local Kubernetes) is optional and not started.

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
  main.py         entry point / orchestration (--stage load | transform | train | score | plan | monitor | report | all)
  scoring.py      the one place that turns features into churn probability, spend and margin at risk; batch scoring into ml.predictions
  optimizer.py    who to contact within a fixed budget, with a random holdout group, and how the list changes with the assumptions
  monitoring.py   the `monitor` stage: the small aggregate tables the dashboards read (drift, backtest, realised performance)
  metrics.py      Prometheus metrics for the API
  promotion_status.py  what the promotion gate decided about the newest version of each model
  calibration_experiment.py  the recalibration trial (reads and writes nothing)
  survival_experiment.py  the survival-style scorer trial (reads and writes nothing)
  demo_traffic.py  labelled synthetic traffic for the API, for demonstrations
  api.py          FastAPI service: stored and live scores, what-if scoring, the outreach plan
  dashboard_client.py  what the dashboard knows about the API (HTTP client and page calculations), tested without Streamlit
sql/
  init/           runs once when the Postgres volume is first created (extra databases)
  schema/         idempotent DDL (schemas, the raw table, and the segment, prediction, outreach and monitoring tables), applied on every run
  transform/      SQL that rebuilds staging and marts from raw (the cleaning and feature logic)
services/         one folder per supporting service's container setup
  mlflow/         MLflow server image and entrypoint
  dashboard/      the Streamlit dashboard: app, its own image and its own pinned dependencies
  airflow/        Airflow (Celery executor): image with the pipeline inside it, the DAGs, and the database-URL helper
  prometheus/     Prometheus configuration (it scrapes the API's /metrics)
  grafana/        Grafana: data sources and dashboards as files, and the read-only database login's SQL
tests/            pytest suite: unit, database, end-to-end pipeline, error-path and real-warehouse
                  regression tests (database tests use a throwaway Postgres database,
                  model tests a throwaway SQLite MLflow store)
docs/             the problem statement (definition, assumptions, model acceptance rules) and review material
.github/          CI workflows and the Dependabot configuration
data/             source Excel file (UCI Online Retail II, CC BY 4.0, included in the repository)
pyproject.toml    dependencies plus the lint, format, type-check and coverage configuration
CONTRIBUTING.md   engineering workflow: definition of done, what the quality gate proves, where tests go
```

Database layers (PostgreSQL, database `retail_dw`):
 - `raw`: untouched copy of the source (`raw.online_retail`), with `source_sheet` / `source_row` lineage.
 - `staging`: `staging.transactions`, cleaned and deduplicated, every row tagged with a `line_type`.
 - `marts`: `marts.customer_snapshots`, one row per customer per snapshot date with features and churn labels (plus `marts.snapshot_dates`), and `marts.customer_segments`, the latest RFM segment of every customer.
 - `ml`: stored predictions and scoring runs (`ml.predictions`, `ml.scoring_runs`), outreach plans (`ml.outreach_plans`, `ml.outreach_assignments`) and the small monitoring tables (`ml.monitor_*`). Experiments and the model registry live in MLflow, backed by their own `mlflow` database in the same Postgres.

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
#    or the full quality gate (about 20 minutes): lint, format check, strict types, tests +
#    coverage threshold
docker compose run --rm quality
#    the dashboard's own tests (they need Streamlit, which only its image has)
docker compose run --rm dashboard-tests
```
Connect DBeaver to `localhost:5432` with the user, password and database from `.env`.
Browse experiments, metrics, plots and the model registry at http://localhost:5000.

Code quality (configured in `pyproject.toml`, all run by `docker compose run --rm quality`, which exits non-zero if anything fails):
 - **Lint, format and types:** `ruff` and `ruff format`, and `mypy --strict` on `src/`.
 - **Tests:** unit tests, database-backed SQL tests on a throwaway Postgres, end-to-end tests of the whole pipeline on a small synthetic workbook, error-path tests, and regression tests on the real data that skip themselves when it is not loaded. Branch coverage has a 90% minimum.
 - **Error handling:** expected failures (a missing file, the database or MLflow unreachable, a stage run before its input exists, failed data quality checks) raise a specific error from `src/errors.py` that says what happened and what to do. The raw load is all-or-nothing.
 - **Reproducibility:** library versions are locked by hash (`requirements.lock`, `requirements-dev.lock`), and every training run records the git commit and checksums of the Excel file, the transform SQL and the lock file, so a model in the registry can be traced to what produced it.
 - **CI** (`.github/workflows/ci.yml`): lint, types, the fast tests, the dashboard tests and the Airflow DAG load must pass before a change can merge to `main`. After a merge the full training tests, a regression run on the real data and a fresh-machine check also run, and all of it runs again every Monday on a schedule.
 - **Dependency updates:** Dependabot opens weekly pull requests for GitHub Actions and Docker images. It is kept away from major versions of stateful services (Postgres, Redis, Prometheus, Grafana, Airflow) and from the Python libraries, which are locked on purpose.
 - **Releases:** tagging a green commit on `main` publishes the API and dashboard images to the GitHub Container Registry.
 - **Workflow:** `CONTRIBUTING.md` defines what "done" means for a change.
 - Not set up yet: pre-commit hooks.

Data and features (`sql/transform/`):
 - **Cleaning:** the two sheets overlap for nine days, and each invoice is kept from the earliest sheet only. Nothing is dropped for being odd: every row gets a `line_type` (`sale`, `return`, `non_product`, `stock_adjustment`, `adjustment`), so later code chooses what to use and the counts reconcile with raw. About a fifth of sale lines have no customer id, so customer features cover the rest.
 - **Money is net of returns,** so a cancelled order does not count as a purchase.
 - **Churn** means zero net spend in the 90 days after a snapshot date. Features use only purchases before that date, so there is no leakage. Snapshots are 90 days apart, and a model is always tested on a later period than it trained on.
 - **Fixed-window features** describe the 365 days before each snapshot, so every snapshot is measured the same way.
 - **Actionable population:** the churn and value models are trained, gated and reported on customers who bought within the last 180 days. Customers silent for longer are a reactivation problem where churn is close to certain, and including them flattered every headline number. The 180 days came from the campaign logic and was not tuned on the test snapshot, though full-population test results had been seen first.
 - **Data quality checks** run after every transform and the pipeline fails if any fails. Transform, checks and publication happen in one transaction, so a failed check leaves the previous tables unchanged.

Modeling (`src/model.py`, tracked in MLflow experiment `retail-retention`):
 - **Selection without peeking:** candidates are compared by expanding-window cross-validation over the training snapshots only. The test snapshot is used to report results and by the promotion gate, not to choose models. The value model's objective was changed once after its first test result had been seen, motivated by cross-validation on the training snapshots.
 - **Churn classifier:** chosen between logistic regression and gradient boosting by cross-validated PR-AUC, and compared with ranking by recency and with the BG/NBD statistical baseline, at the test snapshot and at four backtest origins, with a paired bootstrap to tell a real gain from noise.
 - **Calibration:** a Platt calibrator is applied only if it improves log loss in cross-validation. So far it has not, so the model is uncalibrated, and its probabilities are off in different directions in different seasons.
 - **90-day value model:** a two-part (hurdle) model, expected spend = P(buy) x E[spend | buy], so churn and value cannot disagree about who is likely to buy. It ranks customers as well as a single-stage model and is biased low in the pre-Christmas test window, so it adds explanation, not accuracy.
 - **RFM segments:** KMeans on log-scaled recency, frequency and monetary value, written to `marts.customer_segments`.
 - **Quality gate** (`src/gate.py`, rules in `docs/problem_statement.md`): a new model version becomes the `champion` (what the API serves) only if every check passes, and the decision with its evidence is stored with the model version. Churn must beat the recency baseline with a bootstrap interval that excludes 0 and must not be worse than the champion. Value must beat a naive forecast and keep its bias within 40% of actual spend; that bound was set after seeing the value model's bias in the pre-Christmas window, and is an assumption to revisit as more seasons arrive.
 - Models are saved with cloudpickle, so only load models from this project's own MLflow registry.

Serving and decisions (step 4):
 - **Margin at risk** (`src/scoring.py`): P(churn) x spend if the customer buys x 30% margin. It is an upper bound on what contacting a customer could save: there is no campaign data, so it ranks who is at risk and does not claim who can be saved.
 - **Only customers the models were trained for are scored** (bought within the last 180 days). Asking about a lapsed customer gives an explained error.
 - **Stored and live scores, both on purpose:** batch scoring writes `ml.predictions` (every run kept, with the model versions), and the API can also score a customer live, or score features you send as a what-if. Both call the same function, and a test checks that they agree.
 - **Outreach plan** (`src/optimizer.py`): rank by margin at risk, take the top 500 whose margin at risk exceeds the £5 contact cost, and hold out a random 10% (fixed seed) who are deliberately not contacted, so impact could be measured later. The 500, the £5 and the 30% are stated assumptions in `docs/problem_statement.md`.
 - **API** (`src/api.py`, FastAPI, docs at `/docs`): health, live and stored scores, features, what-if scoring, the outreach plan and its sensitivity, and a model reload. Inputs are validated and errors say what to do. If the models or tables are not there yet (a new machine, or MLflow not ready at start-up), it answers a clear 503 that says what to run, and it fetches the models by itself once they exist.
 - **Dashboard** (`services/dashboard/`, Streamlit): who to contact, customer lookup, what-if and the assumptions. It talks only to the API and runs in its own small image.
 - **Access control:** with no `API_KEY` in `.env` the API is open, which is fine on your own machine. With one, every route except health, metrics and the docs needs it in the `X-API-Key` header, and only `API_ADMIN_KEY` may reload the models.
 - Not yet: HTTPS and rate limiting (everything is bound to localhost).

Orchestration (step 5):
 - **Airflow with the Celery executor and Redis,** started with `docker compose --profile airflow up -d airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer`. The interface is at `http://localhost:8080` (localhost only); log in with `AIRFLOW_ADMIN_USER` and `AIRFLOW_ADMIN_PASSWORD` from `.env`. Stop it with `docker compose --profile airflow stop`.
 - **The pipeline runs in its own Python environment inside the Airflow image,** built from this project's hashed lock file, so tasks use the same library versions the tests did.
 - **Secrets come from `.env`, never from defaults:** `airflow-init` refuses to run without them.
 - **DAGs:** `pipeline_healthcheck` (run by hand; shows that the environment, the command line, Postgres and MLflow work from where tasks run), `retail_etl` (load, then transform and data quality checks, published in one transaction or not at all; no schedule, because the dataset does not change) and `retail_retrain_score` (starts when `retail_etl` has refreshed `customer_snapshots`: train, then report the promotion decision and score, then plan, then monitor). A new model not being promoted is a normal result, not a failure: the previous champion keeps serving. Trigger one by hand with `docker compose --profile airflow exec airflow-scheduler airflow dags trigger <dag_id>`.
 - To make runs record the commit, `export GIT_SHA=$(git rev-parse HEAD)` before `docker compose --profile airflow build`.
 - Not yet: a clock-based schedule for the DAGs.

Monitoring (step 6):
 - **Started with** `docker compose --profile monitoring up -d` (start the API too, so there is something to scrape: `docker compose up -d api`). Grafana is at `http://localhost:3000` (log in with `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` from `.env`) and Prometheus at `http://localhost:9090`; both are bound to localhost. To stop only monitoring: `docker compose stop grafana prometheus`.
 - **The API exposes `GET /metrics`** (`src/metrics.py`): requests, latency and failures per route, and the version of each champion model it is serving. Routes are labelled by template, never by the requested URL.
 - **Grafana reads Postgres directly,** through `grafana_ro`, a login that can only read, created by the `grafana-db-init` job. In the warehouse it can see only the scoring-run log and the aggregate `ml.monitor_*` tables, with no customer-level data.
 - **Everything Grafana shows is a file in the repository** (`services/grafana/`): the data sources, four dashboards (Retail API, Customers and churn, Models, Drift) and the alert rules. **Grafana did not notice edits while running on this Windows setup, so after changing a dashboard or provisioning file run `docker compose restart grafana`.**
 - **The `monitor` stage** (`python -m src.main --stage monitor`, part of `--stage all`) writes the small aggregate tables the dashboards read (`ml.monitor_*`, `src/monitoring.py`): customers and churn rate per snapshot, score distributions, segment sizes, drift, every model version with the gate's decision, logged metrics, the churn backtest and realised performance.
 - **Drift check:** the population stability index of every model input and of the churn probability, each snapshot against the one about a year earlier, so the seasons are not mistaken for drift. The dataset is static, so real drift never happens here; tests that shift a snapshot on purpose show that the check detects it.
 - **Realised performance:** once the 90 days after a scoring run have passed, the churn probabilities that were served are compared with what customers did. The table is empty on the real data, because every live scoring run is on the last snapshot, whose 90 days never end; tests on generated customers show that it works.
 - **Alert rules** (`services/grafana/provisioning/alerting/rules.yml`): drift, poor realised performance, a model not promoted, an old champion, a failed pipeline run, the API serving an older model than the registry's champion, the API not answering, and server errors. A firing rule goes to one Slack channel through an incoming webhook. The webhook address is a secret: it lives only in the git-ignored `.env` as `ALERT_WEBHOOK_URL`, and without it Grafana still starts and evaluates the rules.
 - **Two trials that changed no model:** a recalibration trial (`src/calibration_experiment.py`) found that the probabilities are not a steady over-prediction, because the gap changes sign with the season, and that none of four ways of shifting them helped. A survival-style scorer trial (`src/survival_experiment.py`) asks how long until the next purchase instead of one yes/no question, and it never beat the live model. Both read the warehouse and MLflow and write nothing. Run one with `docker compose build app`, then `docker compose run --rm app python -m src.calibration_experiment` (or `src.survival_experiment`).
 - **Demo traffic** (`python -m src.demo_traffic`): sends labelled synthetic requests so the API dashboards are not flat. Every request carries `X-Demo-Traffic: true` and is counted apart from real ones. Run it with `docker compose run --rm app python -m src.demo_traffic --api-url http://api:8000 --minutes 5`.

Roadmap:
 1. Postgres + raw load (done)
 2. Staging cleaning, customer features and churn labels (done)
 3. Model training tracked in MLflow: RFM segmentation, churn, 90-day value (done)
 4. Batch scoring, outreach optimizer, FastAPI + Streamlit (done)
 5. Airflow DAGs for ETL and retraining (done)
 6. Prometheus + Grafana monitoring, with alerts to Slack, and GitHub Actions CI with a weekly scheduled run (done)
 7. (optional) deploy the API to local Kubernetes

## License

The code is released under the MIT License (see `LICENSE`). The dataset in `data/` is the UCI Online Retail II dataset, which keeps its own licence (CC BY 4.0); credit the UCI Machine Learning Repository if you reuse it.
