-- Small aggregate tables for the monitoring dashboards, refreshed by the `monitor` stage
-- (src/monitoring.py). Each holds counts and rates, never a customer row, and Grafana's read-only
-- login can read these and nothing at customer level.
--
-- They are plain tables, not views, on purpose: the transform stage drops and rebuilds the marts
-- tables on every run, and a view over a table that gets dropped would either block that rebuild
-- or be deleted by it. Every table is replaced as a whole inside one transaction, so a dashboard
-- never sees a half-written refresh. Idempotent.

CREATE SCHEMA IF NOT EXISTS ml;

-- One row per customer snapshot (the dataset's quarterly points in time).
CREATE TABLE IF NOT EXISTS ml.monitor_snapshot_summary (
    snapshot_date        date             PRIMARY KEY,
    split                text             NOT NULL,
    customers            integer          NOT NULL,
    actionable_customers integer          NOT NULL,  -- bought within 180 days: the models' population
    churn_rate           double precision,           -- among actionable customers; NULL for the unlabelled score snapshot
    avg_future_spend     double precision,           -- 90-day spend among actionable customers; NULL likewise
    refreshed_at         timestamptz      NOT NULL DEFAULT now()
);

-- How the churn probabilities of each scoring run are spread, in ten equal buckets.
CREATE TABLE IF NOT EXISTS ml.monitor_score_distribution (
    scoring_run_id bigint           NOT NULL,
    bucket         integer          NOT NULL CHECK (bucket BETWEEN 0 AND 9),  -- p_churn in [bucket/10, (bucket+1)/10)
    bucket_label   text             NOT NULL,
    customers      integer          NOT NULL,
    margin_at_risk double precision NOT NULL,
    PRIMARY KEY (scoring_run_id, bucket)
);

CREATE TABLE IF NOT EXISTS ml.monitor_segment_sizes (
    snapshot_date date    NOT NULL,
    segment_name  text    NOT NULL,
    customers     integer NOT NULL,
    PRIMARY KEY (snapshot_date, segment_name)
);

-- Every registered version of the project's models, with what the promotion gate decided.
CREATE TABLE IF NOT EXISTS ml.monitor_model_versions (
    model_name  text        NOT NULL,
    version     integer     NOT NULL,
    created_at  timestamptz NOT NULL,
    decision    text        NOT NULL CHECK (decision IN ('promoted', 'rejected', 'unrecorded')),
    reason      text        NOT NULL,
    is_champion boolean     NOT NULL,
    PRIMARY KEY (model_name, version)
);

-- Every metric logged for those versions, one row per metric (long format, so a dashboard picks
-- the metric by name).
CREATE TABLE IF NOT EXISTS ml.monitor_model_metrics (
    model_name text             NOT NULL,
    version    integer          NOT NULL,
    metric     text             NOT NULL,
    value      double precision NOT NULL,
    PRIMARY KEY (model_name, version, metric)
);

-- The churn model's backtest: how it compared with the recency baseline at each time origin.
CREATE TABLE IF NOT EXISTS ml.monitor_backtest (
    version        integer          NOT NULL,
    origin         date             NOT NULL,
    customers      integer          NOT NULL,
    base_rate      double precision NOT NULL,
    model_pr_auc   double precision NOT NULL,
    recency_pr_auc double precision NOT NULL,
    gain           double precision NOT NULL,
    gain_ci_low    double precision NOT NULL,
    gain_ci_high   double precision NOT NULL,
    significant    boolean          NOT NULL,
    PRIMARY KEY (version, origin)
);
