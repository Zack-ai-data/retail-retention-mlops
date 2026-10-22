-- Drift of the models' inputs and of their churn probabilities, written by the `monitor` stage
-- (src/monitoring.py). Each row compares one snapshot with the snapshot about a year earlier (the
-- same time of year, so the seasons do not count as drift) for one column, using the population
-- stability index (PSI). It holds one number per column, never a customer row. Idempotent.

CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.monitor_drift (
    snapshot_date  date             NOT NULL,  -- the snapshot being checked
    reference_date date             NOT NULL,  -- the snapshot about a year earlier it is compared with
    column_name    text             NOT NULL,  -- a model input, or p_churn (the churn probability)
    psi            double precision NOT NULL,
    status         text             NOT NULL CHECK (status IN ('ok', 'warn', 'alert')),
    PRIMARY KEY (snapshot_date, column_name)
);

COMMENT ON TABLE ml.monitor_drift IS
    'PSI of each model input and of p_churn against the same season a year earlier. ok < 0.1, warn 0.1 to 0.25, alert >= 0.25 (the usual convention). A snapshot with no snapshot a year earlier has no rows: there is nothing to compare it with yet.';
