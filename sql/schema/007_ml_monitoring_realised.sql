-- How well the churn probabilities that were SERVED did against what the customers then did, written
-- by the `monitor` stage (src/monitoring.py). One row per scoring run whose snapshot has outcome
-- labels (the 90-day window has ended). It holds one number per measure, never a customer row.
-- Idempotent.
--
-- A scoring run on the newest snapshot has no labels until its 90 days are over, and in this project
-- the dataset is static and ends in December 2011, so the live scoring runs never get them: the
-- table is empty on the real data and is shown to work by tests.

CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.monitor_realised (
    scoring_run_id      bigint           PRIMARY KEY,
    snapshot_date       date             NOT NULL,
    churn_model_version text             NOT NULL,
    customers           integer          NOT NULL,  -- scored customers whose outcome is known
    actual_churn_rate   double precision NOT NULL,
    mean_p_churn        double precision NOT NULL,  -- the average probability that was served
    calibration_gap     double precision NOT NULL,  -- mean_p_churn - actual_churn_rate; > 0 over-predicts
    ece                 double precision NOT NULL,  -- expected calibration error, 10 equal-size bins
    model_pr_auc        double precision NOT NULL,
    recency_pr_auc      double precision NOT NULL,  -- the "longer since the last order" rule
    gain                double precision NOT NULL,  -- model_pr_auc - recency_pr_auc
    gain_ci_low         double precision NOT NULL,  -- 95% bootstrap interval of the gain
    gain_ci_high        double precision NOT NULL,
    significant         boolean          NOT NULL,  -- the whole interval is above zero
    status              text             NOT NULL CHECK (status IN ('ok', 'warn', 'alert'))
);

COMMENT ON TABLE ml.monitor_realised IS
    'Realised performance of served churn probabilities. status: alert when |calibration_gap| >= 0.10 or the gain over the recency rule is not clearly above zero, warn when |calibration_gap| >= 0.05, else ok (assumed thresholds, not fitted).';
