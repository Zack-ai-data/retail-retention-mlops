-- Output of batch scoring (src/scoring.py). Every scoring run is kept, so monitoring can compare
-- runs and a score can be traced to the model versions that produced it. Idempotent.

CREATE TABLE IF NOT EXISTS ml.scoring_runs (
    scoring_run_id      bigserial        PRIMARY KEY,
    scored_at           timestamptz      NOT NULL DEFAULT now(),
    snapshot_date       date             NOT NULL,  -- the customers' features are as of this date
    churn_model_version text             NOT NULL,
    value_model_version text             NOT NULL,
    margin              double precision NOT NULL,  -- assumption used for margin_at_risk
    contact_cost        double precision NOT NULL,  -- assumption used for margin_at_risk_after_cost
    customers           integer          NOT NULL
);

-- One row per scored customer. Only customers in the models' population (bought within the last
-- 180 days) are scored; lapsed customers are a separate win-back list.
CREATE TABLE IF NOT EXISTS ml.predictions (
    scoring_run_id            bigint           NOT NULL REFERENCES ml.scoring_runs,
    customer_id               integer          NOT NULL,
    p_churn                   double precision NOT NULL CHECK (p_churn BETWEEN 0 AND 1),
    spend_if_retained         double precision NOT NULL CHECK (spend_if_retained >= 0),
    expected_spend            double precision NOT NULL CHECK (expected_spend >= 0),
    margin_at_risk            double precision NOT NULL,
    margin_at_risk_after_cost double precision NOT NULL,
    PRIMARY KEY (scoring_run_id, customer_id)
);

COMMENT ON COLUMN ml.predictions.p_churn IS 'P(no net-positive spend in the next 90 days)';
COMMENT ON COLUMN ml.predictions.spend_if_retained IS 'Expected 90-day spend IF the customer buys (the spend stage of the value model)';
COMMENT ON COLUMN ml.predictions.expected_spend IS 'Expected 90-day spend = P(buy) x spend_if_retained';
COMMENT ON COLUMN ml.predictions.margin_at_risk IS 'p_churn x spend_if_retained x margin: the margin expected to be lost. An upper bound on what contact could save: there is no campaign data, so no uplift is claimed';
COMMENT ON COLUMN ml.predictions.margin_at_risk_after_cost IS 'margin_at_risk minus the cost of contacting the customer';

-- The most recent scoring run, with the columns people usually want
CREATE OR REPLACE VIEW ml.latest_predictions AS
SELECT r.scoring_run_id, r.scored_at, r.snapshot_date, r.churn_model_version,
       r.value_model_version, p.customer_id, p.p_churn, p.spend_if_retained, p.expected_spend,
       p.margin_at_risk, p.margin_at_risk_after_cost
FROM ml.predictions p
JOIN ml.scoring_runs r USING (scoring_run_id)
WHERE r.scoring_run_id = (SELECT max(scoring_run_id) FROM ml.scoring_runs);
