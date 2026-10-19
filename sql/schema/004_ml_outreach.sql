-- Outreach plans (src/optimizer.py): who is to be contacted and who is deliberately held out.
-- A plan is stored so the random holdout group is fixed and can be audited later. Idempotent.

CREATE TABLE IF NOT EXISTS ml.outreach_plans (
    plan_id            bigserial        PRIMARY KEY,
    created_at         timestamptz      NOT NULL DEFAULT now(),
    scoring_run_id     bigint           NOT NULL REFERENCES ml.scoring_runs,
    k                  integer          NOT NULL CHECK (k >= 1),
    holdout_fraction   double precision NOT NULL CHECK (holdout_fraction >= 0 AND holdout_fraction < 1),
    seed               integer          NOT NULL,
    margin             double precision NOT NULL,  -- assumptions the scores were computed with
    contact_cost       double precision NOT NULL,
    contact_customers  integer          NOT NULL,
    holdout_customers  integer          NOT NULL
);

CREATE TABLE IF NOT EXISTS ml.outreach_assignments (
    plan_id     bigint  NOT NULL REFERENCES ml.outreach_plans,
    customer_id integer NOT NULL,
    rank        integer NOT NULL,  -- 1 = highest margin at risk
    grp         text    NOT NULL CHECK (grp IN ('contact', 'holdout', 'not_selected')),
    PRIMARY KEY (plan_id, customer_id)
);

COMMENT ON COLUMN ml.outreach_assignments.grp IS
    'contact = selected and contacted; holdout = selected but deliberately not contacted (random sample, to measure impact later); not_selected = outside the top k or not worth the contact cost';
