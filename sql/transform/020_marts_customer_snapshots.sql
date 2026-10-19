-- Rebuilds the customer feature table used for retention modelling (full refresh).
--
-- One row per (snapshot_date, customer): features describe the customer using ONLY
-- transactions strictly before snapshot_date; the label_* columns describe what the
-- customer did in the following horizon_days days. Features and labels never overlap
-- in time, which prevents data leakage.
--
-- Money is NET of returns. A cancelled order must not count as a purchase: one customer's
-- 77,183.60 order was cancelled the same day, which made them look like the best retained
-- customer in the data. So:
--   monetary           = gross_sales - return_value (sales minus returns, to date)
--   spend_last_90d     = sales minus returns booked in the 90 days before the snapshot
--   label_future_spend = sales minus returns booked in the label window, never below 0
--
-- Churn definition: label_churned = the customer has no net-positive spend in
-- [snapshot_date, snapshot_date + horizon_days), i.e. label_future_spend = 0. Only
-- customers with at least one purchase before the snapshot are included.
--
-- Two kinds of history features:
--   * lifetime columns (frequency, monetary, tenure_days, ...) accumulate from the start of
--     the data, so they cover 3 months at the first snapshot and 21 months at the test
--     snapshot. They are kept for reporting and customer segmentation.
--   * *_365d columns count only the 365 days before the snapshot, so every snapshot is
--     measured the same way. history_days_available (capped at 365) says how much of that
--     window actually exists in the data. These are the model inputs.
-- season_sin / season_cos encode the time of year of the snapshot, so a model can learn the
-- holiday swing from the previous year's snapshots.
--
-- is_actionable marks the customers a retention campaign would target: those who bought
-- within the last 180 days (two horizons). Customers silent for longer are lapsed, a
-- different problem (reactivation) whose churn is close to certain, and they dominate the
-- population (47% at the test snapshot). The churn and value models train, are
-- gated and report on actionable customers only; src/features.py holds the same threshold.
--
-- Snapshots are exactly 90 days apart, so label windows are adjacent and never overlap, and
-- they move forward in time. The last labelled one is the test set, so a model is always
-- evaluated on a period later than anything it trained on. The 2010-09-14 snapshot is the
-- pre-Christmas period of the previous year (the test snapshot, 2011-09-09, is 360 days later).
-- The data ends on 2011-12-09, so the last label window ends on 2011-12-08, inside the data.
--
-- The 'score' snapshot (2011-12-10) describes every customer as of the end of the data.
-- Its future has not happened yet, so its label_* columns are NULL. It is what a
-- deployed model scores, and what customer segmentation is based on.

DROP TABLE IF EXISTS marts.customer_snapshots;
DROP TABLE IF EXISTS marts.snapshot_dates;

CREATE TABLE marts.snapshot_dates (
    snapshot_date date    PRIMARY KEY,
    split         text    NOT NULL CHECK (split IN ('train', 'test', 'score')),
    horizon_days  integer NOT NULL CHECK (horizon_days > 0)
);

INSERT INTO marts.snapshot_dates (snapshot_date, split, horizon_days) VALUES
    ('2010-03-18', 'train', 90),
    ('2010-06-16', 'train', 90),
    ('2010-09-14', 'train', 90),
    ('2010-12-13', 'train', 90),
    ('2011-03-13', 'train', 90),
    ('2011-06-11', 'train', 90),
    ('2011-09-09', 'test',  90),
    ('2011-12-10', 'score', 90);

CREATE TABLE marts.customer_snapshots AS
WITH bounds AS (
    SELECT min(invoice_date)::date AS data_start FROM staging.transactions
),
tx AS (
    SELECT customer_id, invoice, invoice_date, line_type, stock_code, quantity, line_total, country
    FROM staging.transactions
    WHERE customer_id IS NOT NULL
      AND line_type IN ('sale', 'return')
),
history AS (
    SELECT s.snapshot_date,
           s.split,
           s.horizon_days,
           t.customer_id,
           -- lifetime (since the start of the data)
           (min(t.invoice_date) FILTER (WHERE t.line_type = 'sale'))::date AS first_purchase_date,
           (max(t.invoice_date) FILTER (WHERE t.line_type = 'sale'))::date AS last_purchase_date,
           count(DISTINCT t.invoice) FILTER (WHERE t.line_type = 'sale')   AS frequency,
           sum(t.line_total)         FILTER (WHERE t.line_type = 'sale')   AS gross_sales,
           sum(t.quantity)           FILTER (WHERE t.line_type = 'sale')   AS total_units,
           count(DISTINCT t.stock_code) FILTER (WHERE t.line_type = 'sale') AS distinct_products,
           count(DISTINCT t.invoice) FILTER (WHERE t.line_type = 'return') AS return_invoices,
           coalesce(-sum(t.line_total) FILTER (WHERE t.line_type = 'return'), 0) AS return_value,
           -- the 365 days before the snapshot
           count(DISTINCT t.invoice) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ) AS frequency_365d,
           coalesce(sum(t.line_total) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ), 0) AS gross_sales_365d,
           coalesce(-sum(t.line_total) FILTER (
               WHERE t.line_type = 'return' AND t.invoice_date >= s.snapshot_date - 365
           ), 0) AS return_value_365d,
           coalesce(sum(t.quantity) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ), 0) AS total_units_365d,
           count(DISTINCT t.stock_code) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ) AS distinct_products_365d,
           count(DISTINCT t.invoice) FILTER (
               WHERE t.line_type = 'return' AND t.invoice_date >= s.snapshot_date - 365
           ) AS return_invoices_365d,
           (min(t.invoice_date) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ))::date AS first_purchase_date_365d,
           (max(t.invoice_date) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 365
           ))::date AS last_purchase_date_365d,
           -- recent activity
           count(DISTINCT t.invoice) FILTER (
               WHERE t.line_type = 'sale' AND t.invoice_date >= s.snapshot_date - 90
           ) AS orders_last_90d,
           count(DISTINCT t.invoice) FILTER (
               WHERE t.line_type = 'sale'
                 AND t.invoice_date >= s.snapshot_date - 180
                 AND t.invoice_date <  s.snapshot_date - 90
           ) AS orders_prev_90d,
           coalesce(sum(t.line_total) FILTER (
               WHERE t.invoice_date >= s.snapshot_date - 90
           ), 0) AS spend_last_90d,  -- sales and returns together, so net
           (array_agg(t.country ORDER BY t.invoice_date DESC, t.invoice DESC))[1] AS country
    FROM marts.snapshot_dates s
    JOIN tx t ON t.invoice_date < s.snapshot_date
    GROUP BY s.snapshot_date, s.split, s.horizon_days, t.customer_id
    HAVING count(*) FILTER (WHERE t.line_type = 'sale') > 0
),
future AS (
    SELECT s.snapshot_date,
           t.customer_id,
           count(DISTINCT t.invoice) FILTER (WHERE t.line_type = 'sale') AS future_orders,
           sum(t.line_total)                                             AS future_net_spend
    FROM marts.snapshot_dates s
    JOIN tx t
      ON t.invoice_date >= s.snapshot_date
     AND t.invoice_date <  s.snapshot_date + s.horizon_days
    GROUP BY s.snapshot_date, t.customer_id
)
SELECT h.snapshot_date,
       h.split,
       h.customer_id,
       -- features (known at snapshot_date)
       h.first_purchase_date,
       h.last_purchase_date,
       h.snapshot_date - h.last_purchase_date  AS recency_days,
       h.snapshot_date - h.last_purchase_date <= 180 AS is_actionable,
       h.snapshot_date - h.first_purchase_date AS tenure_days,
       least(h.snapshot_date - h.first_purchase_date, 365) AS tenure_days_capped,
       least(h.snapshot_date - b.data_start, 365)          AS history_days_available,
       -- lifetime
       h.frequency,
       h.gross_sales,
       h.gross_sales - h.return_value          AS monetary,
       (h.gross_sales - h.return_value) / h.frequency AS avg_order_value,
       h.total_units,
       h.distinct_products,
       CASE WHEN h.frequency > 1
            THEN (h.last_purchase_date - h.first_purchase_date)::numeric / (h.frequency - 1)
       END                                     AS avg_days_between_orders,
       h.return_invoices,
       h.return_value,
       h.return_value / nullif(h.gross_sales, 0) AS return_rate,
       -- the 365 days before the snapshot
       h.frequency_365d,
       h.gross_sales_365d,
       h.gross_sales_365d - h.return_value_365d AS monetary_365d,
       (h.gross_sales_365d - h.return_value_365d) / nullif(h.frequency_365d, 0) AS avg_order_value_365d,
       h.total_units_365d,
       h.distinct_products_365d,
       CASE WHEN h.frequency_365d > 1
            THEN (h.last_purchase_date_365d - h.first_purchase_date_365d)::numeric
                 / (h.frequency_365d - 1)
       END                                     AS avg_days_between_orders_365d,
       h.return_invoices_365d,
       h.return_value_365d,
       h.return_value_365d / nullif(h.gross_sales_365d, 0) AS return_rate_365d,
       -- recent activity
       h.orders_last_90d,
       h.orders_prev_90d,
       h.spend_last_90d,
       h.country,
       h.country = 'United Kingdom'            AS is_uk,
       -- time of year of the snapshot
       sin(2 * pi() * extract(doy FROM h.snapshot_date) / 365.25) AS season_sin,
       cos(2 * pi() * extract(doy FROM h.snapshot_date) / 365.25) AS season_cos,
       -- labels (what happens in the next horizon_days days; never use as features).
       -- NULL for the 'score' snapshot: its future is unknown.
       CASE WHEN h.split = 'score' THEN NULL ELSE coalesce(f.future_orders, 0) END AS label_future_orders,
       CASE WHEN h.split = 'score' THEN NULL
            ELSE greatest(coalesce(f.future_net_spend, 0), 0) END                 AS label_future_spend,
       CASE WHEN h.split = 'score' THEN NULL
            ELSE greatest(coalesce(f.future_net_spend, 0), 0) = 0 END             AS label_churned
FROM history h
CROSS JOIN bounds b
LEFT JOIN future f USING (snapshot_date, customer_id);

ALTER TABLE marts.customer_snapshots ADD PRIMARY KEY (snapshot_date, customer_id);

COMMENT ON TABLE marts.customer_snapshots IS
    'Customer features at each snapshot_date (money net of returns; lifetime columns plus *_365d windowed columns and seasonal terms) plus forward-looking label_* columns (net spend, churn, order count).';

ANALYZE marts.customer_snapshots;
