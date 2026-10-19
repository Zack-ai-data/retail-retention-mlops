-- Rebuilds staging.transactions from raw.online_retail (full refresh, safe to re-run).
--
-- Cleaning rules (all verified against the data, see README "Data notes"):
--   1. Sheet overlap: the two Excel sheets share 2010-12-01..2010-12-09 as an exact copy.
--      Each invoice is taken from the earliest sheet that contains it. Sheet names
--      sort chronologically ('Year 2009-2010' < 'Year 2010-2011').
--   2. Identical lines inside one sheet are real repeat scans (e.g. 20 lines of qty 1),
--      so they are kept and flagged with is_repeated_line instead of being dropped.
--   3. Text is normalised (stock codes upper-cased and trimmed, blank descriptions -> NULL).
--   4. No row is removed for being odd; every row gets a line_type so downstream
--      queries choose what to use and counts can always be reconciled with raw.
--
-- line_type:
--   non_product       postage, fees, discounts, manual entries, vouchers, test items
--   adjustment        bad-debt adjustment invoices (invoice starts with 'A')
--   return            cancellation invoices (invoice starts with 'C')
--   sale              normal purchase line (quantity > 0 and price > 0)
--   stock_adjustment  anything else on a normal invoice: write-offs (quantity <= 0, no
--                     customer) and zero-price lines

DROP TABLE IF EXISTS staging.transactions CASCADE;

CREATE TABLE staging.transactions AS
WITH first_sheet AS (
    SELECT invoice, min(source_sheet) AS source_sheet
    FROM raw.online_retail
    GROUP BY invoice
),
deduped AS (
    SELECT r.source_sheet,
           r.source_row,
           r.invoice,
           upper(btrim(r.stock_code))        AS stock_code,
           nullif(btrim(r.description), '')  AS description,
           r.quantity,
           r.invoice_date,
           r.price                           AS unit_price,
           r.customer_id,
           btrim(r.country)                  AS country
    FROM raw.online_retail r
    JOIN first_sheet f
      ON f.invoice = r.invoice AND f.source_sheet = r.source_sheet
)
SELECT d.source_sheet,
       d.source_row,
       d.invoice,
       d.stock_code,
       d.description,
       d.quantity,
       d.invoice_date,
       d.unit_price,
       d.quantity * d.unit_price AS line_total,
       d.customer_id,
       d.country,
       CASE
           WHEN d.stock_code IN ('POST', 'DOT', 'M', 'C2', 'C3', 'D', 'S', 'BANK CHARGES',
                                 'ADJUST', 'ADJUST2', 'AMAZONFEE', 'CRUK', 'B',
                                 'TEST001', 'TEST002', 'GIFT')
                OR d.stock_code LIKE 'GIFT\_0001\_%'  THEN 'non_product'
           WHEN d.invoice LIKE 'A%'                   THEN 'adjustment'
           WHEN d.invoice LIKE 'C%'                   THEN 'return'
           WHEN d.quantity > 0 AND d.unit_price > 0   THEN 'sale'
           ELSE 'stock_adjustment'
       END AS line_type,
       row_number() OVER (
           PARTITION BY d.invoice, d.stock_code, d.description, d.quantity,
                        d.invoice_date, d.unit_price, d.customer_id, d.country
           ORDER BY d.source_row
       ) > 1 AS is_repeated_line
FROM deduped d;

ALTER TABLE staging.transactions ADD PRIMARY KEY (source_sheet, source_row);
CREATE INDEX transactions_customer_date_idx ON staging.transactions (customer_id, invoice_date);
CREATE INDEX transactions_invoice_idx       ON staging.transactions (invoice);

COMMENT ON TABLE staging.transactions IS
    'Cleaned raw.online_retail: sheet overlap removed, text normalised, every row tagged with line_type.';

ANALYZE staging.transactions;
