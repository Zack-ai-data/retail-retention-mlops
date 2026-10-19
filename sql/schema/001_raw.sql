-- Idempotent: safe to run on every pipeline start (applied by src/main.py).

CREATE SCHEMA IF NOT EXISTS raw;      -- untouched copy of the source data
CREATE SCHEMA IF NOT EXISTS staging;  -- cleaned / deduplicated data
CREATE SCHEMA IF NOT EXISTS marts;    -- features and analytics tables
CREATE SCHEMA IF NOT EXISTS ml;       -- model runs, predictions, monitoring

-- One row per row of the source Excel file. No cleaning happens here:
-- cancellations (invoice starts with 'C'), bad-debt adjustments ('A'),
-- negative quantities/prices and missing customer ids are all kept as-is.
CREATE TABLE IF NOT EXISTS raw.online_retail (
    source_sheet text        NOT NULL,  -- Excel sheet the row came from
    source_row   integer     NOT NULL,  -- Excel row number (header is row 1)
    invoice      text        NOT NULL,
    stock_code   text        NOT NULL,
    description  text,
    quantity     integer     NOT NULL,
    invoice_date timestamp   NOT NULL,
    price        numeric     NOT NULL,
    customer_id  integer,
    country      text        NOT NULL,
    loaded_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_sheet, source_row)
);

-- One row per load of the raw table: which file was loaded and what it contained. Training reads
-- the latest row so that every model can be traced to the exact data it was built from.
CREATE TABLE IF NOT EXISTS raw.load_log (
    load_id     bigserial   PRIMARY KEY,
    loaded_at   timestamptz NOT NULL DEFAULT now(),
    source_file text        NOT NULL,
    file_sha256 text        NOT NULL,
    row_count   bigint      NOT NULL
);

COMMENT ON TABLE raw.online_retail IS
    'UCI Online Retail II, loaded as-is. The two sheets overlap on 2010-12-01..2010-12-09, so duplicates across sheets are expected.';
