import logging
from typing import Any

import psycopg

from src import config, database
from src.errors import DataQualityError

log = logging.getLogger(__name__)

# Each check is a query that counts violations; a healthy database returns 0.
QUALITY_CHECKS = {
    "every raw row dropped from staging has an identical copy that was kept": """
        WITH dropped AS (
            SELECT r.*
            FROM raw.online_retail r
            LEFT JOIN staging.transactions s USING (source_sheet, source_row)
            WHERE s.source_row IS NULL
        )
        SELECT count(*) FROM (
            SELECT invoice, upper(btrim(stock_code)), nullif(btrim(description), ''),
                   quantity, invoice_date, price, customer_id, btrim(country)
            FROM dropped
            EXCEPT ALL
            SELECT invoice, stock_code, description,
                   quantity, invoice_date, unit_price, customer_id, country
            FROM staging.transactions
        ) AS unmatched
    """,
    "every invoice comes from exactly one sheet": """
        SELECT count(*) FROM (
            SELECT invoice FROM staging.transactions
            GROUP BY invoice HAVING count(DISTINCT source_sheet) > 1
        ) AS split_invoices
    """,
    "every staging row has a line_type": """
        SELECT count(*) FROM staging.transactions WHERE line_type IS NULL
    """,
    "features only use purchases before the snapshot date": """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE last_purchase_date >= snapshot_date
    """,
    "snapshot customers have a purchase history": """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE frequency < 1 OR recency_days < 0 OR tenure_days < recency_days
    """,
    "no customer is labelled retained without positive net spend in the window": """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE split <> 'score' AND (NOT label_churned) AND label_future_spend <= 0
    """,
    "365-day window features never exceed the lifetime features": """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE frequency_365d > frequency OR gross_sales_365d > gross_sales
           OR return_value_365d > return_value
           OR history_days_available NOT BETWEEN 0 AND 365
    """,
    "only the score snapshot is unlabelled": """
        SELECT count(*) FROM marts.customer_snapshots
        WHERE (split = 'score') <> (label_churned IS NULL)
    """,
}

# Labels are only trustworthy when the whole horizon after the snapshot is in the data.
# (The 'score' snapshot has no labels, so it is exempt.)
LABEL_COVERAGE_CHECK = (
    "every label window ends inside the data",
    """
        SELECT count(*) FROM marts.snapshot_dates
        WHERE split <> 'score'
          AND snapshot_date + horizon_days
              > (SELECT max(invoice_date)::date FROM staging.transactions)
    """,
)


def build_staging(conn: psycopg.Connection[Any], commit: bool = True) -> None:
    """Rebuild staging.transactions from the raw layer."""
    database.run_sql_file(conn, config.TRANSFORM_DIR / "010_staging_transactions.sql", commit)


def build_marts(conn: psycopg.Connection[Any], commit: bool = True) -> None:
    """Rebuild the customer snapshot features and labels from staging."""
    database.run_sql_file(conn, config.TRANSFORM_DIR / "020_marts_customer_snapshots.sql", commit)


def run_quality_checks(
    conn: psycopg.Connection[Any],
    require_label_coverage: bool = True,
    end_transaction: bool = True,
) -> list[str]:
    """Run the data quality checks and return a message for each one that fails.

    end_transaction=False leaves the transaction open, so the checks can look at tables that
    have been rebuilt but not yet committed (see rebuild).
    """
    checks = dict(QUALITY_CHECKS)
    if require_label_coverage:
        name, query = LABEL_COVERAGE_CHECK
        checks[name] = query

    failures = []
    for name, query in checks.items():
        violations = database.fetch_scalar(conn, query)
        if violations:
            failures.append(f"{name}: {violations:,} violations")
        else:
            log.info("Check passed: %s", name)
    if end_transaction:
        conn.rollback()  # read-only checks; end the transaction cleanly
    return failures


def rebuild(conn: psycopg.Connection[Any], require_label_coverage: bool = True) -> None:
    """Rebuild staging and marts and publish them only if every quality check passes.

    The two transform scripts and the checks run in ONE transaction (write, audit, publish):
    if any step fails, everything is rolled back and the previous staging and marts tables stay
    exactly as they were, so a later stage can never train on data that failed its checks.
    """
    try:
        build_staging(conn, commit=False)
        build_marts(conn, commit=False)
        failures = run_quality_checks(conn, require_label_coverage, end_transaction=False)
    except Exception:
        conn.rollback()
        raise
    if failures:
        conn.rollback()
        raise DataQualityError(
            "Data quality checks failed; the previous staging and marts tables were left "
            "unchanged:\n  " + "\n  ".join(failures)
        )
    conn.commit()


def snapshot_summary(conn: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    """Customers and churn rate per snapshot: (snapshot_date, split, customers, churn_rate)."""
    rows = conn.execute(
        """
        SELECT snapshot_date, split, count(*) AS customers,
               round(avg(label_churned::int), 3) AS churn_rate
        FROM marts.customer_snapshots
        GROUP BY snapshot_date, split
        ORDER BY snapshot_date
        """
    ).fetchall()
    conn.rollback()
    return rows
