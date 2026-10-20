"""Reload the raw data and rebuild the warehouse tables the models are trained from.

    load_raw  ->  transform_marts

load_raw reads the Excel file into raw.online_retail. The load is all-or-nothing, so a failure
leaves the previous data untouched. It is retried once, because a failure can be a passing problem.

transform_marts rebuilds staging and the customer feature table and runs the data quality checks in
one transaction: the new tables are published only if every check passes. A failed check is not
retried, because the same data would fail the same way. Fix the data or the SQL, then run again.

Run by hand: the dataset does not change, so a schedule would reload the same file. When it
succeeds the DAG records that customer_snapshots was refreshed (an Asset), which is what the
retraining DAG waits for.
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import DAG
from pipeline_common import CUSTOMER_SNAPSHOTS, stage

with DAG(
    dag_id="retail_etl",
    description="Reload the raw data and rebuild the feature tables, with data quality checks",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,  # two runs at once would load and rebuild the same tables together
    tags=["etl"],
    doc_md=__doc__,
) as dag:
    load_raw = stage(
        "load_raw",
        "load",
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=30),
    )

    transform_marts = stage(
        "transform_marts",
        "transform",
        retries=0,
        execution_timeout=timedelta(minutes=15),
        outlets=[CUSTOMER_SNAPSHOTS],
    )

    load_raw >> transform_marts
