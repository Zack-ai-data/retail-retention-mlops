"""Retrain the models, say what the promotion gate decided, then score and plan outreach.

    train  ->  report_promotion
           ->  score  ->  plan

Starts by itself when retail_etl has refreshed customer_snapshots (an Asset), or by hand.

train fits the churn, value and segmentation models and registers them in MLflow. The promotion
gate decides whether each new version replaces the champion. A rejection is a normal result, not a
failure: the previous champion keeps serving, so train succeeds either way.

report_promotion makes that visible. It reads the gate's decisions back and logs them per model.
If any model was not promoted the task ends as skipped (yellow), with the reason in its log; it
does not stop score and plan, which carry on with the models that are live. A decision that
cannot be read (a version with no recorded decision) fails the task.

score and plan are retried once: each writes its results in a single transaction, so a failed
attempt leaves nothing behind. Every successful plan run adds a new row to ml.outreach_plans.
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import DAG
from pipeline_common import CUSTOMER_SNAPSHOTS, PROMOTION_REJECTED_EXIT_CODE, stage

with DAG(
    dag_id="retail_retrain_score",
    description="Retrain the models, report the gate's decisions, score customers, plan outreach",
    schedule=[CUSTOMER_SNAPSHOTS],
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,  # two runs at once would register and score the models together
    tags=["ml"],
    doc_md=__doc__,
) as dag:
    train = stage(
        "train",
        "train",
        retries=1,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=60),
    )

    report_promotion = stage(
        "report_promotion",
        "report",
        retries=0,
        execution_timeout=timedelta(minutes=5),
        skip_on_exit_code=PROMOTION_REJECTED_EXIT_CODE,
    )

    score = stage(
        "score",
        "score",
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=15),
    )

    plan = stage(
        "plan",
        "plan",
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=10),
    )

    train >> [report_promotion, score]
    score >> plan
