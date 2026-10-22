"""Aggregate tables for the monitoring dashboards (the `monitor` stage).

Grafana reads these tables, not the customer-level tables and not MLflow's internals, so what a
dashboard can show is fixed here, in code that is tested, and the database login Grafana uses
never needs access to a row about a customer.

  from the warehouse   ml.monitor_snapshot_summary   churn rate, spend and customer counts per
                                                    snapshot
                       ml.monitor_score_distribution how the churn probabilities of each scoring
                                                    run are spread
                       ml.monitor_segment_sizes      customers per segment
  from MLflow          ml.monitor_model_versions     every version of the project's models and
                                                    what the promotion gate decided
                       ml.monitor_model_metrics      every metric logged for those versions
                       ml.monitor_backtest           the churn model's backtest, per time origin

The tables are replaced as a whole in one transaction. They are plain tables and not views because
the transform stage drops and rebuilds the marts tables on every run, which a view over them would
block or be deleted by (see sql/schema/005_ml_monitoring.sql).
"""

import logging
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from src import config, database, promotion_status, tracking

log = logging.getLogger(__name__)

BACKTEST_ARTIFACT = "backtest.json"
BACKTEST_COLUMNS = [
    "version",
    "origin",
    "customers",
    "base_rate",
    "model_pr_auc",
    "recency_pr_auc",
    "gain",
    "gain_ci_low",
    "gain_ci_high",
    "significant",
]
VERSION_COLUMNS = ["model_name", "version", "created_at", "decision", "reason", "is_champion"]
METRIC_COLUMNS = ["model_name", "version", "metric", "value"]
MAX_REASON_CHARS = 600  # a decision's reason lists every check; the start says what mattered

# Every table, in the order they are replaced
TABLES = [
    "ml.monitor_snapshot_summary",
    "ml.monitor_score_distribution",
    "ml.monitor_segment_sizes",
    "ml.monitor_model_versions",
    "ml.monitor_model_metrics",
    "ml.monitor_backtest",
]

_SNAPSHOT_SUMMARY = """
    INSERT INTO ml.monitor_snapshot_summary
        (snapshot_date, split, customers, actionable_customers, churn_rate, avg_future_spend)
    SELECT snapshot_date,
           split,
           count(*),
           count(*) FILTER (WHERE is_actionable),
           avg(label_churned::int) FILTER (WHERE is_actionable)::float8,
           avg(label_future_spend) FILTER (WHERE is_actionable)::float8
    FROM marts.customer_snapshots
    GROUP BY snapshot_date, split
"""

# Every bucket of every run is present, empty ones with zero customers, so a chart has no gaps.
# p_churn is in [0, 1]; a probability of exactly 1 belongs to the top bucket.
_SCORE_DISTRIBUTION = """
    INSERT INTO ml.monitor_score_distribution
        (scoring_run_id, bucket, bucket_label, customers, margin_at_risk)
    SELECT r.scoring_run_id,
           b.bucket,
           to_char(b.bucket / 10.0, 'FM0.0') || '-' || to_char((b.bucket + 1) / 10.0, 'FM0.0'),
           count(p.customer_id),
           coalesce(sum(p.margin_at_risk), 0)::float8
    FROM ml.scoring_runs r
    CROSS JOIN generate_series(0, 9) AS b(bucket)
    LEFT JOIN ml.predictions p
           ON p.scoring_run_id = r.scoring_run_id
          AND least(floor(p.p_churn * 10)::int, 9) = b.bucket
    GROUP BY r.scoring_run_id, b.bucket
"""

_SEGMENT_SIZES = """
    INSERT INTO ml.monitor_segment_sizes (snapshot_date, segment_name, customers)
    SELECT snapshot_date, segment_name, count(*)
    FROM marts.customer_segments
    GROUP BY snapshot_date, segment_name
"""


def _is_missing(exc: MlflowException) -> bool:
    """Whether MLflow said the thing asked for does not exist (a model, a run or an artifact).

    Only that is treated as "nothing there". Any other failure (the server is down, a permission
    problem) is raised: swallowing it would write empty tables and report success, and the
    dashboards would go blank without anyone being told.
    """
    return bool(exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST))


def _by_number(versions: Iterable[ModelVersion]) -> list[ModelVersion]:
    """Oldest first (the registry lists the newest first), so the output does not depend on it."""
    return sorted(versions, key=lambda version: int(version.version))


def _decision(tags: dict[str, str]) -> tuple[str, str]:
    """The gate's decision and its reason, from a model version's `promotion` tag."""
    text = tags.get("promotion")
    if text is None:
        return "unrecorded", "no promotion decision was recorded for this version"
    for decision in ("promoted", "rejected"):
        if text.startswith(f"{decision}:"):
            return decision, text.split(":", 1)[1].strip()[:MAX_REASON_CHARS]
    return "unrecorded", f"unrecognised decision: {text[:MAX_REASON_CHARS]}"


def model_tables(client: MlflowClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per version of each project model with its gate decision, and every metric logged
    for those versions."""
    versions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for name in promotion_status.MODEL_NAMES:
        try:
            registered = client.get_registered_model(name)
        except MlflowException as exc:
            if not _is_missing(exc):
                raise
            continue  # the model has not been trained yet
        champion = registered.aliases.get(tracking.CHAMPION_ALIAS)
        for version in _by_number(client.search_model_versions(f"name='{name}'")):
            decision, reason = _decision(dict(version.tags))
            versions.append(
                {
                    "model_name": name,
                    "version": int(version.version),
                    "created_at": datetime.fromtimestamp(version.creation_timestamp / 1000, UTC),
                    "decision": decision,
                    "reason": reason,
                    "is_champion": champion is not None and str(champion) == str(version.version),
                }
            )
            if version.run_id:
                try:
                    logged = client.get_run(version.run_id).data.metrics
                except MlflowException as exc:
                    if not _is_missing(exc):
                        raise
                    logged = {}  # the run was deleted; the version is still worth listing
                metrics.extend(
                    {
                        "model_name": name,
                        "version": int(version.version),
                        "metric": metric,
                        "value": float(value),
                    }
                    for metric, value in logged.items()
                    if pd.notna(value)
                )
    return (
        pd.DataFrame(versions, columns=VERSION_COLUMNS),
        pd.DataFrame(metrics, columns=METRIC_COLUMNS),
    )


def backtest_table(client: MlflowClient) -> pd.DataFrame:
    """The churn model's backtest at each time origin, for every version that saved one.

    Training saves the table as an artifact (it is not a metric). Versions trained before the
    backtest existed have none, which is normal and skipped.
    """
    frames = []
    versions = _by_number(client.search_model_versions(f"name='{config.CHURN_MODEL_NAME}'"))
    for version in versions:
        if not version.run_id:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = client.download_artifacts(version.run_id, BACKTEST_ARTIFACT, tmp)
            except MlflowException as exc:
                if not _is_missing(exc):
                    raise
                log.info("Version %s has no %s; skipped", version.version, BACKTEST_ARTIFACT)
                continue
            table = pd.read_json(Path(path), orient="split", convert_dates=False)
        table.insert(0, "version", int(version.version))
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)
    return pd.concat(frames, ignore_index=True)[BACKTEST_COLUMNS]


def refresh(conn: psycopg.Connection[Any], client: MlflowClient) -> dict[str, int]:
    """Replace every monitoring table. Returns the number of rows each now holds.

    Everything is collected from MLflow first, so a registry that cannot be read fails before
    anything is written. The tables are then replaced in one transaction: if any step fails the
    previous contents stay exactly as they were.
    """
    versions, metrics = model_tables(client)
    backtest = backtest_table(client)

    with conn.transaction():
        for table in reversed(TABLES):
            conn.execute(f"DELETE FROM {table}")
        conn.execute(_SNAPSHOT_SUMMARY)
        conn.execute(_SCORE_DISTRIBUTION)
        conn.execute(_SEGMENT_SIZES)
        database.copy_dataframe(conn, versions, "ml.monitor_model_versions")
        database.copy_dataframe(conn, metrics, "ml.monitor_model_metrics")
        database.copy_dataframe(conn, backtest, "ml.monitor_backtest")
        counts = {
            table: int(database.fetch_scalar(conn, f"SELECT count(*) FROM {table}"))
            for table in TABLES
        }
    conn.commit()
    return counts
