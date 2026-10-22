"""Entry point and orchestration for the end to end MLE and MLOps project."""

import argparse
import logging
from collections.abc import Sequence

from src import config, database, extractor, lineage, optimizer, processor
from src.errors import PipelineError

log = logging.getLogger("pipeline")

# Airflow's BashOperator marks a task skipped (not failed) when it exits with 99 by default
PROMOTION_REJECTED_EXIT_CODE = 99


def load_raw() -> None:
    """Reload raw.online_retail from the Excel file (full refresh, all-or-nothing)."""
    # Parse the whole workbook first (minutes): a bad file fails here, and nothing is locked while
    # the slow part runs. The table is only truncated once the data is ready to be copied in.
    sheets = list(extractor.extract(config.DATA_PATH))
    file_sha256 = lineage.sha256_file(config.DATA_PATH)

    with database.get_connection() as conn:
        database.run_sql_dir(conn, config.SCHEMA_DIR)

        with conn.transaction():
            conn.execute(f"TRUNCATE {config.RAW_TABLE}")
            for df in sheets:
                sheet = df["source_sheet"].iloc[0]
                loaded = database.copy_dataframe(conn, df, config.RAW_TABLE)
                if loaded != len(df):
                    raise PipelineError(f"Sheet {sheet!r}: loaded {loaded} of {len(df)} rows")
                log.info("Loaded %s rows from sheet %r", f"{loaded:,}", sheet)
            total = database.fetch_scalar(conn, f"SELECT count(*) FROM {config.RAW_TABLE}")
            lineage.record_load(conn, config.DATA_PATH, file_sha256, total)

        log.info("%s now holds %s rows", config.RAW_TABLE, f"{total:,}")


def transform() -> None:
    """Rebuild staging and marts from raw; a failing quality check leaves the old tables intact."""
    with database.get_connection() as conn:
        database.require_rows(
            conn, config.RAW_TABLE, "Run the load stage first: python -m src.main --stage load"
        )
        processor.rebuild(conn)  # staging + marts + quality checks, published only if all pass

        for snapshot_date, split, customers, churn_rate in processor.snapshot_summary(conn):
            log.info(
                "Snapshot %s (%s): %s customers, churn rate %s",
                snapshot_date,
                split,
                f"{customers:,}",
                churn_rate,
            )


def train() -> None:
    """Train the churn, value and segmentation models and register them in MLflow."""
    from src import model  # imported here: pulls in scikit-learn and MLflow

    with database.get_connection() as conn:
        database.run_sql_dir(conn, config.SCHEMA_DIR)  # makes sure the model output tables exist
        database.require_rows(
            conn,
            "marts.customer_snapshots",
            "Run the transform stage first: python -m src.main --stage transform",
        )
        model.train_all(conn)


def score() -> None:
    """Score the customers in the models' population with the live models into ml.predictions."""
    from src import scoring, tracking  # imported here: pulls in scikit-learn and MLflow

    with database.get_connection() as conn:
        database.run_sql_dir(conn, config.SCHEMA_DIR)  # makes sure ml.predictions exists
        database.require_rows(
            conn,
            "marts.customer_snapshots",
            "Run the transform stage first: python -m src.main --stage transform",
        )
        scoring.score_batch(conn, tracking.configure())


def plan(outreach_size: int, holdout_fraction: float) -> None:
    """Choose who to contact from the latest scores and store the plan in ml.outreach_*."""
    with database.get_connection() as conn:
        database.run_sql_dir(conn, config.SCHEMA_DIR)  # makes sure the ml tables exist
        optimizer.make_plan(conn, outreach_size, holdout_fraction)


def monitor() -> None:
    """Refresh the aggregate tables the monitoring dashboards read (ml.monitor_*)."""
    from src import monitoring, tracking  # imported here: pulls in MLflow

    with database.get_connection() as conn:
        database.run_sql_dir(conn, config.SCHEMA_DIR)  # makes sure the monitoring tables exist
        database.require_rows(
            conn,
            "marts.customer_snapshots",
            "Run the transform stage first: python -m src.main --stage transform",
        )
        counts = monitoring.refresh(conn, tracking.configure())
    for table, rows in counts.items():
        log.info("%s now holds %s rows", table, f"{rows:,}")


def report_promotions() -> int:
    """Say, per model, whether the newest version was promoted. Returns the exit code.

    Training succeeds when the gate rejects a model (the old champion keeps serving), so a
    scheduler cannot see that from the train stage's exit status. This stage can: it exits with
    PROMOTION_REJECTED_EXIT_CODE when any model was not promoted.
    """
    from src import promotion_status, tracking  # imported here: pulls in MLflow

    outcomes = promotion_status.latest_outcomes(tracking.configure())
    for outcome in outcomes:
        log.info("%s", promotion_status.describe(outcome))
    rejected = [outcome.model for outcome in outcomes if not outcome.promoted]
    if rejected:
        log.warning("Not promoted: %s. The previous champions keep serving.", ", ".join(rejected))
        return PROMOTION_REJECTED_EXIT_CODE
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stages. Returns the process exit code (0 success, 1 expected failure,
    2 unexpected error, 99 from the report stage when a model was not promoted)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["load", "transform", "train", "score", "plan", "monitor", "report", "all"],
        default="all",
        help=(
            "load: Excel -> raw; transform: raw -> staging -> marts; "
            "train: marts -> models in MLflow; score: live models -> ml.predictions; "
            "plan: scores -> who to contact, with a random holdout, in ml.outreach_*; "
            "monitor: refresh the aggregate tables the monitoring dashboards read (ml.monitor_*); "
            "report: say whether each model's newest version was promoted (exit "
            f"{PROMOTION_REJECTED_EXIT_CODE} if any was not); not part of all; "
            "all: load, transform, train, score, plan and monitor in order (default)"
        ),
    )
    parser.add_argument(
        "--outreach-size",
        type=int,
        default=optimizer.DEFAULT_OUTREACH_SIZE,
        help="plan stage: how many customers to select (default %(default)s)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=optimizer.DEFAULT_HOLDOUT_FRACTION,
        help="plan stage: share of the selected customers deliberately not contacted, "
        "so impact can be measured later (default %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.stage == "report":
            return report_promotions()
        if args.stage in ("load", "all"):
            load_raw()
        if args.stage in ("transform", "all"):
            transform()
        if args.stage in ("train", "all"):
            train()
        if args.stage in ("score", "all"):
            score()
        if args.stage in ("plan", "all"):
            plan(args.outreach_size, args.holdout_fraction)
        if args.stage in ("monitor", "all"):
            monitor()
    except PipelineError as exc:
        log.error("%s", exc)  # an explained, fixable failure: message only, no traceback
        return 1
    except Exception:
        log.exception("Unexpected error")  # a bug: keep the traceback
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
