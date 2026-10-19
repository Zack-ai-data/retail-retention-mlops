"""The command-line pipeline (src/main.py): the whole run, and what each failure looks like."""

import logging

import pandas as pd
import pytest
from mlflow import MlflowClient

from src import config, database, lineage, main, scoring
from tests.conftest import SHEET_1, SHEET_2
from tests.test_scoring import stub_champions


@pytest.fixture
def pipeline_env(scratch_conn, tmp_path, monkeypatch):
    """Point the pipeline at a throwaway database and a throwaway MLflow store."""
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(config, "POSTGRES_DB", scratch_conn.info.dbname)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", mlflow_uri)
    monkeypatch.setattr(config, "DATA_PATH", tmp_path / "missing.xlsx")
    monkeypatch.chdir(tmp_path)  # MLflow writes ./mlruns relative to the working directory
    return scratch_conn, mlflow_uri


# -------------------------------------------------------------- full run


@pytest.mark.e2e
def test_the_whole_pipeline_runs_on_a_small_workbook(pipeline_env, sample_workbook, monkeypatch):
    conn, mlflow_uri = pipeline_env
    monkeypatch.setattr(config, "DATA_PATH", sample_workbook)
    # a workbook this small never produces a model that passes the promotion gate, so there is
    # no champion to score with; stand-in models exercise the scoring stage itself
    monkeypatch.setattr(scoring, "load_champions", lambda client: stub_champions())

    assert main.main([]) == 0

    sheets = pd.read_excel(sample_workbook, sheet_name=None)
    first, second = sheets[SHEET_1], sheets[SHEET_2]
    overlap = second["Invoice"].isin(first["Invoice"])
    assert overlap.any(), "the sample workbook must contain the sheet overlap"

    assert database.fetch_scalar(conn, "SELECT count(*) FROM raw.online_retail") == len(
        first
    ) + len(second)
    # staging keeps each invoice once: the overlap rows of the second sheet are dropped
    assert database.fetch_scalar(conn, "SELECT count(*) FROM staging.transactions") == len(
        first
    ) + int((~overlap).sum())
    # eight snapshots: three earlier training snapshots were added in step 2b
    assert database.fetch_scalar(conn, "SELECT count(*) FROM marts.snapshot_dates") == 8
    scored = database.fetch_scalar(
        conn, "SELECT count(*) FROM marts.customer_snapshots WHERE split = 'score'"
    )
    assert scored > 100
    assert database.fetch_scalar(conn, "SELECT count(*) FROM marts.customer_segments") == scored
    # the score stage stored a prediction for every customer in the models' population
    population = database.fetch_scalar(
        conn,
        "SELECT count(*) FROM marts.customer_snapshots WHERE split = 'score' AND is_actionable",
    )
    assert 0 < population < scored  # lapsed customers are not scored
    assert database.fetch_scalar(conn, "SELECT count(*) FROM ml.latest_predictions") == population
    # ...and the plan stage assigned every one of them to a group, with a holdout
    assert database.fetch_scalar(conn, "SELECT count(*) FROM ml.outreach_plans") == 1
    assert database.fetch_scalar(conn, "SELECT count(*) FROM ml.outreach_assignments") == population
    assert (
        database.fetch_scalar(
            conn, "SELECT count(*) FROM ml.outreach_assignments WHERE grp = 'holdout'"
        )
        >= 1
    )

    # the load is recorded, so a model trained from this data can be traced to the exact file
    log = conn.execute("SELECT source_file, file_sha256, row_count FROM raw.load_log").fetchall()
    assert log == [
        (sample_workbook.name, lineage.sha256_file(sample_workbook), len(first) + len(second))
    ]

    # ...and so is every registered model, through the tags of the run that produced it
    client = MlflowClient(mlflow_uri)
    # (the first version, whether or not the gate promoted it: on this tiny synthetic workbook
    # the stricter gate may reject it, and the lineage is recorded either way)
    version = client.get_model_version("retail-churn-classifier", "1")
    tags = client.get_run(version.run_id).data.tags
    assert tags["raw_file_sha256"] == lineage.sha256_file(sample_workbook)
    assert tags["transform_sql_sha256"] == lineage.sha256_sql_files(config.TRANSFORM_DIR)
    assert tags["raw_rows"] == str(len(first) + len(second))

    registered = {m.name for m in MlflowClient(mlflow_uri).search_registered_models()}
    assert registered == {
        "retail-churn-classifier",
        "retail-value-regressor",
        "retail-rfm-segments",
    }


@pytest.mark.e2e
def test_running_the_pipeline_twice_gives_the_same_tables(
    pipeline_env, sample_workbook, monkeypatch
):
    conn, _ = pipeline_env
    monkeypatch.setattr(config, "DATA_PATH", sample_workbook)
    counts_query = (
        "SELECT (SELECT count(*) FROM raw.online_retail), "
        "(SELECT count(*) FROM staging.transactions), "
        "(SELECT count(*) FROM marts.customer_snapshots)"
    )

    assert main.main(["--stage", "load"]) == 0
    assert main.main(["--stage", "transform"]) == 0
    first = conn.execute(counts_query).fetchone()
    conn.rollback()
    assert main.main(["--stage", "load"]) == 0
    assert main.main(["--stage", "transform"]) == 0

    assert conn.execute(counts_query).fetchone() == first


# --------------------------------------------------------- failure paths


def test_a_missing_source_file_is_a_clean_failure(pipeline_env, caplog):
    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "load"])

    assert code == 1
    assert "Source file not found" in caplog.text
    assert "Traceback" not in caplog.text


def test_a_failed_load_leaves_the_existing_raw_data_untouched(
    pipeline_env, sample_workbook, tmp_path, monkeypatch
):
    conn, _ = pipeline_env
    monkeypatch.setattr(config, "DATA_PATH", sample_workbook)
    assert main.main(["--stage", "load"]) == 0
    loaded = database.fetch_scalar(conn, "SELECT count(*) FROM raw.online_retail")
    conn.rollback()

    broken = tmp_path / "broken.xlsx"
    pd.DataFrame({"Invoice": ["1"]}).to_excel(broken, index=False)
    monkeypatch.setattr(config, "DATA_PATH", broken)

    assert main.main(["--stage", "load"]) == 1
    assert database.fetch_scalar(conn, "SELECT count(*) FROM raw.online_retail") == loaded


def test_transform_before_load_tells_you_to_load(pipeline_env, caplog):
    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "transform"])

    assert code == 1
    assert "raw.online_retail is empty" in caplog.text
    assert "--stage load" in caplog.text


def test_train_before_transform_tells_you_to_transform(pipeline_env, caplog):
    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "train"])

    assert code == 1
    assert "marts.customer_snapshots does not exist" in caplog.text
    assert "--stage transform" in caplog.text


def test_score_before_transform_tells_you_to_transform(pipeline_env, caplog):
    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "score"])

    assert code == 1
    assert "--stage transform" in caplog.text


def test_plan_before_score_tells_you_to_score(pipeline_env, caplog):
    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "plan"])

    assert code == 1
    assert "--stage score" in caplog.text


@pytest.mark.e2e
def test_score_without_a_champion_tells_you_to_train(
    pipeline_env, sample_workbook, monkeypatch, caplog
):
    monkeypatch.setattr(config, "DATA_PATH", sample_workbook)
    assert main.main(["--stage", "load"]) == 0
    assert main.main(["--stage", "transform"]) == 0

    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "score"])  # nothing has been trained

    assert code == 1
    assert "no champion" in caplog.text and "--stage train" in caplog.text
    assert "Traceback" not in caplog.text


def test_an_unreachable_database_is_a_clean_failure(monkeypatch, caplog):
    monkeypatch.setattr(config, "POSTGRES_USER", "someone")
    monkeypatch.setattr(config, "POSTGRES_PASSWORD", "s3cret-pw-do-not-print")
    monkeypatch.setattr(config, "POSTGRES_DB", "anything")
    monkeypatch.setattr(config, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "POSTGRES_PORT", 1)
    monkeypatch.setattr(database.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO):
        # transform needs the database straight away (load now parses the Excel file first)
        code = main.main(["--stage", "transform"])

    assert code == 1
    assert "Cannot connect to PostgreSQL" in caplog.text
    assert "s3cret-pw-do-not-print" not in caplog.text
    assert "Traceback" not in caplog.text


def test_an_unexpected_error_keeps_its_traceback_and_exits_with_2(monkeypatch, caplog):
    def boom():
        raise RuntimeError("a bug, not a fixable situation")

    monkeypatch.setattr(main, "load_raw", boom)

    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "load"])

    assert code == 2
    assert "Traceback" in caplog.text
    assert "a bug, not a fixable situation" in caplog.text


def test_an_unknown_stage_is_rejected_by_the_argument_parser():
    with pytest.raises(SystemExit) as caught:
        main.main(["--stage", "bogus"])

    assert caught.value.code == 2


def test_failing_quality_checks_leave_no_new_tables_behind(pipeline_env, insert_raw, caplog):
    conn, _ = pipeline_env
    # one 2011-01 purchase: the data ends long before the label windows, so a check fails
    insert_raw([(SHEET_1, "1", "85123A", "MUG", 1, "2011-01-04 09:00", 2.5, 1, "United Kingdom")])

    with caplog.at_level(logging.INFO):
        code = main.main(["--stage", "transform"])

    assert code == 1
    assert "Data quality checks failed" in caplog.text
    assert conn.execute("SELECT to_regclass('staging.transactions')").fetchone()[0] is None
