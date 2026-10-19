"""What happens when things go wrong: each failure should say what happened and what to do."""

import pandas as pd
import pytest

from src import config, database, extractor, tracking
from src.errors import (
    ConfigurationError,
    DatabaseUnavailableError,
    DataFileError,
    EmptyTableError,
    PipelineError,
    TrackingServerError,
)

# ------------------------------------------------------------ source file


def test_missing_source_file_names_the_path_and_the_fix(tmp_path):
    with pytest.raises(DataFileError, match=r"not found.*data/ folder"):
        list(extractor.extract(tmp_path / "nope.xlsx"))


def test_a_file_that_is_not_excel_is_rejected(tmp_path):
    fake = tmp_path / "online_retail_II.xlsx"
    fake.write_text("this is not a workbook")

    with pytest.raises(DataFileError, match="not a readable Excel workbook"):
        list(extractor.extract(fake))


def test_a_sheet_without_the_expected_columns_is_rejected(tmp_path):
    path = tmp_path / "wrong_columns.xlsx"
    pd.DataFrame({"Invoice": ["1"], "Wrong": [2]}).to_excel(path, index=False)

    with pytest.raises(DataFileError, match="missing columns"):
        list(extractor.extract(path))


# ---------------------------------------------------------------- database


def test_missing_credentials_are_reported_before_connecting(monkeypatch):
    monkeypatch.setattr(config, "POSTGRES_PASSWORD", "")

    with pytest.raises(ConfigurationError, match="POSTGRES_PASSWORD"):
        database.get_connection(retries=1)


def test_unreachable_database_error_is_actionable_and_never_leaks_the_password(monkeypatch):
    monkeypatch.setattr(config, "POSTGRES_USER", "someone")
    monkeypatch.setattr(config, "POSTGRES_PASSWORD", "s3cret-pw-do-not-print")
    monkeypatch.setattr(config, "POSTGRES_DB", "anything")
    monkeypatch.setattr(config, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "POSTGRES_PORT", 1)  # nothing listens here
    monkeypatch.setattr(database.time, "sleep", lambda _seconds: None)

    with pytest.raises(DatabaseUnavailableError) as caught:
        database.get_connection(retries=3)

    message = str(caught.value)
    assert "127.0.0.1:1" in message
    assert "3 attempt(s)" in message
    assert "docker compose up -d postgres" in message
    assert "s3cret-pw-do-not-print" not in message


def test_fetch_scalar_returns_the_first_value(scratch_conn):
    assert database.fetch_scalar(scratch_conn, "SELECT 41 + 1") == 42


def test_fetch_scalar_fails_clearly_when_there_is_no_row(scratch_conn):
    with pytest.raises(PipelineError, match="returned no rows"):
        database.fetch_scalar(scratch_conn, "SELECT 1 WHERE false")


def test_require_rows_explains_a_missing_table(scratch_conn):
    with pytest.raises(EmptyTableError, match=r"marts\.nope does not exist.*do this first"):
        database.require_rows(scratch_conn, "marts.nope", "do this first")
    # the failed query must not leave the connection unusable
    assert database.fetch_scalar(scratch_conn, "SELECT 1") == 1


def test_require_rows_explains_an_empty_table(scratch_conn):
    with pytest.raises(EmptyTableError, match=r"raw\.online_retail is empty"):
        database.require_rows(scratch_conn, "raw.online_retail", "load it")


def test_require_rows_returns_the_count(scratch_conn, insert_raw):
    insert_raw([("s", "1", "85123A", "MUG", 1, "2010-01-04 09:00", 1.0, 1, "United Kingdom")])

    assert database.require_rows(scratch_conn, "raw.online_retail", "load it") == 1


def test_read_dataframe_returns_rows_and_column_names(scratch_conn):
    frame = database.read_dataframe(scratch_conn, "SELECT 1 AS a, 'x' AS b")

    assert list(frame.columns) == ["a", "b"]
    assert frame.iloc[0].tolist() == [1, "x"]


def test_read_dataframe_rejects_a_statement_without_a_result(scratch_conn):
    with pytest.raises(PipelineError, match="no result set"):
        database.read_dataframe(scratch_conn, "SET search_path = public")


# ------------------------------------------------------------------ MLflow


def test_unreachable_tracking_server_is_reported_with_the_fix(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")

    with pytest.raises(TrackingServerError, match=r"127\.0\.0\.1:9.*docker compose up -d mlflow"):
        tracking.configure("http://127.0.0.1:9")
