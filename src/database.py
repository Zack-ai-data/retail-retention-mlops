# Utilize PostgreSQL for database management
import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from src import config
from src.errors import ConfigurationError, DatabaseUnavailableError, EmptyTableError, PipelineError

log = logging.getLogger(__name__)


def get_connection(
    dbname: str | None = None,
    retries: int = 10,
    delay: float = 3.0,
    autocommit: bool = False,
) -> psycopg.Connection[Any]:
    """Connect to Postgres, retrying while the database container starts up.

    dbname defaults to the project warehouse (POSTGRES_DB). Raises ConfigurationError when
    credentials are missing and DatabaseUnavailableError when every attempt fails; the password
    is never included in a message.
    """
    dbname = dbname or config.POSTGRES_DB
    if not (config.POSTGRES_USER and config.POSTGRES_PASSWORD and dbname):
        raise ConfigurationError(
            "POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB must be set (see .env.example)"
        )

    attempts = max(1, retries)
    last_error: psycopg.OperationalError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return psycopg.connect(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
                dbname=dbname,
                autocommit=autocommit,
                connect_timeout=config.POSTGRES_CONNECT_TIMEOUT,
            )
        except psycopg.OperationalError as exc:
            last_error = exc
            if attempt < attempts:
                log.warning("Database not ready (%s/%s): %s", attempt, attempts, exc)
                time.sleep(delay)

    raise DatabaseUnavailableError(
        f"Cannot connect to PostgreSQL at {config.POSTGRES_HOST}:{config.POSTGRES_PORT} "
        f"(database {dbname!r}, user {config.POSTGRES_USER!r}) after {attempts} attempt(s): "
        f"{last_error}. Is the container running? Try: docker compose up -d postgres"
    ) from last_error


def run_sql_file(conn: psycopg.Connection[Any], path: Path, commit: bool = True) -> None:
    """Run one SQL script as a single transaction (all statements or none).

    With commit=False the caller owns the transaction (to publish several scripts together).
    """
    log.info("Running %s", path.name)
    conn.execute(path.read_text(encoding="utf-8"))
    if commit:
        conn.commit()


def run_sql_dir(conn: psycopg.Connection[Any], directory: Path) -> None:
    """Run every *.sql file in directory in name order. The files are idempotent."""
    for path in sorted(directory.glob("*.sql")):
        run_sql_file(conn, path)


def fetch_scalar(conn: psycopg.Connection[Any], query: str) -> Any:
    """Return the first column of the first row, or raise if the query returned no row."""
    row = conn.execute(query).fetchone()
    if row is None:
        raise PipelineError(f"Query returned no rows: {' '.join(query.split())[:100]}")
    return row[0]


def require_rows(conn: psycopg.Connection[Any], table: str, hint: str) -> int:
    """Return the row count of table; raise EmptyTableError (with hint) if missing or empty."""
    try:
        count = int(fetch_scalar(conn, f"SELECT count(*) FROM {table}"))
    except psycopg.errors.UndefinedTable as exc:
        conn.rollback()
        raise EmptyTableError(f"{table} does not exist. {hint}") from exc
    if count == 0:
        raise EmptyTableError(f"{table} is empty. {hint}")
    return count


def read_dataframe(
    conn: psycopg.Connection[Any], query: str, params: Sequence[Any] | None = None
) -> pd.DataFrame:
    """Run a read-only query and return the result as a DataFrame.

    Values that come from outside (an API request, a form) must go in params and be referenced
    as %s in the query, never formatted into the SQL text.
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        if cur.description is None:
            raise PipelineError("Query returned no result set")
        columns = [column.name for column in cur.description]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    conn.rollback()  # end the read-only transaction
    return frame


def copy_dataframe(conn: psycopg.Connection[Any], df: pd.DataFrame, table: str) -> int:
    """Bulk-load a DataFrame into table with COPY. Missing values become NULL.

    The DataFrame's columns must be the table's columns, in the same order.
    Does not commit; the caller controls the transaction.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    columns = ", ".join(df.columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '')") as copy:
            while chunk := buffer.read(1 << 20):
                copy.write(chunk)
        return cur.rowcount
