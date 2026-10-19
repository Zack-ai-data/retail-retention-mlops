"""Where a model came from: the code, the data and the libraries that produced it.

Every training run is tagged with these facts so that a model in the registry can be traced
back, and a result that changes can be explained by what changed:

  git_sha               the commit the image was built from (passed in at build time, because the
                        image has no .git folder)
  raw_file_sha256       the checksum of the Excel file that was loaded into the raw table, recorded
                        when it was loaded (raw.load_log), not looked up now
  transform_sql_sha256  a checksum of the SQL that builds staging and the feature table
  lockfile_sha256       a checksum of the pinned dependency list the image was built from
  lib.<package>         the versions of the libraries that matter for a pickled model
"""

import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path
from typing import Any

import psycopg

from src import config

LOCK_FILE = config.PROJECT_ROOT / "requirements.lock"

# A pickled model can only be trusted with the library versions that produced it
TRACKED_PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "mlflow-skinny")

UNKNOWN = "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sql_files(directory: Path) -> str:
    """One checksum for every .sql file in a folder: it changes if a file is edited, added,
    removed or renamed, and does not depend on file times."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.sql")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def record_load(conn: psycopg.Connection[Any], source: Path, file_sha256: str, rows: int) -> None:
    """Remember which file was loaded into the raw table, and how many rows it gave."""
    conn.execute(
        "INSERT INTO raw.load_log (source_file, file_sha256, row_count) VALUES (%s, %s, %s)",
        (source.name, file_sha256, rows),
    )


def _latest_load(conn: psycopg.Connection[Any]) -> dict[str, str]:
    row = conn.execute(
        "SELECT source_file, file_sha256, row_count, loaded_at "
        "FROM raw.load_log ORDER BY load_id DESC LIMIT 1"
    ).fetchone()
    conn.rollback()  # a read: do not leave the connection in a transaction
    if row is None:
        return {
            "raw_file_name": UNKNOWN,
            "raw_file_sha256": UNKNOWN,
            "raw_rows": UNKNOWN,
            "raw_loaded_at": UNKNOWN,
        }
    return {
        "raw_file_name": row[0],
        "raw_file_sha256": row[1],
        "raw_rows": str(row[2]),
        "raw_loaded_at": row[3].isoformat(),
    }


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return UNKNOWN


def run_tags(conn: psycopg.Connection[Any]) -> dict[str, str]:
    """The lineage tags for a training run started now."""
    tags = {
        "git_sha": os.getenv("GIT_SHA", UNKNOWN),
        **_latest_load(conn),
        "transform_sql_sha256": sha256_sql_files(config.TRANSFORM_DIR),
        "lockfile_sha256": sha256_file(LOCK_FILE) if LOCK_FILE.exists() else "unavailable",
        "python_version": platform.python_version(),
    }
    tags.update({f"lib.{package}": _version(package) for package in TRACKED_PACKAGES})
    return tags
