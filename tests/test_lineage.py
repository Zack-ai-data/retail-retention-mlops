"""Lineage: what a model's tags say about the code, data and libraries that made it."""

import hashlib
import importlib.metadata
import os
from datetime import datetime
from pathlib import Path

from src import config, lineage

# ------------------------------------------------------------------ checksums


def test_sha256_file_matches_the_standard_checksum(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")

    assert lineage.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
    assert lineage.sha256_file(path).startswith("ba7816bf")  # the well-known value for "abc"


def test_sha256_file_reads_large_files_in_chunks_without_changing_the_result(tmp_path):
    path = tmp_path / "big.bin"
    content = os.urandom(3 * 1024 * 1024 + 17)  # crosses the 1 MB chunk boundaries
    path.write_bytes(content)

    assert lineage.sha256_file(path) == hashlib.sha256(content).hexdigest()


def _sql_folder(tmp_path, files):
    folder = tmp_path / "sql"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text)
    return folder


def test_the_sql_checksum_changes_when_a_file_is_edited_added_removed_or_renamed(tmp_path):
    folder = _sql_folder(tmp_path, {"010_a.sql": "select 1;", "020_b.sql": "select 2;"})
    original = lineage.sha256_sql_files(folder)

    (folder / "020_b.sql").write_text("select 3;")
    edited = lineage.sha256_sql_files(folder)
    (folder / "030_c.sql").write_text("select 4;")
    added = lineage.sha256_sql_files(folder)
    (folder / "030_c.sql").unlink()
    (folder / "020_b.sql").write_text("select 2;")
    (folder / "020_b.sql").rename(folder / "025_b.sql")
    renamed = lineage.sha256_sql_files(folder)

    assert len({original, edited, added, renamed}) == 4


def test_the_sql_checksum_ignores_other_files_and_file_times(tmp_path):
    folder = _sql_folder(tmp_path, {"010_a.sql": "select 1;"})
    before = lineage.sha256_sql_files(folder)

    (folder / "notes.txt").write_text("not sql")
    os.utime(folder / "010_a.sql", (1, 1))  # a different modification time

    assert lineage.sha256_sql_files(folder) == before


def test_the_sql_checksum_does_not_confuse_where_one_file_ends_and_the_next_begins(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.sql").write_text("ab")
    (first / "b.sql").write_text("c")
    (second / "a.sql").write_text("a")
    (second / "b.sql").write_text("bc")  # same text overall, split at a different place

    assert lineage.sha256_sql_files(first) != lineage.sha256_sql_files(second)


# ------------------------------------------------------------------ run tags


def test_tags_say_unknown_before_anything_was_loaded(scratch_conn):
    tags = lineage.run_tags(scratch_conn)

    for key in ("raw_file_name", "raw_file_sha256", "raw_rows", "raw_loaded_at"):
        assert tags[key] == lineage.UNKNOWN


def test_tags_describe_the_latest_load_of_the_raw_table(scratch_conn):
    lineage.record_load(scratch_conn, Path("/data/online_retail_II.xlsx"), "a" * 64, 1000)
    lineage.record_load(scratch_conn, Path("/data/other.xlsx"), "b" * 64, 2500)
    scratch_conn.commit()

    tags = lineage.run_tags(scratch_conn)

    assert tags["raw_file_name"] == "other.xlsx"  # the file name, not the path
    assert tags["raw_file_sha256"] == "b" * 64
    assert tags["raw_rows"] == "2500"
    datetime.fromisoformat(tags["raw_loaded_at"])  # an ISO timestamp


def test_reading_the_tags_leaves_the_connection_idle(scratch_conn):
    lineage.run_tags(scratch_conn)

    assert scratch_conn.info.transaction_status.name == "IDLE"


def test_the_git_sha_comes_from_the_build(monkeypatch, scratch_conn):
    monkeypatch.setenv("GIT_SHA", "9f2c1d7")
    assert lineage.run_tags(scratch_conn)["git_sha"] == "9f2c1d7"

    monkeypatch.delenv("GIT_SHA")
    assert lineage.run_tags(scratch_conn)["git_sha"] == "unknown"


def test_the_transform_sql_checksum_covers_the_real_sql(scratch_conn):
    assert lineage.run_tags(scratch_conn)["transform_sql_sha256"] == lineage.sha256_sql_files(
        config.TRANSFORM_DIR
    )


def test_the_lock_file_checksum_is_recorded_when_there_is_one(monkeypatch, tmp_path, scratch_conn):
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==2.0.0\n")
    monkeypatch.setattr(lineage, "LOCK_FILE", lock)
    assert lineage.run_tags(scratch_conn)["lockfile_sha256"] == lineage.sha256_file(lock)

    monkeypatch.setattr(lineage, "LOCK_FILE", tmp_path / "missing.lock")
    assert lineage.run_tags(scratch_conn)["lockfile_sha256"] == "unavailable"


def test_library_versions_are_recorded(monkeypatch, scratch_conn):
    tags = lineage.run_tags(scratch_conn)

    assert tags["lib.scikit-learn"] == importlib.metadata.version("scikit-learn")
    assert tags["lib.numpy"] == importlib.metadata.version("numpy")

    monkeypatch.setattr(lineage, "TRACKED_PACKAGES", ("not-a-real-package",))
    assert lineage.run_tags(scratch_conn)["lib.not-a-real-package"] == "unknown"
    assert "lib.numpy" not in lineage.run_tags(scratch_conn)


def test_every_tag_is_a_short_string(scratch_conn):
    for key, value in lineage.run_tags(scratch_conn).items():
        assert isinstance(value, str) and 0 < len(value) < 500, key
