"""The monitor stage: the aggregate tables the dashboards read.

The warehouse tables are checked against an independent calculation in pandas from the same data;
the MLflow tables against a real (throwaway SQLite) model registry with real saved artifacts.
"""

import json
import logging
import re
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src import config, database, features, main, monitoring, processor, scoring, tracking
from src.errors import EmptyTableError, RegistryError  # noqa: F401
from tests.test_drift import _ChurnByRecency
from tests.test_model import make_panel
from tests.test_scoring import RAW, stub_champions

CHURN, VALUE, SEGMENTS = config.CHURN_MODEL_NAME, config.VALUE_MODEL_NAME, config.SEGMENT_MODEL_NAME

# ----------------------------------------------------------------------- the warehouse


@pytest.fixture
def warehouse(scratch_conn, insert_raw, monkeypatch):
    """A database with snapshots, one scoring run and one segment row, and the monitoring tables."""
    insert_raw(RAW)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    monkeypatch.setattr(scoring, "load_champions", lambda client: stub_champions())
    scoring.score_batch(scratch_conn, client=None)
    scratch_conn.execute(
        "INSERT INTO marts.customer_segments VALUES "
        "('2011-12-10', 1, 0, 'High value / Active', '3'), "
        "('2011-12-10', 2, 1, 'Mid value / Active', '3')"
    )
    scratch_conn.commit()
    database.run_sql_dir(scratch_conn, config.SCHEMA_DIR)
    monkeypatch.setattr(config, "POSTGRES_DB", scratch_conn.info.dbname)
    return scratch_conn


def _table(conn, name, order="1"):
    return database.read_dataframe(conn, f"SELECT * FROM {name} ORDER BY {order}")


def test_the_snapshot_summary_matches_an_independent_calculation(warehouse, mlflow_client):
    monitoring.refresh(warehouse, mlflow_client)

    summary = _table(warehouse, "ml.monitor_snapshot_summary", "snapshot_date")
    snapshots = features.load_snapshots(warehouse)
    for _, row in summary.iterrows():
        mine = snapshots[snapshots["snapshot_date"] == row["snapshot_date"]]
        actionable = mine[mine["is_actionable"]]
        assert row["split"] == mine["split"].iloc[0]
        assert row["customers"] == len(mine)
        assert row["actionable_customers"] == len(actionable)
        if actionable.empty or actionable[features.CHURN_TARGET].isna().all():
            assert pd.isna(row["churn_rate"]) and pd.isna(row["avg_future_spend"])
        else:
            assert row["churn_rate"] == pytest.approx(actionable[features.CHURN_TARGET].mean())
            assert row["avg_future_spend"] == pytest.approx(
                actionable[features.VALUE_TARGET].mean()
            )
    assert len(summary) == snapshots["snapshot_date"].nunique()


def test_the_score_snapshot_has_no_churn_rate_because_it_has_no_labels(warehouse, mlflow_client):
    monitoring.refresh(warehouse, mlflow_client)

    score = _table(warehouse, "ml.monitor_snapshot_summary").query("split == 'score'")

    assert len(score) == 1
    assert pd.isna(score["churn_rate"].iloc[0]) and pd.isna(score["avg_future_spend"].iloc[0])
    assert score["actionable_customers"].iloc[0] > 0


def test_every_scoring_run_has_all_ten_buckets_and_nothing_is_lost(warehouse, mlflow_client):
    monitoring.refresh(warehouse, mlflow_client)

    buckets = _table(warehouse, "ml.monitor_score_distribution", "scoring_run_id, bucket")
    run = database.read_dataframe(warehouse, "SELECT * FROM ml.scoring_runs")
    stored = database.read_dataframe(warehouse, "SELECT * FROM ml.predictions")

    assert list(buckets["bucket"]) == list(range(10))
    assert list(buckets["bucket_label"][:2]) == ["0.0-0.1", "0.1-0.2"]
    assert buckets["bucket_label"].iloc[9] == "0.9-1.0"
    assert buckets["customers"].sum() == run["customers"].iloc[0] == len(stored)
    assert buckets["margin_at_risk"].sum() == pytest.approx(stored["margin_at_risk"].sum())


@pytest.mark.parametrize(
    ("p_churn", "bucket"),
    [(0.0, 0), (0.05, 0), (0.1, 1), (0.5, 5), (0.99, 9), (1.0, 9)],  # exactly 1.0 is the top bucket
)
def test_a_probability_lands_in_the_bucket_that_contains_it(
    warehouse, mlflow_client, p_churn, bucket
):
    warehouse.execute("UPDATE ml.predictions SET p_churn = %s", (p_churn,))
    warehouse.commit()

    monitoring.refresh(warehouse, mlflow_client)

    counts = _table(warehouse, "ml.monitor_score_distribution", "bucket").set_index("bucket")[
        "customers"
    ]
    assert counts[bucket] == counts.sum() > 0


def test_a_scoring_run_without_predictions_still_shows_ten_empty_buckets(warehouse, mlflow_client):
    warehouse.execute(
        "INSERT INTO ml.scoring_runs (snapshot_date, churn_model_version, value_model_version, "
        "margin, contact_cost, customers) VALUES ('2011-12-10', '1', '1', 0.3, 5, 0)"
    )
    warehouse.commit()

    monitoring.refresh(warehouse, mlflow_client)

    empty = database.read_dataframe(
        warehouse,
        "SELECT count(*), sum(customers) FROM ml.monitor_score_distribution "
        "WHERE scoring_run_id = (SELECT max(scoring_run_id) FROM ml.scoring_runs)",
    )
    assert (empty.iloc[0, 0], empty.iloc[0, 1]) == (10, 0)


def test_segment_sizes_count_customers_per_segment(warehouse, mlflow_client):
    monitoring.refresh(warehouse, mlflow_client)

    sizes = _table(warehouse, "ml.monitor_segment_sizes", "segment_name")

    assert list(sizes["segment_name"]) == ["High value / Active", "Mid value / Active"]
    assert list(sizes["customers"]) == [1, 1]


def test_no_monitoring_table_has_a_customer_level_column(warehouse):
    columns = database.read_dataframe(
        warehouse,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'ml' AND table_name LIKE 'monitor\\_%'",
    )

    assert set(columns["table_name"]) == {t.split(".")[1] for t in monitoring.TABLES}
    assert not columns["column_name"].str.contains("customer_id").any()


def test_refreshing_twice_replaces_the_tables_and_does_not_add_to_them(warehouse, mlflow_client):
    first = monitoring.refresh(warehouse, mlflow_client)
    second = monitoring.refresh(warehouse, mlflow_client)

    assert first == second


def test_a_refresh_that_fails_part_way_leaves_every_table_as_it_was(
    warehouse, mlflow_client, monkeypatch
):
    monitoring.refresh(warehouse, mlflow_client)
    before = {t: _table(warehouse, t, "1, 2") for t in monitoring.TABLES[:3]}
    warehouse.execute(
        "DELETE FROM marts.customer_segments"
    )  # a change the failed refresh must not publish
    warehouse.commit()
    real_copy = database.copy_dataframe

    def fail_on_the_last_table(conn, df, table):
        if table == "ml.monitor_backtest":
            raise RuntimeError("disk full")
        return real_copy(conn, df, table)

    monkeypatch.setattr(database, "copy_dataframe", fail_on_the_last_table)

    with pytest.raises(RuntimeError, match="disk full"):
        monitoring.refresh(warehouse, mlflow_client)

    warehouse.rollback()
    for table, expected in before.items():
        pd.testing.assert_frame_equal(_table(warehouse, table, "1, 2"), expected)
    assert len(_table(warehouse, "ml.monitor_segment_sizes")) == 2  # not the emptied state


# --------------------------------------------------------------------------------- MLflow


def _train_like(client, name, decision, metrics=None, backtest=None, champion=False):
    """Register a new version the way training does: a run with metrics (and a backtest table),
    a model version, and the gate's decision recorded on it."""
    with mlflow.start_run() as run:
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, value)
        if backtest is not None:
            mlflow.log_table(backtest, monitoring.BACKTEST_ARTIFACT)
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name)
    version = client.create_model_version(name, source="file:///not-used", run_id=run.info.run_id)
    if decision is not None:
        client.set_model_version_tag(name, version.version, "promotion", decision)
    if champion:
        client.set_registered_model_alias(name, tracking.CHAMPION_ALIAS, version.version)
    return int(version.version)


def _backtest(origins=("2010-12-13", "2011-03-13")):
    return pd.DataFrame(
        {
            "origin": list(origins),
            "train_snapshots": [3, 4][: len(origins)],
            "customers": [3468, 3340][: len(origins)],
            "base_rate": [0.65, 0.57][: len(origins)],
            "model_pr_auc": [0.85, 0.80][: len(origins)],
            "recency_pr_auc": [0.78, 0.71][: len(origins)],
            "gain": [0.068, 0.09][: len(origins)],
            "gain_ci_low": [0.051, 0.07][: len(origins)],
            "gain_ci_high": [0.085, 0.11][: len(origins)],
            "significant": [True, True][: len(origins)],
            "model_recall_top20": [0.27, 0.3][: len(origins)],
        }
    )


def test_every_version_is_listed_with_the_gates_decision_and_which_is_champion(mlflow_client):
    _train_like(mlflow_client, CHURN, "rejected: does not beat the baseline")
    _train_like(mlflow_client, CHURN, "promoted: passed: beats the recency baseline", champion=True)
    _train_like(mlflow_client, CHURN, "rejected: gain is within noise")

    versions, _ = monitoring.model_tables(mlflow_client)

    assert list(versions["version"]) == [1, 2, 3] and set(versions["model_name"]) == {CHURN}
    assert list(versions["decision"]) == ["rejected", "promoted", "rejected"]
    assert list(versions["is_champion"]) == [False, True, False]
    assert versions["reason"].iloc[1] == "passed: beats the recency baseline"
    assert str(versions["created_at"].dt.tz) == "UTC"


def test_a_version_with_no_recorded_decision_is_shown_as_unrecorded_not_hidden(mlflow_client):
    _train_like(mlflow_client, CHURN, None)

    versions, _ = monitoring.model_tables(mlflow_client)

    assert versions["decision"].iloc[0] == "unrecorded"


def test_a_decision_in_an_unknown_format_is_shown_as_unrecorded(mlflow_client):
    _train_like(mlflow_client, CHURN, "maybe: unsure")

    versions, _ = monitoring.model_tables(mlflow_client)

    assert versions["decision"].iloc[0] == "unrecorded"
    assert "maybe: unsure" in versions["reason"].iloc[0]


def test_a_long_reason_is_cut_so_one_row_cannot_swamp_a_table(mlflow_client):
    _train_like(mlflow_client, CHURN, "promoted: " + "x" * 5000)

    versions, _ = monitoring.model_tables(mlflow_client)

    assert len(versions["reason"].iloc[0]) == monitoring.MAX_REASON_CHARS


def test_models_that_were_never_trained_are_skipped_without_error(mlflow_client):
    _train_like(mlflow_client, VALUE, "promoted: ok")  # churn and segments do not exist yet

    versions, metrics = monitoring.model_tables(mlflow_client)

    assert set(versions["model_name"]) == {VALUE}
    assert list(versions.columns) == monitoring.VERSION_COLUMNS
    assert list(metrics.columns) == monitoring.METRIC_COLUMNS


def test_only_the_projects_models_are_listed(mlflow_client):
    _train_like(mlflow_client, CHURN, "promoted: ok")
    _train_like(mlflow_client, "mlflow-demo.prompts.code-reviewer", None)  # MLflow's own demo data

    versions, _ = monitoring.model_tables(mlflow_client)

    assert set(versions["model_name"]) == {CHURN}


def test_metrics_are_one_row_each_per_version_with_numeric_versions(mlflow_client):
    _train_like(mlflow_client, CHURN, "promoted: ok", {"test_pr_auc": 0.65, "test_roc_auc": 0.77})
    _train_like(mlflow_client, CHURN, "promoted: ok", {"test_pr_auc": 0.66})

    _, metrics = monitoring.model_tables(mlflow_client)

    got = {(r.version, r.metric): r.value for r in metrics.itertuples()}
    assert got == {(1, "test_pr_auc"): 0.65, (1, "test_roc_auc"): 0.77, (2, "test_pr_auc"): 0.66}
    assert metrics["version"].dtype.kind == "i"


def test_the_backtest_is_read_from_each_versions_saved_table(mlflow_client):
    _train_like(mlflow_client, CHURN, "promoted: ok", backtest=_backtest())
    _train_like(mlflow_client, CHURN, "promoted: ok", backtest=_backtest(("2011-06-11",)))

    table = monitoring.backtest_table(mlflow_client)

    assert list(table.columns) == monitoring.BACKTEST_COLUMNS
    assert sorted(zip(table["version"], table["origin"], strict=True)) == [
        (1, "2010-12-13"),
        (1, "2011-03-13"),
        (2, "2011-06-11"),
    ]
    assert table["gain"].iloc[0] == pytest.approx(0.068)
    assert bool(table["significant"].iloc[0]) is True


def test_a_version_trained_before_the_backtest_existed_is_skipped_not_an_error(
    mlflow_client, caplog
):
    _train_like(mlflow_client, CHURN, "promoted: ok")  # no backtest saved
    _train_like(mlflow_client, CHURN, "promoted: ok", backtest=_backtest())

    with caplog.at_level(logging.INFO, logger="src.monitoring"):
        table = monitoring.backtest_table(mlflow_client)

    assert set(table["version"]) == {2}
    assert "Version 1 has no backtest.json" in caplog.text


def test_no_backtest_anywhere_gives_an_empty_table_with_the_right_columns(mlflow_client):
    assert list(monitoring.backtest_table(mlflow_client).columns) == monitoring.BACKTEST_COLUMNS
    assert monitoring.backtest_table(mlflow_client).empty


# --------------------------------------------------------------------- the whole stage


def test_refresh_fills_every_table_and_reports_how_many_rows(warehouse, mlflow_client):
    _train_like(mlflow_client, CHURN, "promoted: ok", {"test_pr_auc": 0.65}, _backtest(), True)
    _train_like(mlflow_client, VALUE, "rejected: no", {"test_spearman": 0.59})

    counts = monitoring.refresh(warehouse, mlflow_client)

    assert list(counts) == monitoring.TABLES
    assert counts["ml.monitor_model_versions"] == 2
    assert counts["ml.monitor_model_metrics"] == 2
    assert counts["ml.monitor_backtest"] == 2
    assert counts["ml.monitor_score_distribution"] == 10
    assert counts["ml.monitor_segment_sizes"] == 2
    assert counts["ml.monitor_snapshot_summary"] == len(
        database.read_dataframe(
            warehouse, "SELECT DISTINCT snapshot_date FROM marts.customer_snapshots"
        )
    )


def test_the_stage_runs_from_the_command_line(warehouse, mlflow_client, monkeypatch, caplog):
    monkeypatch.setattr(tracking, "configure", lambda *a, **k: mlflow_client)
    _train_like(mlflow_client, CHURN, "promoted: ok", backtest=_backtest(), champion=True)

    with caplog.at_level(logging.INFO, logger="pipeline"):
        code = main.main(["--stage", "monitor"])

    assert code == 0
    assert "ml.monitor_backtest now holds 2 rows" in caplog.text
    assert len(_table(warehouse, "ml.monitor_model_versions")) == 1


def test_the_stage_explains_what_to_run_first_when_there_are_no_snapshots(
    scratch_conn, mlflow_client, monkeypatch, caplog
):
    monkeypatch.setattr(config, "POSTGRES_DB", scratch_conn.info.dbname)
    monkeypatch.setattr(tracking, "configure", lambda *a, **k: mlflow_client)

    with caplog.at_level(logging.ERROR, logger="pipeline"):
        code = main.main(["--stage", "monitor"])

    assert code == 1
    assert "Run the transform stage first" in caplog.text


# ------------------------------------------------------------------------ the dashboards


def _warehouse_queries():
    """Every dashboard query that reads the warehouse, with where it came from."""
    folder = config.PROJECT_ROOT / "services" / "grafana" / "dashboards"
    for path in sorted(folder.glob("*.json")):
        for panel in json.loads(path.read_text(encoding="utf-8"))["panels"]:
            for target in panel["targets"]:
                sql = target.get("rawSql", "")
                if re.search(r"\bml\.", sql):
                    yield f"{path.stem}: {panel['title']}", sql, target["format"]


def test_every_dashboard_query_on_the_warehouse_runs_against_the_real_tables(
    warehouse, mlflow_client
):
    # a wrong column or table name in a panel would otherwise only show up on someone's screen
    _train_like(mlflow_client, CHURN, "promoted: ok", {"test_pr_auc": 0.65}, _backtest(), True)
    monitoring.refresh(warehouse, mlflow_client)
    queries = list(_warehouse_queries())
    failures = []

    for title, sql, fmt in queries:
        try:
            with warehouse.cursor() as cur:
                cur.execute(sql)
                columns = [column.name for column in cur.description]
            if fmt == "time_series" and columns[0] != "time":
                failures.append(f"{title}: a time series must start with a column named time")
        except Exception as exc:  # report every broken panel at once
            failures.append(f"{title}: {exc}")
        finally:
            warehouse.rollback()

    assert len(queries) >= 15, "the dashboards no longer query the warehouse"
    assert not failures, "\n".join(failures)


def test_a_metric_that_is_not_a_number_is_left_out_not_stored_as_a_value(mlflow_client):
    _train_like(
        mlflow_client, CHURN, "promoted: ok", {"test_pr_auc": 0.65, "test_ece": float("nan")}
    )

    _, metrics = monitoring.model_tables(mlflow_client)

    assert list(metrics["metric"]) == ["test_pr_auc"]


def test_a_version_with_no_recorded_decision_still_reaches_the_table_as_unrecorded(
    warehouse, mlflow_client
):
    # the stage exists to surface a bad registry state, so it must not crash on one
    _train_like(mlflow_client, CHURN, None)
    _train_like(mlflow_client, CHURN, "maybe: unsure")

    monitoring.refresh(warehouse, mlflow_client)

    stored = _table(warehouse, "ml.monitor_model_versions", "version")
    assert list(stored["decision"]) == ["unrecorded", "unrecorded"]


# ------------------------------------------ what counts as "nothing there" and what is an error


class _FailingClient:
    """A registry client whose calls fail the way MLflow does, with a given error code."""

    def __init__(self, code, fail_on):
        self.code, self.fail_on = code, fail_on
        self.version = SimpleNamespace(version="1", run_id="r1", tags={}, creation_timestamp=0)

    def _maybe(self, call):
        if call == self.fail_on:
            raise MlflowException("simulated", error_code=self.code)

    def get_registered_model(self, name):
        self._maybe("get_registered_model")
        return SimpleNamespace(aliases={})

    def search_model_versions(self, _filter):
        self._maybe("search_model_versions")
        return [self.version]

    def get_run(self, run_id):
        self._maybe("get_run")
        return SimpleNamespace(data=SimpleNamespace(metrics={"m": 1.0}))

    def download_artifacts(self, run_id, path, dst):
        self._maybe("download_artifacts")
        raise AssertionError("not reached when the call is meant to fail")


@pytest.mark.parametrize("call", ["get_registered_model", "get_run"])
def test_an_mlflow_failure_other_than_not_found_stops_the_stage_instead_of_emptying_the_tables(
    call,
):
    client = _FailingClient(INTERNAL_ERROR, call)

    with pytest.raises(MlflowException, match="simulated"):
        monitoring.model_tables(client)


def test_a_backtest_download_failure_other_than_not_found_stops_the_stage():
    client = _FailingClient(INTERNAL_ERROR, "download_artifacts")

    with pytest.raises(MlflowException, match="simulated"):
        monitoring.backtest_table(client)


def test_a_failed_registry_search_for_the_backtest_is_an_error_not_an_empty_table():
    client = _FailingClient(INTERNAL_ERROR, "search_model_versions")

    with pytest.raises(MlflowException, match="simulated"):
        monitoring.backtest_table(client)


def test_a_run_that_no_longer_exists_leaves_the_version_listed_with_no_metrics():
    client = _FailingClient(RESOURCE_DOES_NOT_EXIST, "get_run")

    versions, metrics = monitoring.model_tables(client)

    assert len(versions) == 3 and set(versions["version"]) == {1}  # one version of each model
    assert metrics.empty


def test_a_version_registered_without_a_run_is_listed_but_has_no_metrics_or_backtest(
    mlflow_client,
):
    mlflow_client.create_registered_model(CHURN)
    mlflow_client.create_model_version(CHURN, source="file:///not-used")  # no training run
    mlflow_client.set_model_version_tag(CHURN, "1", "promotion", "promoted: ok")

    versions, metrics = monitoring.model_tables(mlflow_client)

    assert list(versions["version"]) == [1] and metrics.empty
    assert monitoring.backtest_table(mlflow_client).empty


# ------------------------------------------------------------------------------------ drift


def _drift_inputs(monkeypatch, panel, champion_error=None):
    """Point the stage at a synthetic pair of December snapshots and a stand-in champion."""
    monkeypatch.setattr(features, "load_snapshots", lambda conn: panel)

    def load_champions(client):
        if champion_error is not None:
            raise champion_error
        return SimpleNamespace(churn_model=_ChurnByRecency())

    monkeypatch.setattr(scoring, "load_champions", load_champions)


def _december_pair(seed=1):
    return make_panel(
        n=600, seed=seed, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")]
    )


def test_refresh_writes_one_drift_row_per_column_for_a_snapshot_with_a_year_earlier(
    warehouse, mlflow_client, monkeypatch
):
    _drift_inputs(monkeypatch, _december_pair())

    counts = monitoring.refresh(warehouse, mlflow_client)

    stored = _table(warehouse, "ml.monitor_drift", "column_name")
    assert counts["ml.monitor_drift"] == len(monitoring.DRIFT_COLUMNS) + 1 == len(stored)
    assert set(stored["status"]) <= {"ok", "warn", "alert"}
    assert set(stored["snapshot_date"]) == {pd.Timestamp("2011-12-10").date()}
    assert set(stored["reference_date"]) == {pd.Timestamp("2010-12-13").date()}
    assert "customer_id" not in stored.columns


def test_a_refresh_replaces_the_drift_rows_and_does_not_add_to_them(
    warehouse, mlflow_client, monkeypatch
):
    _drift_inputs(monkeypatch, _december_pair())

    first = monitoring.refresh(warehouse, mlflow_client)
    second = monitoring.refresh(warehouse, mlflow_client)

    assert first == second


def test_with_no_champion_yet_the_inputs_are_still_checked_and_it_says_what_was_left_out(
    warehouse, mlflow_client, monkeypatch, caplog
):
    _drift_inputs(monkeypatch, _december_pair(), champion_error=RegistryError("no champion yet"))

    with caplog.at_level(logging.WARNING, logger="src.monitoring"):
        monitoring.refresh(warehouse, mlflow_client)

    stored = _table(warehouse, "ml.monitor_drift", "column_name")
    assert monitoring.CHURN_PROBABILITY not in set(stored["column_name"])
    assert len(stored) == len(monitoring.DRIFT_COLUMNS)
    assert "churn probability drift is left out" in caplog.text


def test_an_mlflow_failure_while_loading_the_champion_stops_the_stage_and_keeps_the_old_tables(
    warehouse, mlflow_client, monkeypatch
):
    _drift_inputs(monkeypatch, _december_pair())
    monitoring.refresh(warehouse, mlflow_client)
    before = _table(warehouse, "ml.monitor_drift", "column_name")
    _drift_inputs(
        monkeypatch, _december_pair(seed=9), champion_error=MlflowException("server error")
    )

    with pytest.raises(MlflowException, match="server error"):
        monitoring.refresh(warehouse, mlflow_client)

    pd.testing.assert_frame_equal(_table(warehouse, "ml.monitor_drift", "column_name"), before)


def test_a_deliberately_shifted_snapshot_shows_up_as_alerts_in_the_table(
    warehouse, mlflow_client, monkeypatch
):
    panel = _december_pair(seed=3)
    panel.loc[panel["snapshot_date"] == pd.Timestamp("2011-12-10").date(), "frequency_365d"] *= 3
    _drift_inputs(monkeypatch, panel)

    monitoring.refresh(warehouse, mlflow_client)

    stored = _table(warehouse, "ml.monitor_drift", "column_name").set_index("column_name")
    assert stored.loc["frequency_365d", "status"] == "alert"
    assert stored.loc["is_uk", "status"] == "ok"


def test_every_drift_status_can_be_stored_in_the_table(warehouse, mlflow_client, monkeypatch):
    # the real data has no warning rows in the latest comparison, so make one of each
    rows = [
        (pd.Timestamp("2011-12-10").date(), pd.Timestamp("2010-12-13").date(), name, psi, status)
        for name, psi, status in [("a", 0.01, "ok"), ("b", 0.12, "warn"), ("c", 0.6, "alert")]
    ]
    drift = pd.DataFrame(rows, columns=monitoring.DRIFT_TABLE_COLUMNS)
    monkeypatch.setattr(monitoring, "_drift", lambda conn, client: drift)

    monitoring.refresh(warehouse, mlflow_client)

    stored = _table(warehouse, "ml.monitor_drift", "column_name")
    assert list(stored["status"]) == ["ok", "warn", "alert"]


def test_a_status_outside_ok_warn_alert_is_refused_by_the_table(
    warehouse, mlflow_client, monkeypatch
):
    bad = pd.DataFrame(
        [(pd.Timestamp("2011-12-10").date(), pd.Timestamp("2010-12-13").date(), "a", 0.5, "fine")],
        columns=monitoring.DRIFT_TABLE_COLUMNS,
    )
    monkeypatch.setattr(monitoring, "_drift", lambda conn, client: bad)

    with pytest.raises(Exception, match="monitor_drift"):
        monitoring.refresh(warehouse, mlflow_client)
