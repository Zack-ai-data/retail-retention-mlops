"""Realised performance: how the served churn probabilities did against what customers then did.

The measures are checked on generated cohorts whose true behaviour is known (calibrated, too
high, too low, no better than the recency rule). The join that finds the outcomes is checked on the
scratch warehouse. On the real data the table is empty: the live scoring runs are on a snapshot
whose 90 days never end, because the dataset does.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score

from src import config, database, evaluation, measures, monitoring
from tests.test_monitoring import _table, warehouse  # noqa: F401  (the scratch warehouse fixture)

SNAPSHOT_WITH_OUTCOMES = "2011-09-09"


def _cohort(shift=0.0, n=1500, seed=0):
    """Customers with a known chance of churning. The probabilities served are the true ones plus
    `shift`; recency is a weaker clue than the truth, as it is in the real data."""
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2, 3, n)  # about 40% churn
    churned = rng.binomial(1, p_true)
    recency = 20 + 200 * p_true + rng.normal(0, 70, n)
    return churned, np.clip(p_true + shift, 0.001, 0.999), recency


@pytest.fixture(autouse=True)
def quick_bootstrap(monkeypatch):
    monkeypatch.setattr(evaluation, "N_BOOTSTRAP", 200)  # the intervals here are not close calls


# ---------------------------------------------------------------------------- the measures


def test_probabilities_that_are_right_and_better_than_recency_are_ok():
    churned, served, recency = _cohort()

    row = monitoring.realised_row(churned, served, recency)

    assert row["status"] == "ok" and row["significant"] is True
    assert abs(row["calibration_gap"]) < monitoring.REALISED_WARN
    assert row["gain_ci_low"] > 0 and row["customers"] == len(churned)


def test_the_gap_is_the_average_probability_served_minus_the_share_that_churned():
    churned, served, recency = _cohort(shift=0.07)

    row = monitoring.realised_row(churned, served, recency)

    assert row["calibration_gap"] == pytest.approx(served.mean() - churned.mean())
    assert row["actual_churn_rate"] == pytest.approx(churned.mean())
    assert row["mean_p_churn"] == pytest.approx(served.mean())


def test_probabilities_that_are_too_high_by_a_few_points_are_a_warning():
    churned, served, recency = _cohort(shift=0.07)

    row = monitoring.realised_row(churned, served, recency)

    assert monitoring.REALISED_WARN <= row["calibration_gap"] < monitoring.REALISED_ALERT
    assert row["status"] == "warn" and row["significant"] is True  # the ranking itself is intact


@pytest.mark.parametrize("shift", [0.14, -0.14])
def test_probabilities_far_too_high_or_far_too_low_are_an_alert(shift):
    churned, served, recency = _cohort(shift=shift)

    row = monitoring.realised_row(churned, served, recency)

    assert abs(row["calibration_gap"]) >= monitoring.REALISED_ALERT
    assert row["status"] == "alert"
    assert (row["calibration_gap"] > 0) == (shift > 0)  # positive means it over-predicts


def test_a_model_that_is_calibrated_but_no_better_than_recency_is_an_alert():
    churned, _, recency = _cohort()
    rng = np.random.default_rng(1)
    served = np.clip(rng.normal(churned.mean(), 0.02, len(churned)), 0.001, 0.999)  # no signal

    row = monitoring.realised_row(churned, served, recency)

    assert abs(row["calibration_gap"]) < monitoring.REALISED_WARN  # the average is right
    assert row["significant"] is False and row["gain"] < 0
    assert row["status"] == "alert"


def test_the_numbers_match_an_independent_calculation():
    churned, served, recency = _cohort(shift=0.03, seed=3)

    row = monitoring.realised_row(churned, served, recency)

    assert row["model_pr_auc"] == pytest.approx(average_precision_score(churned, served))
    assert row["recency_pr_auc"] == pytest.approx(average_precision_score(churned, recency))
    assert row["gain"] == pytest.approx(row["model_pr_auc"] - row["recency_pr_auc"])
    assert row["ece"] == pytest.approx(measures.expected_calibration_error(churned, served))
    assert row["gain_ci_low"] <= row["gain"] <= row["gain_ci_high"]


def test_too_few_customers_or_customers_who_all_did_the_same_cannot_be_judged():
    churned, served, recency = _cohort()
    few = monitoring.REALISED_MIN_CUSTOMERS - 1

    assert monitoring.realised_row(churned[:few], served[:few], recency[:few]) is None
    assert monitoring.realised_row(np.ones(500, dtype=int), served[:500], recency[:500]) is None
    assert monitoring.realised_row(np.zeros(500, dtype=int), served[:500], recency[:500]) is None


@pytest.mark.parametrize(
    ("gap", "significant", "status"),
    [
        (0.0, True, "ok"),
        (0.0499, True, "ok"),
        (0.05, True, "warn"),
        (-0.05, True, "warn"),
        (0.0999, True, "warn"),
        (0.10, True, "alert"),
        (-0.10, True, "alert"),
        (0.0, False, "alert"),  # no gap, but not clearly better than the simple rule
    ],
)
def test_the_status_changes_exactly_at_the_thresholds(gap, significant, status):
    assert monitoring.realised_status(gap, significant) == status


def test_the_thresholds_are_the_ones_agreed():
    assert (monitoring.REALISED_WARN, monitoring.REALISED_ALERT) == (0.05, 0.10)


# ------------------------------------------------------------------- finding the outcomes


def _score_the_labelled_snapshot(conn, p_churn=0.9):
    """A scoring run on the snapshot whose outcomes are known, as if it had been scored then."""
    customer = conn.execute(
        "SELECT customer_id FROM marts.customer_snapshots WHERE snapshot_date = %s "
        "ORDER BY customer_id",
        (SNAPSHOT_WITH_OUTCOMES,),
    ).fetchone()[0]
    run = conn.execute(
        "INSERT INTO ml.scoring_runs (snapshot_date, churn_model_version, value_model_version, "
        "margin, contact_cost, customers) VALUES (%s, '5', '5', 0.3, 5.0, 1) "
        "RETURNING scoring_run_id",
        (SNAPSHOT_WITH_OUTCOMES,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO ml.predictions VALUES (%s, %s, %s, 10, 5, 3, -2)", (run, customer, p_churn)
    )
    conn.commit()
    return run, customer


@pytest.mark.usefixtures("warehouse")
def test_a_scoring_run_is_joined_to_the_outcome_of_each_customer_it_scored(scratch_conn):
    run, customer = _score_the_labelled_snapshot(scratch_conn, p_churn=0.9)

    joined = database.read_dataframe(scratch_conn, monitoring._REALISED_QUERY)

    assert list(joined["scoring_run_id"]) == [run]  # the run on the unlabelled snapshot is absent
    expected = database.read_dataframe(
        scratch_conn,
        "SELECT label_churned::int AS churned, recency_days::float8 AS recency_days "
        "FROM marts.customer_snapshots WHERE snapshot_date = %s AND customer_id = %s",
        (SNAPSHOT_WITH_OUTCOMES, customer),
    )
    assert joined.loc[0, "p_churn"] == pytest.approx(0.9)
    assert joined.loc[0, "churned"] == expected.loc[0, "churned"]
    assert joined.loc[0, "recency_days"] == expected.loc[0, "recency_days"]
    assert str(joined.loc[0, "churn_model_version"]) == "5"


@pytest.mark.usefixtures("warehouse")
def test_a_run_on_a_snapshot_with_no_outcomes_has_nothing_to_judge(scratch_conn):
    # the fixture's own scoring run is on the newest snapshot, which has no labels
    joined = database.read_dataframe(scratch_conn, monitoring._REALISED_QUERY)

    assert joined.empty
    assert monitoring.realised_table(scratch_conn).empty
    assert list(monitoring.realised_table(scratch_conn).columns) == monitoring.REALISED_COLUMNS


@pytest.mark.usefixtures("warehouse")
def test_a_run_with_too_few_customers_to_judge_gets_no_row(scratch_conn):
    _score_the_labelled_snapshot(scratch_conn)  # one customer

    assert monitoring.realised_table(scratch_conn).empty


def _crafted_join(runs):
    """What the query returns, for runs of generated customers: {run_id: (version, cohort)}."""
    parts = []
    for run_id, (version, (churned, served, recency)) in runs.items():
        parts.append(
            pd.DataFrame(
                {
                    "scoring_run_id": run_id,
                    "snapshot_date": pd.Timestamp("2011-09-09").date(),
                    "churn_model_version": version,
                    "p_churn": served,
                    "churned": churned,
                    "recency_days": recency,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


@pytest.mark.usefixtures("warehouse")
def test_each_scoring_run_gets_its_own_row_with_its_own_model_version(
    scratch_conn,
    monkeypatch,
):
    crafted = _crafted_join({4: ("5", _cohort()), 7: ("6", _cohort(shift=0.14, seed=2))})
    monkeypatch.setattr(database, "read_dataframe", lambda conn, query, params=None: crafted)

    table = monitoring.realised_table(scratch_conn)

    assert list(table["scoring_run_id"]) == [4, 7]
    assert list(table["churn_model_version"]) == ["5", "6"]
    assert list(table["status"]) == ["ok", "alert"]
    assert list(table.columns) == monitoring.REALISED_COLUMNS


# ----------------------------------------------------------------- stored by the stage


def _row_frame(status="warn"):
    return pd.DataFrame(
        [
            {
                "scoring_run_id": 4,
                "snapshot_date": pd.Timestamp("2011-09-09").date(),
                "churn_model_version": "5",
                "customers": 2775,
                "actual_churn_rate": 0.387,
                "mean_p_churn": 0.474,
                "calibration_gap": 0.087,
                "ece": 0.09,
                "model_pr_auc": 0.652,
                "recency_pr_auc": 0.535,
                "gain": 0.117,
                "gain_ci_low": 0.089,
                "gain_ci_high": 0.144,
                "significant": True,
                "status": status,
            }
        ],
        columns=monitoring.REALISED_COLUMNS,
    )


@pytest.mark.usefixtures("warehouse")
def test_refresh_stores_the_realised_rows_and_replaces_them_on_the_next_refresh(
    scratch_conn,
    mlflow_client,
    monkeypatch,
):
    monkeypatch.setattr(monitoring, "realised_table", lambda conn: _row_frame("warn"))

    counts = monitoring.refresh(scratch_conn, mlflow_client)
    counts_again = monitoring.refresh(scratch_conn, mlflow_client)

    stored = _table(scratch_conn, "ml.monitor_realised", "scoring_run_id")
    assert counts["ml.monitor_realised"] == counts_again["ml.monitor_realised"] == 1
    assert stored.loc[0, "status"] == "warn" and stored.loc[0, "calibration_gap"] == 0.087


@pytest.mark.usefixtures("warehouse")
def test_on_a_warehouse_with_no_outcomes_the_table_is_simply_empty(
    scratch_conn,
    mlflow_client,
):
    counts = monitoring.refresh(scratch_conn, mlflow_client)

    assert counts["ml.monitor_realised"] == 0


@pytest.mark.parametrize("status", ["ok", "warn", "alert"])
@pytest.mark.usefixtures("warehouse")
def test_every_status_can_be_stored(scratch_conn, mlflow_client, monkeypatch, status):
    monkeypatch.setattr(monitoring, "realised_table", lambda conn: _row_frame(status))

    monitoring.refresh(scratch_conn, mlflow_client)

    assert _table(scratch_conn, "ml.monitor_realised").loc[0, "status"] == status


@pytest.mark.usefixtures("warehouse")
def test_a_status_outside_ok_warn_alert_is_refused_by_the_table(
    scratch_conn,
    mlflow_client,
    monkeypatch,
):
    monkeypatch.setattr(monitoring, "realised_table", lambda conn: _row_frame("fine"))

    with pytest.raises(Exception, match="monitor_realised"):
        monitoring.refresh(scratch_conn, mlflow_client)


# ------------------------------------------------ the panels and the alert rule that read it


def _panel_sql(title):
    path = config.PROJECT_ROOT / "services" / "grafana" / "dashboards" / "retail_models.json"
    (panel,) = [
        p for p in json.loads(path.read_text(encoding="utf-8"))["panels"] if p["title"] == title
    ]
    (target,) = panel["targets"]
    return target["rawSql"]


def _rule_sql():
    path = config.PROJECT_ROOT / "services" / "grafana" / "provisioning" / "alerting" / "rules.yml"
    (group,) = yaml.safe_load(path.read_text(encoding="utf-8"))["groups"]
    (rule,) = [r for r in group["rules"] if r["uid"] == "retail-realised-performance"]
    return rule["data"][0]["model"]["rawSql"]


def _store(conn, mlflow_client, monkeypatch, *frames):
    both = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(monitoring, "realised_table", lambda c: both)
    monitoring.refresh(conn, mlflow_client)


def _scalar(conn, sql):
    row = conn.execute(sql).fetchone()
    conn.rollback()
    return row


@pytest.mark.usefixtures("warehouse")
def test_the_panels_show_the_latest_run_with_known_outcomes(
    scratch_conn, mlflow_client, monkeypatch
):
    older = _row_frame("ok").assign(
        scoring_run_id=3, snapshot_date=pd.Timestamp("2011-06-11").date()
    )
    older.loc[0, ["calibration_gap", "gain_ci_low"]] = [0.01, 0.2]
    _store(scratch_conn, mlflow_client, monkeypatch, older, _row_frame("warn"))

    gap = _scalar(scratch_conn, _panel_sql("Calibration gap, latest run with outcomes"))
    gain_low = _scalar(scratch_conn, _panel_sql("Gain over the recency rule, low end"))
    listed = database.read_dataframe(
        scratch_conn, _panel_sql("Scoring runs whose outcomes are known")
    )

    assert gap == (pytest.approx(0.087),) and gain_low == (pytest.approx(0.089),)
    assert list(listed["status"]) == ["warn", "ok"]  # newest run first
    assert float(listed.loc[0, "gap"]) == pytest.approx(0.087)
    assert listed.loc[0, "customers"] == 2775


@pytest.mark.usefixtures("warehouse")
def test_the_panels_are_empty_not_broken_when_no_outcomes_are_known(scratch_conn, mlflow_client):
    monitoring.refresh(scratch_conn, mlflow_client)

    assert _scalar(scratch_conn, _panel_sql("Calibration gap, latest run with outcomes")) is None
    assert _scalar(scratch_conn, _panel_sql("Gain over the recency rule, low end")) is None
    assert database.read_dataframe(
        scratch_conn, _panel_sql("Scoring runs whose outcomes are known")
    ).empty


@pytest.mark.usefixtures("warehouse")
@pytest.mark.parametrize(("status", "value"), [("alert", 1.0), ("warn", 0.0), ("ok", 0.0)])
def test_the_alert_rule_fires_only_for_the_alert_status_of_the_latest_run(
    scratch_conn, mlflow_client, monkeypatch, status, value
):
    older = _row_frame("alert").assign(
        scoring_run_id=3, snapshot_date=pd.Timestamp("2011-06-11").date()
    )
    _store(scratch_conn, mlflow_client, monkeypatch, older, _row_frame(status))  # the newer decides

    assert _scalar(scratch_conn, _rule_sql())[1] == value  # the row is (time, value)


@pytest.mark.usefixtures("warehouse")
def test_the_alert_rule_has_nothing_to_evaluate_when_no_outcomes_are_known(
    scratch_conn, mlflow_client
):
    monitoring.refresh(scratch_conn, mlflow_client)

    assert _scalar(scratch_conn, _rule_sql()) is None  # no data is normal for this rule


@pytest.mark.usefixtures("warehouse")
def test_each_customer_is_matched_to_their_own_outcome(scratch_conn):
    # a second customer in the labelled snapshot, who did the opposite of the first
    scratch_conn.execute(
        "CREATE TEMP TABLE second AS "
        "SELECT * FROM marts.customer_snapshots WHERE snapshot_date = %s",
        (SNAPSHOT_WITH_OUTCOMES,),
    )
    scratch_conn.execute(
        "UPDATE second SET customer_id = 999999, label_churned = NOT label_churned, "
        "recency_days = 7"
    )
    scratch_conn.execute("INSERT INTO marts.customer_snapshots SELECT * FROM second")
    run, _ = _score_the_labelled_snapshot(scratch_conn, p_churn=0.9)
    scratch_conn.execute(
        "INSERT INTO ml.predictions VALUES (%s, 999999, 0.1, 10, 5, 3, -2)", (run,)
    )
    scratch_conn.commit()

    joined = database.read_dataframe(scratch_conn, monitoring._REALISED_QUERY).sort_values(
        "p_churn"
    )

    assert len(joined) == 2  # not four: each customer meets only their own outcome
    assert list(joined["p_churn"]) == [pytest.approx(0.1), pytest.approx(0.9)]
    assert joined["recency_days"].iloc[0] == 7.0
    assert joined["churned"].iloc[0] != joined["churned"].iloc[1]


@pytest.mark.usefixtures("warehouse")
def test_the_rule_looks_at_the_newest_snapshot_even_if_an_older_one_was_scored_later(
    scratch_conn, mlflow_client, monkeypatch
):
    newest_snapshot_ok = _row_frame("ok").assign(scoring_run_id=3)
    older_snapshot_alert = _row_frame("alert").assign(
        scoring_run_id=8, snapshot_date=pd.Timestamp("2011-06-11").date()
    )  # the higher run id, but the older snapshot
    _store(scratch_conn, mlflow_client, monkeypatch, newest_snapshot_ok, older_snapshot_alert)

    assert _scalar(scratch_conn, _rule_sql())[1] == 0.0


@pytest.mark.usefixtures("warehouse")
def test_the_gap_panel_shows_the_size_of_a_gap_that_under_predicts(
    scratch_conn, mlflow_client, monkeypatch
):
    under = _row_frame("warn")
    under.loc[0, "calibration_gap"] = -0.087  # the model said less churn than happened
    _store(scratch_conn, mlflow_client, monkeypatch, under)

    assert _scalar(scratch_conn, _panel_sql("Calibration gap, latest run with outcomes")) == (
        pytest.approx(0.087),
    )
    listed = database.read_dataframe(
        scratch_conn, _panel_sql("Scoring runs whose outcomes are known")
    )
    assert float(listed.loc[0, "gap"]) == pytest.approx(-0.087)  # the table keeps the sign
