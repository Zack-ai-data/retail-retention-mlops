"""Scoring: the arithmetic, loading the live models, and writing ml.predictions."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src import config, database, features, hurdle, processor, scoring, tracking
from src.errors import EmptyTableError, RegistryError, ScoringError
from tests.conftest import SHEET_2
from tests.test_model import make_panel

UK = "United Kingdom"

# ------------------------------------------------------------------ stub models


class ChurnByRecency:
    """P(churn) rises with days since the last order."""

    def predict_proba(self, X):
        p = (X["recency_days"] / 200).clip(0.01, 0.99).to_numpy()
        return np.column_stack([1 - p, p])


class SpendByFrequency:
    def predict(self, X):
        return 100 + 10 * X["frequency_365d"].to_numpy()


def stub_champions():
    value = hurdle.HurdleValueModel(ChurnByRecency(), SpendByFrequency())
    return scoring.Champions(ChurnByRecency(), value, "7", "9")


def customers(**columns):
    n = len(next(iter(columns.values())))
    frame = pd.DataFrame({name: [1.0] * n for name in features.BASE_FEATURES})
    for name, values in columns.items():
        frame[name] = values
    return frame


# ------------------------------------------------------------------ the arithmetic


def test_margin_at_risk_is_churn_probability_times_spend_if_retained_times_margin():
    frame = customers(recency_days=[40.0, 100.0], frequency_365d=[2.0, 5.0])

    scores = scoring.score_features(frame, stub_champions(), margin=0.3, contact_cost=5.0)

    p_churn = np.array([0.2, 0.5])
    spend_if_retained = np.array([120.0, 150.0])
    assert scores["p_churn"].to_numpy() == pytest.approx(p_churn)
    assert scores["spend_if_retained"].to_numpy() == pytest.approx(spend_if_retained)
    assert scores["margin_at_risk"].to_numpy() == pytest.approx(p_churn * spend_if_retained * 0.3)
    assert scores["margin_at_risk_after_cost"].to_numpy() == pytest.approx(
        p_churn * spend_if_retained * 0.3 - 5.0
    )


def test_the_chance_of_buying_is_not_counted_twice():
    frame = customers(recency_days=[100.0], frequency_365d=[5.0])  # P(churn) = P(not buy) = 0.5

    scores = scoring.score_features(frame, stub_champions())

    # expected_spend = P(buy) x spend_if_retained already includes the chance of not buying;
    # margin at risk must use the spend if the customer buys, not expected_spend
    assert scores["expected_spend"].iloc[0] == pytest.approx(0.5 * 150.0)
    assert scores["margin_at_risk"].iloc[0] == pytest.approx(0.5 * 150.0 * config.MARGIN)
    assert scores["margin_at_risk"].iloc[0] != pytest.approx(
        0.5 * scores["expected_spend"].iloc[0] * config.MARGIN
    )


def test_the_defaults_are_the_assumptions_in_the_problem_statement():
    scores = scoring.score_features(customers(recency_days=[100.0]), stub_champions())

    assert config.MARGIN == 0.30 and config.CONTACT_COST_GBP == 5.0
    assert scores["margin_at_risk_after_cost"].iloc[0] == pytest.approx(
        scores["margin_at_risk"].iloc[0] - 5.0
    )


def test_a_customer_whose_margin_at_risk_is_below_the_contact_cost_is_negative_after_cost():
    scores = scoring.score_features(customers(recency_days=[5.0]), stub_champions())

    assert scores["margin_at_risk"].iloc[0] < config.CONTACT_COST_GBP
    assert scores["margin_at_risk_after_cost"].iloc[0] < 0


def test_the_index_is_kept_so_scores_line_up_with_customers():
    frame = customers(recency_days=[10.0, 20.0, 30.0])
    frame.index = [501, 502, 503]

    scores = scoring.score_features(frame, stub_champions())

    assert list(scores.index) == [501, 502, 503]
    assert list(scores.columns) == scoring.SCORE_COLUMNS


class BrokenChurn:
    def __init__(self, values):
        self.values = values

    def predict_proba(self, X):
        p = np.asarray(self.values, dtype=float)
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("bad", [np.nan, 1.5, -0.2])
def test_impossible_scores_are_refused_not_stored(bad):
    champions = stub_champions()
    champions = scoring.Champions(
        BrokenChurn([0.3, bad]), champions.value_model, champions.churn_version, "9"
    )

    with pytest.raises(ScoringError, match="invalid scores for 1 customers"):
        scoring.score_features(customers(recency_days=[10.0, 20.0]), champions)


def test_missing_feature_columns_are_reported():
    with pytest.raises(ValueError, match="Missing feature columns"):
        scoring.score_features(pd.DataFrame({"recency_days": [1.0]}), stub_champions())


# ------------------------------------------------------------------ loading the models


def _register(client, name, model, X, alias=True):
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X, np.zeros(len(X))),
            serialization_format="cloudpickle",
            registered_model_name=name,
        )
    if alias:
        client.set_registered_model_alias(
            name, tracking.CHAMPION_ALIAS, info.registered_model_version
        )
    return info.registered_model_version


@pytest.fixture(scope="module")
def trained():
    panel = make_panel(n=150)
    train = panel[(panel["split"] == "train") & panel["is_actionable"]]
    X = features.prepare_features(train)
    y = train[features.CHURN_TARGET].astype(int)
    churn = HistGradientBoostingClassifier(max_iter=20).fit(X, y)
    spend_y = train[features.VALUE_TARGET].to_numpy()
    spend = HistGradientBoostingRegressor(loss="gamma", max_iter=20).fit(
        X[spend_y > 0], spend_y[spend_y > 0]
    )
    return X, churn, hurdle.HurdleValueModel(churn, spend)


def test_both_live_models_are_loaded_with_their_versions(mlflow_client, trained):
    X, churn, value = trained
    _register(mlflow_client, config.CHURN_MODEL_NAME, churn, X)
    _register(mlflow_client, config.VALUE_MODEL_NAME, value, X)
    _register(mlflow_client, config.VALUE_MODEL_NAME, value, X)  # version 2 becomes champion

    champions = scoring.load_champions(mlflow_client)

    assert (champions.churn_version, champions.value_version) == ("1", "2")
    scores = scoring.score_features(X[features.BASE_FEATURES].head(5), champions)
    assert len(scores) == 5 and scores["p_churn"].between(0, 1).all()


def test_scoring_without_a_champion_says_to_train(mlflow_client):
    with pytest.raises(RegistryError, match=r"no champion.*--stage train"):
        scoring.load_champions(mlflow_client)


def test_a_missing_value_champion_is_reported_even_if_churn_exists(mlflow_client, trained):
    X, churn, _ = trained
    _register(mlflow_client, config.CHURN_MODEL_NAME, churn, X)

    with pytest.raises(RegistryError, match=config.VALUE_MODEL_NAME):
        scoring.load_champions(mlflow_client)


def test_a_champion_trained_on_other_columns_cannot_score(mlflow_client, trained):
    X, _, value = trained
    old = pd.DataFrame({"a": [0.0, 1.0] * 20, "old_feature": [1.0, 0.0] * 20})
    old_model = HistGradientBoostingClassifier(max_iter=5).fit(old, [0, 1] * 20)
    _register(mlflow_client, config.CHURN_MODEL_NAME, old_model, old)
    _register(mlflow_client, config.VALUE_MODEL_NAME, value, X)

    with pytest.raises(RegistryError, match="cannot score the current features"):
        scoring.load_champions(mlflow_client)


def test_a_value_champion_that_is_not_two_part_is_refused(mlflow_client, trained):
    X, churn, _ = trained
    _register(mlflow_client, config.CHURN_MODEL_NAME, churn, X)
    _register(
        mlflow_client,
        config.VALUE_MODEL_NAME,
        HistGradientBoostingRegressor(max_iter=5).fit(X, np.ones(len(X))),
        X,
    )

    with pytest.raises(RegistryError, match="not a two-part value model"):
        scoring.load_champions(mlflow_client)


# --------------------------------------------------- batch scoring into ml.predictions

# Customer 1 bought a week before the scoring snapshot, 2 a month before, 3 over a year before
RAW = [
    (SHEET_2, "800001", "A", "ITEM A", 2, "2011-12-02 10:00", 10, 1, UK),
    (SHEET_2, "800002", "A", "ITEM A", 1, "2011-12-05 10:00", 30, 1, UK),
    (SHEET_2, "800010", "A", "ITEM A", 1, "2011-11-05 10:00", 25, 2, UK),
    (SHEET_2, "800020", "A", "ITEM A", 1, "2010-06-01 10:00", 40, 3, UK),
]


@pytest.fixture
def scoring_db(scratch_conn, insert_raw, monkeypatch):
    insert_raw(RAW)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    monkeypatch.setattr(scoring, "load_champions", lambda client: stub_champions())
    return scratch_conn


def _runs(conn):
    return conn.execute(
        "SELECT snapshot_date::text, churn_model_version, value_model_version, margin, "
        "contact_cost, customers FROM ml.scoring_runs"
    ).fetchall()


def test_batch_scoring_stores_one_row_per_customer_in_the_population(scoring_db):
    run = scoring.score_batch(scoring_db, client=None)

    rows = scoring_db.execute(
        "SELECT customer_id FROM ml.predictions WHERE scoring_run_id = %s ORDER BY 1",
        (run.scoring_run_id,),
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2]  # customer 3 last bought over 180 days before: not scored
    assert run.customers == 2
    assert _runs(scoring_db) == [
        ("2011-12-10", "7", "9", config.MARGIN, config.CONTACT_COST_GBP, 2)
    ]


def test_the_stored_scores_are_what_score_features_returns(scoring_db):
    scoring.score_batch(scoring_db, client=None)

    stored = database.read_dataframe(
        scoring_db,
        "SELECT customer_id, p_churn, margin_at_risk FROM ml.latest_predictions ORDER BY 1",
    )
    snapshot = features.load_score_snapshot(scoring_db)
    expected = scoring.score_features(snapshot[snapshot["is_actionable"]], stub_champions())
    assert stored["p_churn"].to_numpy() == pytest.approx(expected["p_churn"].to_numpy())
    assert stored["margin_at_risk"].to_numpy() == pytest.approx(
        expected["margin_at_risk"].to_numpy()
    )


def test_each_scoring_run_is_kept_and_the_view_shows_the_latest(scoring_db):
    first = scoring.score_batch(scoring_db, client=None)
    second = scoring.score_batch(scoring_db, client=None)

    assert second.scoring_run_id > first.scoring_run_id
    assert len(_runs(scoring_db)) == 2
    latest = scoring_db.execute(
        "SELECT DISTINCT scoring_run_id FROM ml.latest_predictions"
    ).fetchall()
    assert latest == [(second.scoring_run_id,)]
    assert scoring_db.execute("SELECT count(*) FROM ml.predictions").fetchone()[0] == 4


def test_a_failed_write_leaves_no_scoring_run_behind(scoring_db, monkeypatch):
    def fail(conn, df, table):
        raise RuntimeError("disk full")

    monkeypatch.setattr(database, "copy_dataframe", fail)

    with pytest.raises(RuntimeError, match="disk full"):
        scoring.score_batch(scoring_db, client=None)

    assert _runs(scoring_db) == []  # the run row was rolled back with the predictions


def test_invalid_scores_are_not_written(scoring_db, monkeypatch):
    broken = scoring.Champions(BrokenChurn([0.5, 7.0]), stub_champions().value_model, "7", "9")
    monkeypatch.setattr(scoring, "load_champions", lambda client: broken)

    with pytest.raises(ScoringError):
        scoring.score_batch(scoring_db, client=None)

    assert _runs(scoring_db) == []


def test_scoring_before_the_transform_says_what_to_run(scratch_conn, monkeypatch):
    scratch_conn.execute(
        "CREATE TABLE marts.customer_snapshots AS SELECT 1 AS customer_id WHERE false"
    )
    monkeypatch.setattr(features, "load_score_snapshot", lambda conn: pd.DataFrame())

    with pytest.raises(EmptyTableError, match="--stage transform"):
        scoring.score_batch(scratch_conn, client=None)
