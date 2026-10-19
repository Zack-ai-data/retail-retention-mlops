import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

from src import (
    backtesting,
    calibration,
    config,
    features,
    hurdle,
    measures,
    model,
    model_search,
    segmentation,
    tracking,
    transforms,
)

# ----------------------------------------------------------- synthetic data


def make_panel(
    seed: int = 0, n: int = 400, snapshots: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Customer snapshots shaped like marts.customer_snapshots, with learnable structure:
    customers churn more when recency is high and frequency is low."""
    rng = np.random.default_rng(seed)
    snapshots = snapshots or [
        ("2010-12-13", "train"),
        ("2011-03-13", "train"),
        ("2011-06-11", "train"),
        ("2011-09-09", "test"),
        ("2011-12-10", "score"),
    ]
    frames = []
    for date, split in snapshots:
        recency = rng.integers(1, 300, n).astype(float)
        frequency = rng.integers(1, 25, n).astype(float)
        avg_order = rng.lognormal(5, 0.6, n)
        monetary = frequency * avg_order
        spend_last = monetary * rng.uniform(0.0, 0.4, n)
        logit = (recency - 120) / 40 - 0.12 * frequency
        churned = rng.random(n) < 1 / (1 + np.exp(-logit))
        future_spend = np.where(churned, 0.0, avg_order * rng.uniform(0.5, 2.5, n))
        frame = pd.DataFrame(
            {
                "snapshot_date": pd.Timestamp(date).date(),
                "split": split,
                "customer_id": np.arange(n),
                "recency_days": recency,
                "is_actionable": recency <= features.ACTIONABLE_RECENCY_DAYS,
                "tenure_days_capped": np.minimum(recency + rng.integers(0, 400, n), 365),
                "history_days_available": np.full(n, 365.0),
                "frequency_365d": frequency,
                "monetary_365d": monetary,
                "avg_order_value_365d": avg_order,
                "total_units_365d": frequency * rng.integers(1, 30, n),
                "distinct_products_365d": rng.integers(1, 60, n).astype(float),
                "avg_days_between_orders_365d": np.where(
                    frequency > 1, rng.uniform(5, 120, n), np.nan
                ),
                "orders_last_90d": rng.integers(0, 4, n).astype(float),
                "orders_prev_90d": rng.integers(0, 4, n).astype(float),
                "spend_last_90d": spend_last,
                "return_invoices_365d": rng.integers(0, 2, n).astype(float),
                "return_value_365d": rng.uniform(0, 50, n),
                "return_rate_365d": rng.uniform(0, 0.2, n),
                "is_uk": rng.integers(0, 2, n).astype(float),
                "season_sin": np.sin(2 * np.pi * pd.Timestamp(date).dayofyear / 365.25),
                "season_cos": np.cos(2 * np.pi * pd.Timestamp(date).dayofyear / 365.25),
                # lifetime columns: not model inputs, but customer segmentation (RFM) reads these
                "frequency": frequency,
                "monetary": monetary,
                "tenure_days": recency + rng.integers(0, 400, n),
                features.CHURN_TARGET: churned.astype(float),
                features.VALUE_TARGET: future_spend,
            }
        )
        if split == "score":
            frame[[features.CHURN_TARGET, features.VALUE_TARGET]] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _churn_stage(panel):
    """A quickly fitted churn model to serve as the value model's "will they buy" stage."""
    train = panel[(panel["split"] == "train") & panel["is_actionable"]]
    factory = model_search.candidate_factory(model_search.CHURN_CANDIDATES, "logistic_regression")
    return factory({"C": 1.0}).fit(
        features.prepare_features(train), train[features.CHURN_TARGET].astype(int)
    )


def _train_value(panel, client):
    return model.train_value(panel, client, _churn_stage(panel))


# ----------------------------------------------------------------- features


def test_prepare_features_adds_derived_columns():
    df = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in features.BASE_FEATURES})
    df["recency_days"] = [10.0, 20.0, 30.0]
    df["avg_days_between_orders_365d"] = [5.0, np.nan, 0.0]
    df["orders_last_90d"] = [3.0, 1.0, 0.0]
    df["orders_prev_90d"] = [1.0, 2.0, 0.0]

    out = features.prepare_features(df)

    assert list(out.columns) == features.FEATURE_COLUMNS
    assert out["recency_ratio"].iloc[0] == 2.0
    assert np.isnan(out["recency_ratio"].iloc[1])  # one-time buyer: no rhythm to compare with
    assert np.isnan(out["recency_ratio"].iloc[2])  # division by zero becomes NaN, not inf
    assert out["orders_trend"].tolist() == [2.0, -1.0, 0.0]


def test_prepare_features_rejects_missing_columns():
    df = pd.DataFrame({"recency_days": [1.0]})
    with pytest.raises(ValueError, match="Missing feature columns"):
        features.prepare_features(df)


def test_prepare_features_does_not_modify_input():
    df = make_panel(n=20).head(5)
    before = df.copy()
    features.prepare_features(df)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------------------------ metrics


def test_expanding_window_folds_train_on_the_past_only():
    dates = ["d1", "d1", "d2", "d3", "d3"]
    assert model_search.expanding_window_folds(dates) == [(["d1"], "d2"), (["d1", "d2"], "d3")]


def test_recall_at_top_fraction():
    y = [1, 0, 1, 0, 0]
    score = [0.9, 0.8, 0.7, 0.1, 0.0]
    assert measures.recall_at_top_fraction(y, score, 0.4) == 0.5  # top 2 hold 1 of the 2 positives
    assert measures.recall_at_top_fraction(y, score, 0.6) == 1.0


def test_expected_calibration_error_separates_calibrated_from_overconfident():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < prob).astype(float)
    assert measures.expected_calibration_error(y, prob) < 0.05
    assert measures.expected_calibration_error(y, np.full(5000, 0.99)) > 0.3


def test_top_fraction_capture():
    actual = [100, 0, 50, 0, 0, 0, 0, 0, 0, 0]
    assert measures.top_fraction_capture(
        actual, [9, 1, 8, 0, 0, 0, 0, 0, 0, 0], 0.1
    ) == pytest.approx(100 / 150)


def test_value_metrics_perfect_and_all_zero_predictions():
    perfect = measures.value_metrics([0, 10, 20], [0, 10, 20], "")
    assert (
        perfect["rmse"] == 0
        and perfect["bias_pct"] == 0
        and perfect["spearman"] == pytest.approx(1.0)
    )

    zeros = measures.value_metrics([0, 10, 20], [0, 0, 0], "")
    assert zeros["wape"] == pytest.approx(1.0)
    assert zeros["bias_pct"] == pytest.approx(-100.0)
    assert np.isnan(zeros["spearman"])  # constant predictions have no ranking


def test_signed_log1p_handles_zero_and_negative_values():
    out = transforms.signed_log1p(np.array([-3.0, 0.0, 3.0]))
    assert out[1] == 0
    assert out[0] == pytest.approx(-out[2])


# --------------------------------------------------------- model selection


def test_select_model_prefers_the_better_candidate_using_only_the_past():
    panel = make_panel(n=300)
    train = panel[panel["split"] == "train"].reset_index(drop=True)
    X = features.prepare_features(train)
    y = train[features.CHURN_TARGET].to_numpy(dtype=int)
    candidates = [
        model_search.Candidate(
            "logistic",
            lambda p: make_pipeline(*model_search.linear_preprocessing(), LogisticRegression(**p)),
            {"C": [1.0]},
        ),
        model_search.Candidate("no_skill", lambda p: DummyClassifier(strategy="prior"), {}),
    ]

    selection = model_search.select_model(
        candidates,
        X,
        y,
        train["snapshot_date"],
        score_fn=average_precision_score,
        predict_fn=lambda m, data: m.predict_proba(data)[:, 1],
        higher_is_better=True,
    )

    assert selection.algorithm == "logistic"
    assert len(selection.results) == 2
    assert selection.results["cv_score"].is_monotonic_decreasing


def test_select_model_needs_two_snapshots():
    panel = make_panel(n=50)
    one = panel[panel["snapshot_date"] == panel["snapshot_date"].iloc[0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="two training snapshots"):
        model_search.select_model(
            model_search.CHURN_CANDIDATES,
            features.prepare_features(one),
            one[features.CHURN_TARGET].to_numpy(dtype=int),
            one["snapshot_date"],
            score_fn=average_precision_score,
            predict_fn=lambda m, d: m.predict_proba(d)[:, 1],
            higher_is_better=True,
        )


# ------------------------------------------------------------ segment names


def test_name_segments_uses_value_rank_and_recency_and_stays_unique():
    profile = pd.DataFrame({"median_recency_days": [20.0, 300.0, 100.0, 30.0]}, index=[0, 1, 2, 3])
    # scaled centroids: columns are recency, frequency, monetary
    centroids = np.array([[-1.0, 2.0, 2.0], [1.5, -1.5, -1.5], [0.0, 0.0, 0.0], [-0.8, 1.0, 1.0]])

    names = segmentation.name_segments(profile, centroids)

    assert names[0] == "High value / Active"
    assert names[1] == "Low value / Lapsed"
    assert names[2] == "Mid value / Cooling"
    assert len(set(names.values())) == 4

    clash = segmentation.name_segments(
        pd.DataFrame({"median_recency_days": [10.0, 20.0]}, index=[0, 1]),
        np.array([[0.0, 1.0, 1.0], [0.0, 0.9, 0.9]]),
    )
    assert clash[0] != clash[1]  # same descriptor gets a numeric suffix


# ------------------------------------------------- end to end on a local store


def test_train_churn_logs_registers_and_promotes_a_model_that_beats_recency(mlflow_client):
    labelled = _labelled(_signal_panel())

    first = model.train_churn(labelled, mlflow_client)

    assert first["promoted"] is True
    assert first["metrics"]["test_pr_auc"] > first["metrics"]["baseline_recency_test_pr_auc"]
    assert {"test_roc_auc", "test_brier", "test_ece", "test_pr_auc_lift"} <= set(first["metrics"])
    champion = mlflow_client.get_model_version_by_alias(
        config.CHURN_MODEL_NAME, tracking.CHAMPION_ALIAS
    )
    assert str(champion.version) == "1"
    run = mlflow_client.get_run(first["run_id"])
    assert run.data.tags["promotion"].startswith("promoted")
    assert "data_fingerprint" in run.data.tags
    assert run.data.params["test_snapshot"] == "2011-09-09"
    assert len(mlflow_client.search_model_versions(f"name='{config.CHURN_MODEL_NAME}'")) == 1


def test_train_churn_never_sees_the_test_snapshot_when_choosing_hyperparameters(mlflow_client):
    panel = make_panel()
    labelled = panel[panel["split"].isin(["train", "test"])]

    result = model.train_churn(labelled, mlflow_client)

    run = mlflow_client.get_run(result["run_id"])
    assert run.data.params["train_snapshots"] == "2010-12-13,2011-03-13,2011-06-11"
    actionable = labelled[labelled["is_actionable"]]
    assert run.data.params["n_train_rows"] == str((actionable["split"] == "train").sum())
    assert run.data.params["n_test_rows"] == str((actionable["split"] == "test").sum())


def test_train_value_records_the_gate_decision(mlflow_client):
    panel = make_panel()
    labelled = panel[panel["split"].isin(["train", "test"])]

    result = _train_value(labelled, mlflow_client)

    assert {"test_rmse", "test_bias_pct", "baseline_naive_test_rmse", "cv_gamma_deviance"} <= set(
        result["metrics"]
    )
    versions = mlflow_client.search_model_versions(f"name='{config.VALUE_MODEL_NAME}'")
    assert len(versions) == 1
    assert versions[0].tags["promotion"].split(":")[0] in {"promoted", "rejected"}
    assert result["promoted"] == (
        result["metrics"]["test_rmse"] < result["metrics"]["baseline_naive_test_rmse"]
    )


def test_train_segments_writes_assignments_and_registers_champion(mlflow_client, scratch_conn):
    panel = make_panel()

    result = segmentation.train_segments(panel, scratch_conn, mlflow_client)

    rows = scratch_conn.execute(
        "SELECT count(*), count(DISTINCT customer_id), min(snapshot_date), "
        "count(DISTINCT segment_name) "
        "FROM marts.customer_segments"
    ).fetchone()
    assert rows[0] == rows[1] == 400
    assert str(rows[2]) == "2011-12-10"
    assert 1 <= rows[3] <= result["k"]
    champion = mlflow_client.get_model_version_by_alias(
        config.SEGMENT_MODEL_NAME, tracking.CHAMPION_ALIAS
    )
    assert str(champion.version) == "1"

    # a second run replaces the table instead of appending to it
    segmentation.train_segments(panel, scratch_conn, mlflow_client)
    assert scratch_conn.execute("SELECT count(*) FROM marts.customer_segments").fetchone()[0] == 400


def test_segments_require_enough_scoring_customers(mlflow_client):
    with pytest.raises(ValueError, match="Too few customers"):
        segmentation.train_segments(make_panel(n=10), None, mlflow_client)


def test_segmentation_copes_with_customers_whose_returns_exceed_their_purchases(
    mlflow_client, scratch_conn
):
    panel = make_panel()
    in_score = panel["split"] == "score"
    panel.loc[in_score & (panel["customer_id"] < 10), "monetary"] = -50.0  # net of returns, below 0

    result = segmentation.train_segments(panel, scratch_conn, mlflow_client)

    assert result["k"] in segmentation.SEGMENT_K_RANGE
    assert scratch_conn.execute("SELECT count(*) FROM marts.customer_segments").fetchone()[0] == 400


# ------------------------------------------------------- actionable population


def test_recency_band_metrics_report_each_band_separately():
    y = [1, 0, 1, 1, 0, 1]
    recency = [10, 20, 80, 100, 150, 180]
    score = [0.9, 0.1, 0.8, 0.7, 0.2, 0.6]

    out = measures.recency_band_metrics(y, score, recency, "t_")

    assert out["t_band_le90_n"] == 3 and out["t_band_91_180_n"] == 3
    assert out["t_band_le90_base_rate"] == pytest.approx(2 / 3)
    assert out["t_band_91_180_base_rate"] == pytest.approx(2 / 3)
    assert out["t_band_le90_pr_auc"] == pytest.approx(1.0)  # both churners ranked first
    assert out["t_band_91_180_pr_auc"] == pytest.approx(1.0)


def test_recency_band_metrics_skip_pr_auc_when_a_band_has_one_label():
    out = measures.recency_band_metrics([1, 1, 0, 1], [0.9, 0.8, 0.1, 0.5], [10, 20, 30, 120], "t_")

    assert out["t_band_le90_pr_auc"] == pytest.approx(1.0)
    assert out["t_band_91_180_n"] == 1
    assert "t_band_91_180_pr_auc" not in out  # a single customer: nothing to rank


def test_recency_band_boundaries_are_inclusive_at_90_and_180():
    out = measures.recency_band_metrics([1, 0, 1, 0], [1, 1, 1, 1], [90, 91, 180, 181], "t_")

    assert out["t_band_le90_n"] == 1  # 90
    assert out["t_band_91_180_n"] == 2  # 91 and 180
    assert out["t_band_le90_n"] + out["t_band_91_180_n"] == 3  # 181 is in no band


def test_models_train_only_on_actionable_customers(mlflow_client):
    panel = make_panel()

    result = model.train_churn(panel[panel["split"] != "score"], mlflow_client)

    run = mlflow_client.get_run(result["run_id"])
    actionable = panel[(panel["split"] == "train") & panel["is_actionable"]]
    assert (
        int(run.data.params["n_train_rows"])
        == len(actionable)
        < len(panel[panel["split"] == "train"])
    )
    assert run.data.params["population"] == f"recency_days<={features.ACTIONABLE_RECENCY_DAYS}"
    assert "test_band_le90_pr_auc" in run.data.metrics
    assert "baseline_recency_test_band_91_180_pr_auc" in run.data.metrics


def test_the_value_model_also_trains_on_actionable_customers_only(mlflow_client):
    panel = make_panel()

    result = _train_value(panel[panel["split"] != "score"], mlflow_client)

    run = mlflow_client.get_run(result["run_id"])
    actionable = panel[(panel["split"] == "test") & panel["is_actionable"]]
    assert int(run.data.params["n_test_rows"]) == len(actionable)


def test_training_needs_actionable_rows(mlflow_client):
    panel = make_panel()
    panel["is_actionable"] = False

    with pytest.raises(ValueError, match="actionable rows"):
        model.train_churn(panel[panel["split"] != "score"], mlflow_client)


# ------------------------------------------------------------- the backtest


def _logistic():
    return model_search.candidate_factory(model_search.CHURN_CANDIDATES, "logistic_regression"), {
        "C": 1.0
    }


def test_the_backtest_trains_only_on_snapshots_whose_label_window_has_ended():
    # snapshots 45 days apart with 90-day label windows: the two previous snapshots' windows
    # still overlap the origin and must be dropped ("purging")
    dates = pd.date_range("2010-01-01", periods=7, freq="45D").strftime("%Y-%m-%d")
    panel = make_panel(snapshots=[(d, "train") for d in dates], n=300)
    factory, params = _logistic()

    table = backtesting.churn_backtest(panel, factory, params)

    assert table["origin"].tolist() == [dates[4], dates[5], dates[6]]  # at least 3 usable
    assert table["train_snapshots"].tolist() == [3, 4, 5]  # origin i can use snapshots 0..i-2


def test_with_snapshots_exactly_one_horizon_apart_every_earlier_snapshot_is_usable():
    start = pd.Timestamp("2010-01-01")
    dates = [(start + pd.Timedelta(days=90 * i)).strftime("%Y-%m-%d") for i in range(6)]
    panel = make_panel(snapshots=[(d, "train") for d in dates], n=300)
    factory, params = _logistic()

    table = backtesting.churn_backtest(panel, factory, params)

    assert table["train_snapshots"].tolist() == [3, 4, 5]  # origin i uses all i earlier ones


def test_the_backtest_reports_the_gain_over_recency_with_its_interval():
    panel = make_panel(n=300)
    labelled = panel[panel["split"].isin(["train", "test"])]
    factory, params = _logistic()

    table = backtesting.churn_backtest(labelled, factory, params)

    row = table.iloc[0]
    assert row["origin"] == "2011-09-09"
    assert row["gain"] == pytest.approx(row["model_pr_auc"] - row["recency_pr_auc"])
    assert row["gain_ci_low"] < row["gain"] < row["gain_ci_high"]
    assert bool(row["significant"]) == (row["gain_ci_low"] > 0)


def test_no_origin_has_enough_history_is_an_error_in_the_summary():
    with pytest.raises(ValueError, match="no origins"):
        backtesting.backtest_summary(pd.DataFrame())


def test_the_backtest_summary_counts_significant_origins():
    table = pd.DataFrame({"significant": [True, False, True, True]})

    assert backtesting.backtest_summary(table) == {
        "backtest_origins": 4.0,
        "backtest_origins_significant": 3.0,
        "backtest_share_significant": 0.75,
    }


def test_out_of_fold_predictions_come_from_models_that_saw_only_earlier_snapshots():
    sizes = []

    class Recorder:
        def __init__(self, params):
            pass

        def fit(self, X, y):
            sizes.append(len(X))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])

    dates = pd.Series(["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4)
    X = pd.DataFrame({"x": range(12)})
    y = np.arange(12) % 2

    labels, predictions, snapshots = model_search.out_of_fold_predictions(
        Recorder, {}, X, y, dates, lambda m, data: m.predict_proba(data)[:, 1]
    )

    assert sizes == [4, 8]  # d2 predicted by a model trained on d1; d3 on d1+d2
    assert labels.tolist() == y[4:].tolist()  # the first snapshot is never predicted
    assert predictions.tolist() == [0.5] * 8
    assert snapshots.tolist() == ["d2"] * 4 + ["d3"] * 4


# --------------------------------------------- churn model: calibration, gate, champion


def _signal_panel(**kwargs):
    """Churn driven by frequency, which recency alone cannot see: the model must beat recency."""
    panel = make_panel(**kwargs)
    panel[features.CHURN_TARGET] = (panel["frequency"] < 8).astype(float)
    panel.loc[panel["split"] == "score", features.CHURN_TARGET] = np.nan
    return panel


def _labelled(panel):
    return panel[panel["split"].isin(["train", "test"])]


def _gate_json(run_id):
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/gate.json")


def test_the_churn_run_records_calibration_the_interval_the_backtest_and_the_bgnbd_baseline(
    mlflow_client,
):
    result = model.train_churn(_labelled(_signal_panel()), mlflow_client)

    metrics = mlflow_client.get_run(result["run_id"]).data.metrics
    for name in (
        "calibration_slope",
        "uncalibrated_test_ece",
        "test_ece",
        "test_pr_auc_vs_recency_gain",
        "test_pr_auc_vs_recency_gain_ci_low",
        "test_pr_auc_vs_recency_gain_ci_high",
        "backtest_origins",
        "backtest_share_significant",
        "baseline_bgnbd_test_pr_auc",
        "test_precision_at_top20pct",
    ):
        assert name in metrics, name


def test_the_registered_churn_model_returns_calibrated_probabilities(mlflow_client):
    model.train_churn(_labelled(_signal_panel()), mlflow_client)

    registered = mlflow.sklearn.load_model(f"models:/{config.CHURN_MODEL_NAME}/1")

    assert isinstance(registered, calibration.PlattCalibratedClassifier)
    X = features.prepare_features(_signal_panel().head(10))
    assert registered.predict_proba(X).sum(axis=1) == pytest.approx(np.ones(10))


def test_a_model_that_clearly_beats_recency_is_promoted_with_its_evidence(mlflow_client):
    result = model.train_churn(_labelled(_signal_panel()), mlflow_client)

    assert result["promoted"] is True
    assert result["reason"].startswith("passed: beats the recency baseline")
    assert "no champion yet" in result["reason"]
    saved = _gate_json(result["run_id"])
    assert saved["promoted"] is True
    assert [c["name"] for c in saved["checks"]] == [
        "beats the recency baseline",
        "gain holds across backtest origins",
    ]
    champion = mlflow_client.get_model_version_by_alias(
        config.CHURN_MODEL_NAME, tracking.CHAMPION_ALIAS
    )
    assert str(champion.version) == "1"


def test_a_model_with_no_skill_is_rejected_and_no_alias_is_set(mlflow_client):
    panel = make_panel()
    rng = np.random.default_rng(5)
    panel[features.CHURN_TARGET] = (rng.random(len(panel)) < 0.5).astype(float)  # unrelated labels
    panel.loc[panel["split"] == "score", features.CHURN_TARGET] = np.nan

    result = model.train_churn(_labelled(panel), mlflow_client)

    assert result["promoted"] is False
    assert result["reason"].startswith("failed:")
    registered = mlflow_client.get_registered_model(config.CHURN_MODEL_NAME)
    assert tracking.CHAMPION_ALIAS not in registered.aliases
    version = mlflow_client.get_model_version(config.CHURN_MODEL_NAME, "1")
    assert version.tags["promotion"].startswith("rejected: failed:")  # the reason stays with it


def test_a_second_run_re_scores_the_champion_on_the_same_data(mlflow_client):
    panel = _labelled(_signal_panel())
    first = model.train_churn(panel, mlflow_client)

    second = model.train_churn(panel, mlflow_client)

    metrics = mlflow_client.get_run(second["run_id"]).data.metrics
    assert metrics["champion_rescored_test_pr_auc"] == pytest.approx(
        first["metrics"]["test_pr_auc"]
    )  # same data and seeds: the champion scores what it scored before
    assert "champion v1 re-scored on the same data" in second["reason"]
    assert second["promoted"] is True  # equal to the champion is allowed
    assert "not worse than the champion" in [
        c["name"] for c in _gate_json(second["run_id"])["checks"]
    ]


def test_a_better_champion_is_not_replaced_by_a_worse_candidate(mlflow_client):
    good = _labelled(_signal_panel())
    model.train_churn(good, mlflow_client)  # v1 becomes champion

    # a candidate trained and tested on much noisier labels cannot beat v1 re-scored on them
    noisy = good.copy()
    flip = np.random.default_rng(1).random(len(noisy)) < 0.35
    noisy.loc[flip, features.CHURN_TARGET] = 1 - noisy.loc[flip, features.CHURN_TARGET]
    result = model.train_churn(noisy, mlflow_client)

    saved = _gate_json(result["run_id"])
    champion_check = next(c for c in saved["checks"] if c["name"] == "not worse than the champion")
    if not champion_check["passed"]:
        champion = mlflow_client.get_model_version_by_alias(
            config.CHURN_MODEL_NAME, tracking.CHAMPION_ALIAS
        )
        assert str(champion.version) == "1"  # the alias did not move
        assert result["promoted"] is False
    else:  # the noisy-label candidate happened to match v1 on these labels: the gate allowed it
        assert result["promoted"] is True


def test_a_champion_trained_on_other_columns_is_not_compared_but_the_baseline_still_is(
    mlflow_client,
):
    old = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 2)), columns=["a", "old_feature"])
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            LogisticRegression().fit(old, (old["a"] > 0).astype(int)),
            name="model",
            signature=infer_signature(old, np.zeros(50)),
            serialization_format="cloudpickle",
            registered_model_name=config.CHURN_MODEL_NAME,
        )
    mlflow_client.set_registered_model_alias(
        config.CHURN_MODEL_NAME, tracking.CHAMPION_ALIAS, info.registered_model_version
    )

    result = model.train_churn(_labelled(_signal_panel()), mlflow_client)

    assert (
        "was not compared because it was trained on different feature columns" in result["reason"]
    )
    assert result["promoted"] is True  # it still had to beat the recency baseline to get here
    assert "champion_rescored_test_pr_auc" not in result["metrics"]
    assert [c["name"] for c in _gate_json(result["run_id"])["checks"]] == [
        "beats the recency baseline",
        "gain holds across backtest origins",
    ]


def test_the_bgnbd_baseline_is_skipped_not_fatal_when_it_cannot_be_fitted(caplog):
    panel = _labelled(make_panel(n=20))  # fewer than 50 customers per snapshot
    test = panel[panel["split"] == "test"]

    with caplog.at_level("WARNING", logger=backtesting.log.name):
        scores = backtesting.bgnbd_churn_scores(panel, test)

    assert scores is None
    assert "BG/NBD baseline skipped" in caplog.text


def test_bgnbd_scores_are_churn_probabilities_for_the_test_customers():
    panel = _labelled(make_panel())
    test = panel[panel["split"] == "test"]

    scores = backtesting.bgnbd_churn_scores(panel, test)

    assert scores is not None and len(scores) == len(test)
    assert ((scores >= 0) & (scores <= 1)).all()


# ------------------------------------------------------------- value model: gate


def test_the_value_run_records_its_gains_and_the_gate_checks(mlflow_client):
    result = _train_value(_labelled(make_panel()), mlflow_client)

    metrics = mlflow_client.get_run(result["run_id"]).data.metrics
    for name in (
        "test_spearman_vs_naive_gain_ci_low",
        "test_top_decile_capture_vs_naive_gain",
        "test_neg_winsorised_rmse_vs_naive_gain_ci_high",
        "winsorise_cap",
        "baseline_naive_test_bias_pct",
    ):
        assert name in metrics, name
    saved = _gate_json(result["run_id"])
    assert len(saved["checks"]) == 4  # no champion yet: the champion check is absent
    assert saved["promoted"] == result["promoted"] == all(c["passed"] for c in saved["checks"])
    assert "no champion yet" in result["reason"]


def test_the_winsorised_error_ignores_the_largest_spenders(mlflow_client):
    panel = _labelled(make_panel())
    test_rows = panel["split"] == "test"
    panel.loc[test_rows & (panel["customer_id"] < 3), features.VALUE_TARGET] = 1e7  # 3 giants

    result = _train_value(panel, mlflow_client)

    metrics = result["metrics"]
    assert metrics["test_rmse"] > 10 * metrics["test_winsorised_rmse"]  # raw RMSE is dominated
    assert metrics["winsorise_cap"] < 1e5  # the cap comes from the training labels


def test_the_naive_value_forecast_is_never_negative(mlflow_client):
    panel = _labelled(make_panel())
    in_test = panel["split"] == "test"
    # net of returns, half the customers are below zero; the rest spent 10
    panel["spend_last_90d"] = np.where(panel["customer_id"] % 2 == 0, -500.0, 10.0)

    result = _train_value(panel, mlflow_client)

    test = panel[in_test & panel["is_actionable"]]
    actual = test[features.VALUE_TARGET].sum()
    forecast = np.clip(test["spend_last_90d"], 0, None).sum()  # negative spend counts as zero
    assert result["metrics"]["baseline_naive_test_bias_pct"] == pytest.approx(
        100 * (forecast - actual) / actual
    )


def test_a_naive_forecast_that_cannot_be_ranked_stops_training_with_a_clear_message(
    mlflow_client,
):
    panel = _labelled(make_panel())
    panel["spend_last_90d"] = 0.0  # every customer forecast the same: Spearman is undefined

    with pytest.raises(ValueError, match="too small or has too little variety"):
        _train_value(panel, mlflow_client)


def test_calibration_is_only_applied_when_it_helped_on_the_latest_fold(mlflow_client):
    result = model.train_churn(_labelled(_signal_panel()), mlflow_client)

    metrics = result["metrics"]
    assert metrics["calibration_applied"] in (0.0, 1.0)
    if metrics["calibration_applied"] == 0.0:  # identity mapping: nothing changes
        assert metrics["calibration_slope"] == 1.0 and metrics["calibration_intercept"] == 0.0
        assert metrics["test_ece"] == pytest.approx(metrics["uncalibrated_test_ece"], abs=1e-4)
    else:
        assert metrics["calibration_slope"] != 1.0


# ----------------------------------------------------------- value model: two parts


def test_the_spend_stage_is_trained_on_customers_who_bought_only(mlflow_client, monkeypatch):
    seen = []
    real_select = model_search.select_model

    def spy(candidates, X, y, *args, **kwargs):
        seen.append((candidates, np.asarray(y)))
        return real_select(candidates, X, y, *args, **kwargs)

    monkeypatch.setattr(model_search, "select_model", spy)

    _train_value(_labelled(make_panel()), mlflow_client)

    spend_stage = next(y for candidates, y in seen if candidates is model_search.SPEND_CANDIDATES)
    assert len(spend_stage) > 0 and (spend_stage > 0).all()
    single_stage = next(y for candidates, y in seen if candidates is model_search.VALUE_CANDIDATES)
    assert (single_stage == 0).any()  # the single-stage comparison still sees the non-buyers


def test_the_value_model_is_the_churn_stage_times_the_spend_stage(mlflow_client):
    panel = _labelled(make_panel())

    result = _train_value(panel, mlflow_client)

    hurdle_model = result["estimator"]
    X = features.prepare_features(panel[panel["split"] == "test"].head(25))
    expected = hurdle_model.predict_buy_probability(X) * hurdle_model.predict_spend_if_buying(X)
    assert hurdle_model.predict(X) == pytest.approx(expected)
    assert hurdle_model.predict_buy_probability(X) == pytest.approx(
        1 - hurdle_model.churn_model.predict_proba(X)[:, 1]
    )
    assert result["algorithm"].startswith("hurdle_")


def test_the_registered_value_model_is_the_two_part_model(mlflow_client):
    _train_value(_labelled(make_panel()), mlflow_client)

    registered = mlflow.sklearn.load_model(f"models:/{config.VALUE_MODEL_NAME}/1")

    assert isinstance(registered, hurdle.HurdleValueModel)
    X = features.prepare_features(make_panel().head(8))
    assert (registered.predict(X) >= 0).all()


def test_the_value_run_explains_where_the_two_parts_are_right_or_wrong(mlflow_client):
    panel = _labelled(make_panel())

    result = _train_value(panel, mlflow_client)

    metrics = result["metrics"]
    test = panel[(panel["split"] == "test") & panel["is_actionable"]]
    bought = test[features.VALUE_TARGET] > 0
    assert metrics["test_buy_rate"] == pytest.approx(bought.mean())
    assert metrics["test_spend_if_buying_actual_mean"] == pytest.approx(
        test.loc[bought, features.VALUE_TARGET].mean()
    )
    assert 0 < metrics["test_buy_probability_mean"] < 1
    assert metrics["test_spend_if_buying_predicted_mean"] > 0


def test_the_single_stage_model_is_reported_for_comparison_but_does_not_decide(mlflow_client):
    result = _train_value(_labelled(make_panel()), mlflow_client)

    metrics = result["metrics"]
    for name in (
        "single_stage_test_rmse",
        "single_stage_test_spearman",
        "single_stage_test_bias_pct",
    ):
        assert name in metrics
    run = mlflow_client.get_run(result["run_id"])
    assert run.data.params["single_stage_algorithm"] in {
        "hist_gradient_boosting_poisson",
        "poisson_glm",
    }
    assert run.data.params["hurdle_buy_stage"] == "Pipeline"
    gate_names = [c["name"] for c in _gate_json(result["run_id"])["checks"]]
    assert not any("single" in name for name in gate_names)


# ----------------------------------------------------------------- lineage

LINEAGE = {"git_sha": "abc1234", "raw_file_sha256": "f" * 64, "lib.scikit-learn": "1.9.1"}


def test_a_churn_run_carries_the_lineage_tags_and_the_datasets_it_used(mlflow_client):
    result = model.train_churn(_labelled(_signal_panel()), mlflow_client, LINEAGE)

    run = mlflow_client.get_run(result["run_id"])
    for key, value in LINEAGE.items():
        assert run.data.tags[key] == value
    inputs = {i.dataset.name: i.tags[0].value for i in run.inputs.dataset_inputs}
    assert inputs == {"train snapshots": "training", "test snapshot": "testing"}


def test_a_value_run_carries_the_lineage_tags_too(mlflow_client):
    result = _train_value(_labelled(make_panel()), mlflow_client)  # without lineage: still works
    assert "git_sha" not in mlflow_client.get_run(result["run_id"]).data.tags

    tagged = model.train_value(
        _labelled(make_panel()), mlflow_client, _churn_stage(make_panel()), LINEAGE
    )

    assert mlflow_client.get_run(tagged["run_id"]).data.tags["git_sha"] == "abc1234"


def test_a_segmentation_run_carries_the_lineage_tags_too(mlflow_client, scratch_conn):
    result = segmentation.train_segments(make_panel(), scratch_conn, mlflow_client, LINEAGE)

    assert mlflow_client.get_run(result["run_id"]).data.tags["raw_file_sha256"] == "f" * 64


def test_the_datasets_logged_with_a_run_include_the_target_column(mlflow_client):
    result = model.train_churn(_labelled(_signal_panel()), mlflow_client)

    run = mlflow_client.get_run(result["run_id"])
    train_input = next(i for i in run.inputs.dataset_inputs if i.dataset.name == "train snapshots")
    assert features.CHURN_TARGET in train_input.dataset.schema
