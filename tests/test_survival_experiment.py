"""The survival-scorer trial: the person-period rows are right, the score is the product of the
period survival chances, nothing after an origin is used, the measures and the verdict rule are
what the module says they are, and the rule is pinned so it cannot drift towards the answer.
"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import calibration_experiment, database, evaluation, features, measures, tracking
from src import survival_experiment as experiment
from tests.test_model import make_panel

HORIZON = timedelta(days=features.HORIZON_DAYS)
FIRST = date(2010, 3, 18)


def _dates(count):
    return [FIRST + i * HORIZON for i in range(count)]  # exactly one outcome window apart


def _panel(dates, seed=0, n=300):
    snapshots = [(str(d), "train") for d in dates[:-1]] + [(str(dates[-1]), "test")]
    return make_panel(seed=seed, n=n, snapshots=snapshots)


def _events_for(panel, seed=0):
    """Days to the first sale that agree with the churn label: nobody who churned bought, everybody
    else bought somewhere in the 90 days."""
    rng = np.random.default_rng(seed)
    bought = panel[features.CHURN_TARGET].to_numpy() == 0
    return pd.DataFrame(
        {
            "snapshot_date": panel.loc[bought, "snapshot_date"].to_numpy(),
            "customer_id": panel.loc[bought, "customer_id"].to_numpy(),
            "days_to_first_sale": rng.integers(0, features.HORIZON_DAYS, int(bought.sum())),
        }
    )


def _factory(params):
    # the real pipelines handle missing inputs; a plain logistic regression does not
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())


def _gain(low, high, diff=0.1):
    return evaluation.Gain(
        candidate=0.5 + diff, reference=0.5, diff=diff, low=low, high=high, resamples=1000
    )


@pytest.fixture(scope="module")
def trial():
    panel = _panel(_dates(7), seed=3)
    return panel, experiment.evaluate(panel, _events_for(panel, seed=3), _factory, {})


# ---------------------------------------------------------------- what is fixed in advance


def test_the_rule_and_the_settings_are_pinned_so_they_cannot_drift_towards_the_answer():
    assert experiment.INTERVAL_DAYS == 30 and experiment.N_INTERVALS == 3
    assert experiment.INTERVAL_DAYS * experiment.N_INTERVALS == features.HORIZON_DAYS
    assert experiment.TOP_FRACTION == 0.15
    assert experiment.WIN_SHARE == 0.75
    assert experiment.SPEND_COLUMN == "monetary_365d"  # longer than the 90 days that define churn
    assert experiment.METHODS == (
        "current",
        "survival",
        "recency",
        "bgnbd",
        "current_x_spend",
    )


# -------------------------------------------------------------- the person-period rows


def test_each_customer_contributes_a_row_for_every_period_they_were_still_waiting_at_the_start_of():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    days = [10.0, 35.0, 70.0, np.nan]  # first sale in period 1, 2, 3 and never

    rows, target = experiment.person_period(X, days)

    # customer 0 is at risk in period 1 only, 1 in periods 1-2, 2 and 3 in periods 1-3
    assert list(rows["a"]) == [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    assert list(rows[experiment.INTERVAL_COLUMN]) == [1.0] * 4 + [2.0] * 3 + [3.0] * 2
    assert list(target) == [1, 0, 0, 0, 1, 0, 0, 1, 0]


@pytest.mark.parametrize(
    ("day", "period"),
    [(0, 1), (29, 1), (30, 2), (59, 2), (60, 3), (89, 3)],
)
def test_the_period_boundaries_fall_on_whole_multiples_of_the_interval(day, period):
    rows, target = experiment.person_period(pd.DataFrame({"a": [1.0]}), [float(day)])

    assert len(rows) == period  # at risk in every period up to and including the one it happened
    assert list(target) == [0] * (period - 1) + [1]
    assert rows[experiment.INTERVAL_COLUMN].iloc[-1] == period


def test_a_customer_who_bought_has_exactly_one_event_and_one_who_never_did_has_none():
    rng = np.random.default_rng(0)
    days = np.where(rng.random(200) < 0.3, np.nan, rng.integers(0, 90, 200).astype(float))
    X = pd.DataFrame({"a": np.arange(200.0)})

    rows, target = experiment.person_period(X, days)

    bought = ~np.isnan(days)
    assert target.sum() == bought.sum()
    for customer in (np.flatnonzero(bought)[0], np.flatnonzero(~bought)[0]):
        mine = target[(rows["a"] == customer).to_numpy()]
        assert mine.sum() == (1 if bought[customer] else 0)
        assert len(mine) == (int(days[customer] // 30) + 1 if bought[customer] else 3)


def test_the_original_features_are_carried_unchanged_into_every_row():
    X = pd.DataFrame({"a": [1.5, 2.5], "b": [10.0, np.nan]})

    rows, _ = experiment.person_period(X, [np.nan, np.nan])

    assert list(rows.columns) == ["a", "b", experiment.INTERVAL_COLUMN]
    assert rows["a"].tolist() == [1.5, 2.5] * 3
    assert rows["b"].isna().tolist() == [False, True] * 3


# ------------------------------------------------------------------- the survival score


class _FixedHazards:
    """A stand-in classifier whose chance of a purchase depends only on the period."""

    def __init__(self, by_period):
        self.by_period = by_period

    def predict_proba(self, X):
        p = X[experiment.INTERVAL_COLUMN].map(self.by_period).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_the_score_is_the_chance_of_getting_through_every_period_without_a_purchase():
    X = pd.DataFrame({"a": [1.0, 2.0]})

    score = experiment.survival_scores(_FixedHazards({1.0: 0.2, 2.0: 0.5, 3.0: 0.1}), X)

    assert score == pytest.approx([0.8 * 0.5 * 0.9] * 2)


def test_no_chance_of_a_purchase_means_certain_churn_and_a_sure_purchase_means_none():
    X = pd.DataFrame({"a": [1.0]})

    assert experiment.survival_scores(_FixedHazards({1.0: 0, 2.0: 0, 3.0: 0}), X)[0] == 1.0
    assert experiment.survival_scores(_FixedHazards({1.0: 0, 2.0: 1, 3.0: 0}), X)[0] == 0.0


def test_a_higher_chance_of_buying_in_any_period_lowers_the_score():
    X = pd.DataFrame({"a": [1.0]})
    base = {1.0: 0.2, 2.0: 0.2, 3.0: 0.2}
    reference = experiment.survival_scores(_FixedHazards(base), X)[0]

    for period in base:
        raised = {**base, period: 0.3}
        assert experiment.survival_scores(_FixedHazards(raised), X)[0] < reference


def test_the_features_given_to_the_classifier_are_not_changed_by_scoring():
    X = pd.DataFrame({"a": [1.0, 2.0]})

    experiment.survival_scores(_FixedHazards({1.0: 0.2, 2.0: 0.2, 3.0: 0.2}), X)

    assert list(X.columns) == ["a"]


def test_a_hazard_model_trained_on_a_known_process_ranks_the_customers_who_wait_longest_highest():
    rng = np.random.default_rng(0)
    n = 4000
    slow = rng.normal(size=n)  # the higher it is, the lower the daily chance of a purchase
    daily = 0.03 * np.exp(-1.2 * slow)
    first_sale = rng.geometric(np.clip(daily, 1e-4, 0.99)) - 1  # days waited before a purchase
    days = np.where(first_sale < features.HORIZON_DAYS, first_sale, np.nan).astype(float)
    X = pd.DataFrame({"slow": slow})

    fitted = experiment.fit_hazard(_factory, {}, X, days)
    score = experiment.survival_scores(fitted, X)

    waited = np.isnan(days)
    assert roc_auc_score(waited, score) > 0.75
    assert np.corrcoef(score, slow)[0, 1] > 0.5


# --------------------------------------------------------------------- the measures


def test_a_churner_leaves_their_spend_behind_and_a_customer_who_stayed_leaves_none():
    left = experiment.churner_spend([1, 1, 0, 1, 1], [100.0, -30.0, 500.0, np.nan, 0.0])

    assert list(left) == [100.0, 0.0, 0.0, 0.0, 0.0]  # no spend below zero, none for a customer


def test_the_spend_share_is_the_churners_spend_held_by_the_top_scored_customers():
    churned = [1, 1, 0, 1]
    spend = [100.0, 50.0, 500.0, 10.0]  # the 500 is a customer who did not churn: not at stake
    score = [0.9, 0.1, 0.8, 0.5]

    # the top half is customers 0 and 2: 100 of the 160 the churners leave behind
    assert experiment.spend_share_top(churned, spend, score, fraction=0.5) == pytest.approx(
        100 / 160
    )


def test_a_perfect_ranking_reaches_everything_and_a_useless_one_reaches_its_share():
    rng = np.random.default_rng(1)
    churned = (rng.random(4000) < 0.4).astype(float)
    spend = rng.uniform(10, 100, 4000)

    perfect = experiment.spend_share_top(churned, spend, churned + rng.random(4000) * 0.01, 0.3)
    useless = experiment.spend_share_top(churned, spend, rng.random(4000), 0.3)

    assert perfect > 0.7  # a 30% group can hold at most all the churners' spend (40% churn)
    assert useless == pytest.approx(0.3, abs=0.04)


def test_no_churner_spend_gives_no_share_not_a_division_by_zero():
    assert np.isnan(experiment.spend_share_top([0, 0, 1], [5.0, 5.0, 0.0], [0.1, 0.2, 0.3]))


def test_scores_that_tie_at_the_cut_are_split_so_row_order_cannot_change_the_answer():
    churned = np.array([1.0, 1.0, 1.0, 1.0])
    spend = np.array([10.0, 20.0, 30.0, 40.0])
    tied = np.array([0.5, 0.5, 0.5, 0.5])

    first = experiment.spend_share_top(churned, spend, tied, 0.5)
    second = experiment.spend_share_top(churned[::-1], spend[::-1], tied[::-1], 0.5)

    assert first == second == pytest.approx(0.5)


# ----------------------------------------------------------------------- the verdict


@pytest.mark.parametrize(
    ("wins", "origins", "low", "high", "expected"),
    [
        (3, 4, 0.01, 0.2, "better"),  # leads at 3 of 4 and the interval is above zero
        (4, 4, 0.01, 0.2, "better"),
        (2, 4, 0.01, 0.2, "no better"),  # a positive interval is not enough without the wins
        (4, 4, 0.0, 0.2, "no better"),  # an interval that touches zero is not above it
        (4, 4, -0.01, 0.2, "no better"),
        (3, 4, -0.05, -0.001, "worse"),  # the interval is entirely below zero
        (0, 4, -0.05, 0.0, "no better"),  # an upper end exactly at zero is not clearly worse
        (1, 4, -0.1, 0.1, "no better"),
        (6, 8, 0.01, 0.2, "better"),  # 75% of 8 origins is 6
        (5, 8, 0.01, 0.2, "no better"),
        (0, 0, 0.01, 0.2, "no better"),  # no origins to compare is never a win
    ],
)
def test_the_verdict_rule(wins, origins, low, high, expected):
    assert experiment.decide(wins, origins, _gain(low, high)) == expected


def _origin(seed, n=1500, signal=0.0):
    rng = np.random.default_rng(seed)
    churned = (rng.random(n) < 0.45).astype(float)
    spend = rng.uniform(20, 200, n)
    return experiment.OriginData(
        origin=f"o{seed}",
        churned=churned,
        spend=spend,
        scores={
            "candidate": churned * signal + rng.random(n),
            "reference": rng.random(n),
            "same": None,
        },
    )


def _pair(signals):
    origins = [_origin(i, signal=s) for i, s in enumerate(signals)]
    for o in origins:
        o.scores["same"] = o.scores["reference"].copy()
    return experiment.Trial(pd.DataFrame(), origins)


def test_a_method_compared_with_an_identical_one_has_no_advantage_and_a_zero_width_interval():
    gain = experiment.pooled_gain(_pair([0.5, 0.5]), "same", "reference", n_resamples=200)

    assert gain.diff == 0 and gain.low == 0 and gain.high == 0


def test_a_clearly_better_ranking_has_an_interval_above_zero_and_a_worse_one_below():
    trial = _pair([0.6, 0.6, 0.6, 0.6])

    better = experiment.pooled_gain(trial, "candidate", "reference", n_resamples=200)
    worse = experiment.pooled_gain(trial, "reference", "candidate", n_resamples=200)

    assert better.low > 0 and better.diff > 0
    assert worse.high < 0 and worse.diff == pytest.approx(-better.diff)
    assert better.low < better.diff < better.high  # the interval is around the average, not a sum


def test_the_point_advantage_is_the_average_over_the_origins_of_each_origins_difference():
    trial = _pair([0.3, 0.6, 0.9])
    per_origin = [
        experiment.spend_share_top(o.churned, o.spend, o.scores["candidate"])
        - experiment.spend_share_top(o.churned, o.spend, o.scores["reference"])
        for o in trial.origins
    ]

    gain = experiment.pooled_gain(trial, "candidate", "reference", n_resamples=100)

    assert gain.diff == pytest.approx(np.mean(per_origin))
    assert gain.low < gain.diff < gain.high  # the interval is for that average, not for a sum


def test_the_interval_has_the_confidence_level_the_other_comparisons_use(monkeypatch):
    trial = _pair([0.3, 0.3])
    wide = experiment.pooled_gain(trial, "candidate", "reference", 400)

    monkeypatch.setattr(evaluation, "CONFIDENCE_LEVEL", 0.5)
    narrow = experiment.pooled_gain(trial, "candidate", "reference", 400)

    assert (narrow.high - narrow.low) < 0.6 * (wide.high - wide.low)


def test_the_shared_confidence_level_is_95_percent():
    assert evaluation.CONFIDENCE_LEVEL == 0.95


def test_the_interval_is_repeatable_for_a_seed_and_moves_with_it():
    trial = _pair([0.2, 0.2])

    a = experiment.pooled_gain(trial, "candidate", "reference", 200, seed=1)
    b = experiment.pooled_gain(trial, "candidate", "reference", 200, seed=1)
    c = experiment.pooled_gain(trial, "candidate", "reference", 200, seed=2)

    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high)


def test_origins_that_lack_one_of_the_methods_are_left_out_and_none_left_is_an_error():
    trial = _pair([0.5, 0.5])
    trial.origins[0].scores.pop("candidate")

    assert experiment.origins_won(trial, "candidate", "reference")[1] == 1
    assert experiment.pooled_gain(trial, "candidate", "reference", 50).resamples > 0
    with pytest.raises(ValueError, match="No origin"):
        experiment.pooled_gain(trial, "candidate", "missing", 50)


def test_a_tie_at_an_origin_is_not_a_win():
    trial = _pair([0.5, 0.5, 0.5])

    assert experiment.origins_won(trial, "same", "reference") == (0, 3)
    assert experiment.origins_won(trial, "candidate", "reference") == (3, 3)


def test_the_comparison_carries_the_counts_the_advantage_and_the_verdict_that_follow_from_them():
    result = experiment.compare(_pair([0.6, 0.6, 0.6, 0.6]), "candidate", "reference", 200)

    assert result["origins_won"] == 4 and result["origins"] == 4
    assert result["verdict"] == "better"
    assert result["interval"][0] > 0 and result["mean_advantage"] > 0
    assert result["candidate"] == "candidate" and result["reference"] == "reference"


# ---------------------------------------------------------------------- the evaluation


def test_every_origin_with_enough_earlier_history_is_evaluated_and_none_before_it(trial):
    _, result = trial

    assert [o.origin for o in result.origins] == [str(d) for d in _dates(7)[3:]]
    assert list(result.table["origin"].unique()) == [str(d) for d in _dates(7)[3:]]


def test_every_origin_scores_the_methods_that_can_be_computed_on_the_same_customers(trial):
    panel, result = trial
    actionable = panel[panel["is_actionable"].astype(bool)]

    for o in result.origins:
        rows = result.table[result.table["origin"] == o.origin]
        customers = int(((actionable["snapshot_date"].astype(str)) == o.origin).sum())
        assert {"current", "survival", "recency", "current_x_spend"} <= set(rows["method"])
        assert (rows["customers"] == customers).all()
        assert all(len(s) == customers for s in o.scores.values())
        assert (rows["churners"] == rows["churners"].iloc[0]).all()


def test_the_recency_scores_are_the_days_since_the_last_order(trial):
    panel, result = trial
    origin = result.origins[0]
    rows = panel[
        (panel["snapshot_date"].astype(str) == origin.origin) & panel["is_actionable"].astype(bool)
    ]

    assert origin.scores["recency"] == pytest.approx(rows["recency_days"].to_numpy())


def test_the_bgnbd_score_is_one_minus_the_chance_the_customer_is_still_alive(trial):
    panel, result = trial
    origin = result.origins[0]
    previous = _dates(7)[2]  # the latest snapshot whose outcomes were known before the origin
    cohort = panel[panel["snapshot_date"] == previous]
    rows = panel[
        (panel["snapshot_date"].astype(str) == origin.origin) & panel["is_actionable"].astype(bool)
    ]
    fitted = experiment.baselines.BgNbd.fit(
        cohort["frequency"], cohort["recency_days"], cohort["tenure_days"]
    )

    alive = fitted.probability_alive(rows["frequency"], rows["recency_days"], rows["tenure_days"])

    assert origin.scores["bgnbd"] == pytest.approx(1 - alive)


def test_the_spend_at_stake_is_the_365_day_spend_not_the_90_day_one(trial):
    panel, result = trial
    last = result.origins[-1]
    rows = panel[
        (panel["snapshot_date"].astype(str) == last.origin) & panel["is_actionable"].astype(bool)
    ]

    assert last.spend == pytest.approx(rows[experiment.SPEND_COLUMN].to_numpy(dtype=float))
    assert not np.allclose(last.spend, rows["spend_last_90d"].to_numpy(dtype=float), equal_nan=True)


def test_the_stand_in_for_margin_at_risk_is_the_probability_times_the_spend(trial):
    _, result = trial
    for o in result.origins:
        expected = o.scores["current"] * np.clip(o.spend, 0, None)
        assert o.scores["current_x_spend"] == pytest.approx(expected)


def test_the_survival_scorer_finds_the_structure_in_a_panel_where_recency_decides_churn(trial):
    _, result = trial
    survival = result.table[result.table["method"] == "survival"]

    base_rate = survival["churners"] / survival["customers"]
    assert (survival["pr_auc"] > base_rate + 0.1).all()
    assert (survival["recall_top"] > experiment.TOP_FRACTION + 0.05).all()  # a useless one gets 15%


def test_the_measures_in_the_table_are_the_measures_of_the_scores_that_were_stored(trial):
    _, result = trial
    for o in result.origins:
        for method, score in o.scores.items():
            (row,) = result.table[
                (result.table["origin"] == o.origin) & (result.table["method"] == method)
            ].to_dict("records")
            assert row["spend_share_top"] == pytest.approx(
                experiment.spend_share_top(o.churned, o.spend, score, fraction=0.15)
            )
            assert row["recall_top"] == pytest.approx(
                measures.recall_at_top_fraction(o.churned, score, 0.15)
            )
            assert row["precision_top"] == pytest.approx(
                measures.precision_at_top_fraction(o.churned, score, 0.15)
            )
            assert row["pr_auc"] == pytest.approx(average_precision_score(o.churned, score))


def test_an_origin_never_uses_anything_that_happened_after_it():
    panel = _panel(_dates(7), seed=4)
    events = _events_for(panel, seed=4)
    original = experiment.evaluate(panel, events, _factory, {})

    # turn the outcomes of the last two snapshots upside down (churners become buyers and back)
    later = panel["snapshot_date"].isin(_dates(7)[5:])
    flipped = panel.copy()
    flipped.loc[later, features.CHURN_TARGET] = 1 - flipped.loc[later, features.CHURN_TARGET]
    changed = experiment.evaluate(flipped, _events_for(flipped, seed=4), _factory, {})

    # an origin trains on snapshots whose outcomes were known by then, so the origins up to the
    # fifth see the same models; the last one trains on the fifth snapshot's outcomes and does not
    for a, b in zip(original.origins[:-1], changed.origins[:-1], strict=True):
        for method in ("current", "survival"):
            assert a.scores[method] == pytest.approx(b.scores[method], abs=1e-12)
    last_a, last_b = original.origins[-1], changed.origins[-1]
    assert not np.allclose(last_a.scores["survival"], last_b.scores["survival"])


def test_the_trial_is_repeatable():
    panel = _panel(_dates(6), seed=5)
    events = _events_for(panel, seed=5)

    first = experiment.evaluate(panel, events, _factory, {})
    second = experiment.evaluate(panel, events, _factory, {})

    pd.testing.assert_frame_equal(first.table, second.table)


def test_how_often_the_first_sale_and_the_zero_net_spend_label_disagree_is_reported(trial):
    _, result = trial

    assert (result.table["label_mismatch"] == 0).all()  # here the events agree with the labels


def test_a_customer_with_a_sale_who_still_counts_as_churned_is_counted_as_a_disagreement():
    panel = _panel(_dates(6), seed=6)
    events = _events_for(panel, seed=6)
    last = _dates(6)[-1]
    churned = panel[
        (panel["snapshot_date"] == last)
        & (panel[features.CHURN_TARGET] == 1)
        & panel["is_actionable"].astype(bool)
    ]
    extra = pd.DataFrame(
        {
            "snapshot_date": [last],
            "customer_id": [int(churned["customer_id"].iloc[0])],
            "days_to_first_sale": [5],
        }
    )

    result = experiment.evaluate(panel, pd.concat([events, extra]), _factory, {})

    rows = result.table[result.table["origin"] == str(last)]
    actionable = panel[(panel["snapshot_date"] == last) & panel["is_actionable"].astype(bool)]
    assert rows["label_mismatch"].iloc[0] == pytest.approx(1 / len(actionable))


def test_a_baseline_that_cannot_be_fitted_is_left_out_without_stopping_the_rest(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("BG/NBD did not converge")

    monkeypatch.setattr(experiment.baselines.BgNbd, "fit", fail)
    panel = _panel(_dates(6), seed=7)

    result = experiment.evaluate(panel, _events_for(panel, seed=7), _factory, {})

    assert "bgnbd" not in set(result.table["method"])
    assert {"current", "survival", "recency", "current_x_spend"} <= set(result.table["method"])


def test_bgnbd_is_fitted_only_on_customers_whose_outcomes_were_known_before_the_origin(
    monkeypatch,
):
    seen = []
    original = experiment.baselines.BgNbd.fit.__func__

    def spy(cls, frequency, recency_days, tenure_days):
        seen.append(np.asarray(frequency, dtype=float))
        return original(cls, frequency, recency_days, tenure_days)

    monkeypatch.setattr(experiment.baselines.BgNbd, "fit", classmethod(spy))
    panel = _panel(_dates(6), seed=8, n=300)

    experiment.evaluate(panel, _events_for(panel, seed=8), _factory, {})

    # one fit per origin, each on every customer of the latest snapshot whose outcomes were known
    # (all 300, not only the actionable ones): the snapshot just before the origin
    dates = _dates(6)
    assert len(seen) == 3
    for fitted_on, previous in zip(seen, dates[2:5], strict=True):
        expected = panel[panel["snapshot_date"] == previous]["frequency"].to_numpy(dtype=float)
        assert fitted_on == pytest.approx(expected)


# -------------------------------------------------------------- the event times in SQL


def _build_event_tables(conn):
    conn.execute("DROP TABLE IF EXISTS staging.transactions")
    conn.execute(
        """
        CREATE TABLE staging.transactions (
            customer_id integer, invoice_date timestamp, line_type text
        )
        """
    )
    conn.execute(
        "CREATE TABLE marts.snapshot_dates "
        "(snapshot_date date PRIMARY KEY, split text, horizon_days integer)"
    )
    conn.execute(
        """
        INSERT INTO marts.snapshot_dates VALUES
            ('2011-01-01', 'train', 90), ('2011-04-01', 'test', 90), ('2011-07-01', 'score', 90)
        """
    )
    rows = [
        (1, datetime(2011, 1, 11, 10, 0), "sale"),  # 10 days in
        (1, datetime(2011, 1, 20, 9, 0), "sale"),  # a later sale does not matter
        (2, datetime(2011, 3, 31, 23, 59), "sale"),  # day 89: the last day of the first window
        (3, datetime(2011, 4, 1, 0, 0), "sale"),  # midnight at the next snapshot: its day 0
        (4, datetime(2011, 1, 5, 12, 0), "return"),  # a return is not a purchase
        (5, datetime(2011, 1, 1, 0, 0), "sale"),  # midnight at the snapshot itself: day 0
        (None, datetime(2011, 1, 3, 12, 0), "sale"),  # no customer id
        (6, datetime(2011, 2, 1, 12, 0), "non_product"),  # postage and the like
        (7, datetime(2011, 7, 5, 12, 0), "sale"),  # inside the score snapshot's window: not used
        (1, datetime(2011, 4, 4, 12, 0), "sale"),  # customer 1 again, in the next snapshot
    ]
    for row in rows:
        conn.execute("INSERT INTO staging.transactions VALUES (%s, %s, %s)", row)
    conn.commit()


def test_the_days_to_the_first_sale_are_read_for_every_labelled_snapshot(scratch_conn):
    _build_event_tables(scratch_conn)

    events = experiment.load_event_days(scratch_conn)

    found = {
        (str(r.snapshot_date), int(r.customer_id)): int(r.days_to_first_sale)
        for r in events.itertuples()
    }
    assert found == {
        ("2011-01-01", 1): 10,
        ("2011-01-01", 2): 89,
        ("2011-01-01", 5): 0,
        ("2011-04-01", 3): 0,
        ("2011-04-01", 1): 3,
    }


def test_the_event_days_line_up_with_the_snapshot_rows_they_belong_to(scratch_conn):
    _build_event_tables(scratch_conn)
    events = experiment.load_event_days(scratch_conn)
    frame = pd.DataFrame(
        {
            "snapshot_date": [date(2011, 1, 1)] * 3 + [date(2011, 4, 1)] * 3,
            "customer_id": [1, 2, 4, 3, 9, 1],
        }
    )

    days = experiment.attach_event_days(frame, events)

    assert days[:2].tolist() == [10.0, 89.0]
    assert np.isnan(days[2])  # customer 4 only returned something
    assert days[3] == 0.0
    assert np.isnan(days[4])  # nobody bought
    assert days[5] == 3.0  # customer 1 in the later snapshot is not their 10 days in the first


# ----------------------------------------------------------------------- run and main


def test_run_evaluates_the_warehouse_with_the_champions_settings(monkeypatch):
    panel = _panel(_dates(7), seed=9)
    events = _events_for(panel, seed=9)
    monkeypatch.setattr(tracking, "configure", lambda: None)
    monkeypatch.setattr(
        calibration_experiment,
        "champion_settings",
        lambda client: ("logistic_regression", {"C": 1.0}),
    )

    @contextmanager
    def connection():
        yield None

    monkeypatch.setattr(database, "get_connection", connection)
    monkeypatch.setattr(features, "load_snapshots", lambda conn: panel)
    monkeypatch.setattr(experiment, "load_event_days", lambda conn: events)

    result, verdict, stand_in = experiment.run()

    assert {"current", "survival"} <= set(result.table["method"])
    assert verdict["candidate"] == "survival" and verdict["reference"] == "current"
    assert stand_in["candidate"] == "current_x_spend" and stand_in["reference"] == "current"
    assert verdict["verdict"] in {"better", "worse", "no better"}


def test_the_score_snapshot_is_never_evaluated_because_it_has_no_outcomes(monkeypatch):
    panel = make_panel(seed=10, n=300)  # train x3, test, and a score snapshot with no labels
    events = _events_for(panel[panel["split"] != "score"], seed=10)
    monkeypatch.setattr(tracking, "configure", lambda: None)
    monkeypatch.setattr(
        calibration_experiment, "champion_settings", lambda client: ("logistic_regression", {})
    )

    @contextmanager
    def connection():
        yield None

    monkeypatch.setattr(database, "get_connection", connection)
    monkeypatch.setattr(features, "load_snapshots", lambda conn: panel)
    monkeypatch.setattr(experiment, "load_event_days", lambda conn: events)

    result, _, _ = experiment.run()

    assert "2011-12-10" not in set(result.table["origin"])


def test_the_command_prints_the_table_and_the_verdict_and_exits_cleanly(monkeypatch, capsys):
    panel = _panel(_dates(7), seed=11)
    result = experiment.evaluate(panel, _events_for(panel, seed=11), _factory, {})
    monkeypatch.setattr(
        experiment,
        "run",
        lambda: (
            result,
            experiment.compare(result, "survival", "current", 50),
            experiment.compare(result, "current_x_spend", "current", 50),
        ),
    )

    assert experiment.main() == 0

    shown = capsys.readouterr().out
    assert "spend_share_top" in shown and "survival" in shown
    assert "verdict" in shown and "stand-in" in shown
