"""Drift: the population stability index, the same-season reference, and what is left out."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import features, monitoring
from tests.test_model import make_panel

RNG = np.random.default_rng(0)


def _series(values):
    return pd.Series(np.asarray(values, dtype=float))


# ------------------------------------------------------------------------------------- psi


def test_the_same_distribution_has_almost_no_drift():
    a, b = RNG.normal(size=5000), RNG.normal(size=5000)

    assert monitoring.psi(_series(a), _series(b)) < 0.02


def test_identical_values_have_exactly_zero_drift():
    values = _series(RNG.normal(size=500))

    assert monitoring.psi(values, values) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("shift", "at_least"), [(0.5, 0.1), (1.0, 0.25), (3.0, 1.0)])
def test_a_shifted_distribution_scores_higher_the_further_it_moved(shift, at_least):
    a = RNG.normal(size=5000)

    assert monitoring.psi(_series(a), _series(a + shift)) >= at_least


def test_drift_is_not_negative():
    assert monitoring.psi(_series(RNG.normal(size=800)), _series(RNG.normal(1, 2, 800))) >= 0


def test_a_column_with_two_values_is_compared_value_by_value():
    reference = _series([0] * 500 + [1] * 500)
    same = _series([0] * 480 + [1] * 520)
    moved = _series([0] * 100 + [1] * 900)

    assert monitoring.psi(reference, same) < 0.01
    assert monitoring.psi(reference, moved) > 0.25


def test_a_value_the_reference_never_had_counts_as_drift():
    reference = _series([0] * 500 + [1] * 500)
    new = _series([0] * 400 + [1] * 400 + [2] * 200)

    assert monitoring.psi(reference, new) > 0.25


def test_values_going_missing_is_drift():
    reference = _series(RNG.normal(size=1000))
    gappy = reference.copy()
    gappy.iloc[:400] = np.nan

    assert monitoring.psi(reference, gappy) > 0.25


def test_missing_values_in_both_at_the_same_rate_are_not_drift():
    a = _series(RNG.normal(size=4000))
    b = _series(RNG.normal(size=4000))
    a.iloc[:800], b.iloc[:800] = np.nan, np.nan

    assert monitoring.psi(a, b) < 0.02


@pytest.mark.parametrize("small", [0, 10, monitoring.DRIFT_MIN_CUSTOMERS - 1])
def test_too_few_customers_say_nothing(small):
    enough = _series(RNG.normal(size=500))

    assert monitoring.psi(enough, _series(RNG.normal(size=small))) is None
    assert monitoring.psi(_series(RNG.normal(size=small)), enough) is None


def test_a_reference_with_no_values_at_all_says_nothing():
    assert monitoring.psi(_series([np.nan] * 100), _series(RNG.normal(size=100))) is None


# ---------------------------------------------------------------------------------- status


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.0, "ok"), (0.0999, "ok"), (0.1, "warn"), (0.2499, "warn"), (0.25, "alert"), (5.0, "alert")],
)
def test_the_thresholds_are_the_conventional_ones_and_the_boundaries_belong_to_the_higher_status(
    value, status
):
    assert (monitoring.PSI_WARN, monitoring.PSI_ALERT) == (0.1, 0.25)
    assert monitoring.drift_status(value) == status


# ---------------------------------------------------------------------------- the reference


def test_the_reference_is_the_snapshot_a_year_earlier():
    dates = [date(2010, 9, 14), date(2010, 12, 13), date(2011, 3, 13), date(2011, 12, 10)]

    assert monitoring.reference_snapshot(date(2011, 12, 10), dates) == date(2010, 12, 13)


def test_with_no_snapshot_a_year_earlier_there_is_nothing_to_compare_with():
    dates = [date(2011, 3, 13), date(2011, 6, 11), date(2011, 9, 9), date(2011, 12, 10)]

    assert monitoring.reference_snapshot(date(2011, 12, 10), dates) is None


def test_of_two_candidates_the_one_nearest_364_days_is_chosen():
    dates = [date(2010, 12, 1), date(2010, 12, 13), date(2011, 12, 10)]  # 374 and 362 days before

    assert monitoring.reference_snapshot(date(2011, 12, 10), dates) == date(2010, 12, 13)


# --------------------------------------------------------------------------- the table


class _ChurnByRecency:
    def predict_proba(self, X):
        p = (X["recency_days"] / 200).clip(0.01, 0.99).to_numpy()
        return np.column_stack([1 - p, p])


def _panel(**kwargs):
    return make_panel(n=600, **kwargs)


def test_the_same_season_a_year_apart_with_no_change_has_no_drift():
    panel = _panel(seed=1, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])
    # identical customers in both snapshots (and the same ones scored): nothing moved
    columns = [*features.BASE_FEATURES, "is_actionable"]
    panel.loc[panel["snapshot_date"] == date(2011, 12, 10), columns] = panel.loc[
        panel["snapshot_date"] == date(2010, 12, 13), columns
    ].to_numpy()

    table = monitoring.drift_table(panel, _ChurnByRecency())

    assert set(table["status"]) == {"ok"} and table["psi"].max() < 0.01
    assert set(table["snapshot_date"]) == {date(2011, 12, 10)}
    assert set(table["reference_date"]) == {date(2010, 12, 13)}


def test_a_deliberately_shifted_snapshot_raises_alerts_on_the_inputs_that_moved():
    panel = _panel(seed=2, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])
    latest = panel["snapshot_date"] == date(2011, 12, 10)
    panel.loc[latest, "frequency_365d"] *= 3  # customers suddenly buy three times as often
    panel.loc[latest, "return_rate_365d"] += 0.5

    table = monitoring.drift_table(panel, _ChurnByRecency()).set_index("column_name")

    assert table.loc["frequency_365d", "status"] == "alert"
    assert table.loc["return_rate_365d", "status"] == "alert"
    assert table.loc["is_uk", "status"] == "ok"  # what did not move stays ok


def test_a_shift_that_changes_the_churn_probabilities_is_seen_in_p_churn():
    panel = _panel(seed=3, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])
    panel.loc[panel["snapshot_date"] == date(2011, 12, 10), "recency_days"] *= 0.4

    table = monitoring.drift_table(panel, _ChurnByRecency()).set_index("column_name")

    assert table.loc[monitoring.CHURN_PROBABILITY, "status"] == "alert"


def test_without_a_champion_the_churn_probability_is_left_out_and_the_inputs_are_not():
    panel = _panel(seed=4, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])

    table = monitoring.drift_table(panel, None)

    assert monitoring.CHURN_PROBABILITY not in set(table["column_name"])
    assert set(table["column_name"]) == set(monitoring.DRIFT_COLUMNS)


def test_columns_that_are_not_about_customers_are_never_compared():
    panel = _panel(seed=5, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])
    panel.loc[panel["snapshot_date"] == date(2011, 12, 10), "tenure_days_capped"] += 100

    columns = set(monitoring.drift_table(panel, _ChurnByRecency())["column_name"])

    assert not columns & {
        "season_sin",
        "season_cos",
        "history_days_available",
        "tenure_days_capped",
    }
    assert {
        "season_sin",
        "season_cos",
        "history_days_available",
        "tenure_days_capped",
    } == monitoring.DRIFT_EXCLUDED


def test_the_columns_compared_are_the_model_inputs_less_the_excluded_ones():
    assert (
        set(monitoring.DRIFT_COLUMNS) == set(features.FEATURE_COLUMNS) - monitoring.DRIFT_EXCLUDED
    )
    assert "frequency_365d" in monitoring.DRIFT_COLUMNS


def test_snapshots_with_no_year_earlier_snapshot_get_no_rows_not_an_error():
    panel = _panel(seed=6)  # five snapshots, but only the December pair is a year apart
    table = monitoring.drift_table(panel, _ChurnByRecency())

    assert set(table["snapshot_date"]) == {date(2011, 12, 10)}

    short = _panel(seed=6, snapshots=[("2011-03-13", "train"), ("2011-06-11", "score")])
    assert monitoring.drift_table(short, _ChurnByRecency()).empty
    assert (
        list(monitoring.drift_table(short, _ChurnByRecency()).columns)
        == monitoring.DRIFT_TABLE_COLUMNS
    )


def test_only_the_customers_the_models_score_are_compared():
    panel = _panel(seed=7, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])
    latest = panel["snapshot_date"] == date(2011, 12, 10)
    lapsed = latest & ~panel["is_actionable"]
    panel.loc[lapsed, "frequency_365d"] = 9999.0  # wild values, but only among customers not scored

    table = monitoring.drift_table(panel, _ChurnByRecency()).set_index("column_name")

    assert table.loc["frequency_365d", "status"] == "ok"


def test_the_drift_table_holds_numbers_per_column_and_no_customer():
    panel = _panel(seed=8, snapshots=[("2010-12-13", "train"), ("2011-12-10", "score")])

    table = monitoring.drift_table(panel, _ChurnByRecency())

    assert list(table.columns) == monitoring.DRIFT_TABLE_COLUMNS
    assert "customer_id" not in table.columns
    assert len(table) == len(monitoring.DRIFT_COLUMNS) + 1  # every input, and p_churn


def test_a_column_with_a_few_values_is_not_binned_by_interpolated_edges():
    # regression test: decile edges of a 0/1 column fall at 0.5, which would put a new value 2 in
    # the same bin as the 1s and hide it
    reference = _series([0] * 500 + [1] * 500)

    assert monitoring.psi(reference, _series([0] * 500 + [2] * 500)) > 1.0


def test_a_column_with_many_values_still_uses_quantile_bins():
    reference = _series(RNG.normal(size=3000))

    # a value far outside the reference range falls in the top bin, which is allowed to grow
    assert monitoring.psi(reference, _series(RNG.normal(size=3000) + 2)) > 0.25


def test_a_column_with_several_values_is_compared_value_by_value_too():
    # regression test: whole-number columns such as order counts have 3 to 5 values; with decile
    # edges a new value 5 shares a bin with the 4s and a shift from 4 to 5 goes unseen
    reference = _series(np.repeat([0, 1, 2, 3, 4], 200))
    shifted = _series(np.repeat([0, 1, 2, 3, 5], 200))

    assert monitoring.psi(reference, shifted) > 1.0
    assert monitoring.psi(reference, _series(np.repeat([0, 1, 2, 3, 4], 180))) < 0.01
