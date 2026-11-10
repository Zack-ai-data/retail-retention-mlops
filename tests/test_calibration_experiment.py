"""The rolling-recalibration trial: the shift keeps the ranking, each variant moves the average to
what it says it does, nothing is used before its outcomes are known, and the verdict rule is fixed.
"""

from contextlib import contextmanager
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import calibration_experiment as experiment
from src import database, features, model_search, tracking
from src.errors import RegistryError
from tests.test_model import make_panel

HORIZON = timedelta(days=features.HORIZON_DAYS)
FIRST = date(2010, 3, 18)


def _dates(count):
    return [FIRST + i * HORIZON for i in range(count)]  # exactly one outcome window apart


def _panel(dates, seed=0, n=300):
    snapshots = [(str(d), "train") for d in dates[:-1]] + [(str(dates[-1]), "test")]
    return make_panel(seed=seed, n=n, snapshots=snapshots)


def _model():
    # the real pipelines handle missing inputs; a plain logistic regression does not
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())


def _factory(params):
    return _model()


def _labelled(panel):
    return panel[panel["is_actionable"].astype(bool)].reset_index(drop=True)


@pytest.fixture(scope="module")
def trial():
    panel = _labelled(_panel(_dates(7)))
    return panel, experiment.evaluate(panel, _factory, {})


def _row(table, origin, variant):
    (row,) = table[(table["origin"] == str(origin)) & (table["variant"] == variant)].to_dict(
        "records"
    )
    return row


# ------------------------------------------------------------------------- the shift


@pytest.mark.parametrize("target", [0.1, 0.39, 0.5, 0.65, 0.9])
def test_the_shifted_probabilities_have_exactly_the_target_mean(target):
    p = np.random.default_rng(0).beta(2, 3, 2000)

    assert experiment.shift_to_mean(p, target).mean() == pytest.approx(target, abs=1e-9)


def test_the_shift_keeps_the_order_of_the_customers_and_stays_inside_zero_and_one():
    p = np.random.default_rng(1).beta(2, 3, 2000)

    shifted = experiment.shift_to_mean(p, 0.8)

    assert (np.argsort(p, kind="stable") == np.argsort(shifted, kind="stable")).all()
    assert ((shifted > 0) & (shifted < 1)).all()


def test_a_target_equal_to_the_current_mean_changes_nothing():
    p = np.random.default_rng(2).beta(2, 3, 1000)

    assert experiment.shift_to_mean(p, float(p.mean())) == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("target", [0.0, 1.0, -0.2, 1.5])
def test_an_impossible_target_mean_is_refused(target):
    with pytest.raises(ValueError, match="between 0 and 1"):
        experiment.shift_to_mean([0.2, 0.4], target)


def test_probabilities_of_exactly_zero_and_one_do_not_break_the_shift():
    shifted = experiment.shift_to_mean([0.0, 0.0, 1.0, 0.5], 0.5)

    assert np.isfinite(shifted).all() and shifted.mean() == pytest.approx(0.5, abs=1e-9)


# ----------------------------------------------------------------------- the origins


def test_every_origin_with_enough_earlier_history_is_evaluated_and_none_before_it(trial):
    _, table = trial

    # seven snapshots 90 days apart: the first three only train, the last four are origins
    assert list(table["origin"].unique()) == [str(d) for d in _dates(7)[3:]]


def test_the_first_origin_has_no_earlier_gap_so_it_has_no_avg2_and_is_not_comparable(trial):
    _, table = trial
    first = str(_dates(7)[3])

    assert sorted(table[table["origin"] == first]["variant"]) == ["full", "half", "none"]
    assert not table[table["origin"] == first]["comparable"].any()
    later = table[table["origin"] != first]
    assert later["comparable"].all() and set(later["variant"]) == set(experiment.VARIANTS)


def test_the_unshifted_row_matches_an_independent_fit_and_prediction(trial):
    panel, table = trial
    origin = _dates(7)[5]
    usable = [d for d in _dates(7) if d + HORIZON <= origin]
    train = panel[panel["snapshot_date"].isin(usable)]
    here = panel[panel["snapshot_date"] == origin]
    fitted = _model().fit(
        features.prepare_features(train), train[features.CHURN_TARGET].astype(int)
    )
    p = fitted.predict_proba(features.prepare_features(here))[:, 1]

    row = _row(table, origin, "none")

    assert row["mean_p"] == pytest.approx(p.mean())
    assert row["actual"] == pytest.approx(here[features.CHURN_TARGET].mean())
    assert row["gap"] == pytest.approx(p.mean() - here[features.CHURN_TARGET].mean())
    assert row["abs_gap"] == pytest.approx(abs(row["gap"]))


def test_no_variant_changes_the_ranking_so_pr_auc_is_identical(trial):
    _, table = trial

    spread = table.groupby("origin")["pr_auc"].agg(lambda s: float(s.max() - s.min()))

    assert (spread < 1e-12).all()


# ---------------------------------------------------------------- what each variant does


def test_full_moves_the_average_to_the_previous_snapshots_churn_rate(trial):
    panel, table = trial
    dates = _dates(7)
    for origin, previous in zip(dates[4:], dates[3:-1], strict=True):
        previous_rate = panel[panel["snapshot_date"] == previous][features.CHURN_TARGET].mean()

        assert _row(table, origin, "full")["mean_p"] == pytest.approx(previous_rate, abs=1e-6)


def test_half_moves_the_average_halfway_between_the_model_and_that_churn_rate(trial):
    _, table = trial
    for origin in _dates(7)[4:]:
        none, full = _row(table, origin, "none"), _row(table, origin, "full")

        assert _row(table, origin, "half")["mean_p"] == pytest.approx(
            (none["mean_p"] + full["mean_p"]) / 2, abs=1e-6
        )


def test_avg2_subtracts_the_mean_of_the_models_last_two_gaps(trial):
    _, table = trial
    dates = _dates(7)
    gaps = {d: _row(table, d, "none")["gap"] for d in dates[3:]}
    for position, origin in enumerate(dates[4:], start=4):
        recent = [gaps[d] for d in dates[max(3, position - 2) : position]]
        expected = _row(table, origin, "none")["mean_p"] - np.mean(recent)

        assert _row(table, origin, "avg2")["mean_p"] == pytest.approx(
            float(np.clip(expected, 0.01, 0.99)), abs=1e-6
        )


def test_a_target_outside_a_plausible_churn_rate_is_held_inside_it(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "_targets",
        lambda mean_p, previous_rate, known_gaps: {
            "none": mean_p,
            "full": 1.4,  # asked for: more than every customer churning
            "half": -0.2,
            "avg2": 0.5,
        },
    )
    panel = _labelled(_panel(_dates(7), seed=3))

    table = experiment.evaluate(panel, _factory, {})

    assert table[table["variant"] == "full"]["mean_p"].to_numpy() == pytest.approx(
        experiment.MAX_TARGET, abs=1e-6
    )
    assert table[table["variant"] == "half"]["mean_p"].to_numpy() == pytest.approx(
        experiment.MIN_TARGET, abs=1e-6
    )


# ------------------------------------------------------- nothing is used before it is known


def test_a_snapshot_whose_outcome_window_has_not_ended_is_never_used():
    # the last snapshot is only 30 days after the one before it: that one's outcomes are not
    # known yet, so the previous snapshot as far as the origin knows is the one before
    dates = [*_dates(4), _dates(4)[-1] + timedelta(days=30)]
    panel = _labelled(_panel(dates, seed=4))
    table = experiment.evaluate(panel, _factory, {})
    origin = dates[-1]
    known = dates[2]  # +90 days is on or before the origin; dates[3] + 90 days is not
    assert dates[3] + HORIZON > origin >= known + HORIZON

    rate = panel[panel["snapshot_date"] == known][features.CHURN_TARGET].mean()

    assert _row(table, origin, "full")["mean_p"] == pytest.approx(rate, abs=1e-6)
    assert "avg2" not in set(table[table["origin"] == str(origin)]["variant"])  # no gap known yet


# ------------------------------------------------------------------------ the verdict


def _table(rows):
    return pd.DataFrame(
        [{"origin": o, "variant": v, "abs_gap": g, "comparable": c} for o, v, g, c in rows]
    )


def _verdict(none, other, comparable=True):
    rows = [(f"o{i}", "none", g, comparable) for i, g in enumerate(none)]
    rows += [(f"o{i}", "full", g, comparable) for i, g in enumerate(other)]
    return experiment.verdict(_table(rows))["full"]


def test_a_variant_helps_when_its_mean_gap_is_lower_and_no_origin_is_much_worse():
    assert _verdict([0.10, 0.05, 0.09], [0.05, 0.06, 0.04])["helps"] is True


def test_a_lower_mean_is_not_enough_if_one_origin_gets_worse_by_more_than_the_limit():
    result = _verdict([0.02, 0.30, 0.30], [0.08, 0.01, 0.01])  # one origin worse by 0.06

    assert result["mean_abs_gap"] < result["baseline_mean_abs_gap"]
    assert result["worst_origin_worsening"] == pytest.approx(0.06)
    assert result["helps"] is False


def test_a_worsening_of_exactly_the_limit_is_allowed(monkeypatch):
    monkeypatch.setattr(experiment, "MAX_ORIGIN_WORSENING", 0.0625)  # exact in binary: no rounding

    result = _verdict([0.25, 0.5], [0.3125, 0.0])  # the first origin is worse by exactly the limit

    assert result["worst_origin_worsening"] == 0.0625 and result["helps"] is True


def test_a_variant_with_a_higher_mean_gap_does_not_help():
    assert (
        _verdict(
            [0.05, 0.05],
            [
                0.06,
                0.04,
            ][:2],
        )["helps"]
        is False
    )  # equal mean: not lower


def test_origins_that_are_not_comparable_are_left_out_of_the_verdict():
    table = _table(
        [
            ("a", "none", 0.5, False),
            ("a", "full", 0.0, False),  # would look like a win, but this origin is not comparable
            ("b", "none", 0.1, True),
            ("b", "full", 0.2, True),
            ("b", "half", 0.1, True),
            ("b", "avg2", 0.1, True),
        ]
    )

    assert experiment.verdict(table)["full"]["origins"] == 1
    assert experiment.verdict(table)["full"]["helps"] is False


def test_with_no_comparable_origin_nothing_is_said_to_help():
    table = _table([("a", "none", 0.5, False), ("a", "full", 0.0, False)])

    assert experiment.verdict(table)["full"] == {"origins": 0, "helps": False}


def test_the_variants_and_the_limit_are_the_ones_fixed_in_advance():
    assert experiment.VARIANTS == ("none", "full", "half", "avg2")
    assert experiment.MAX_ORIGIN_WORSENING == 0.05


# ------------------------------------------------------- the champion's own settings


def _client(params, run_id="run-1"):
    version = SimpleNamespace(version="5", run_id=run_id)
    return SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: version,
        get_run=lambda rid: SimpleNamespace(data=SimpleNamespace(params=params)),
    )


def test_the_champions_logged_text_parameters_come_back_with_the_types_the_model_expects():
    client = _client(
        {
            "algorithm": "hist_gradient_boosting",
            "hp_learning_rate": "0.05",
            "hp_max_leaf_nodes": "8",
            "hp_max_iter": "100",
            "hp_l2_regularization": "1.0",
        }
    )

    algorithm, params = experiment.champion_settings(client)

    assert algorithm == "hist_gradient_boosting"
    assert params == {
        "learning_rate": 0.05,
        "max_leaf_nodes": 8,
        "max_iter": 100,
        "l2_regularization": 1.0,
    }
    assert isinstance(params["max_leaf_nodes"], int)
    assert isinstance(params["l2_regularization"], float)
    model_search.candidate_factory(model_search.CHURN_CANDIDATES, algorithm)(params)  # builds


def test_a_champion_without_a_training_run_is_explained_not_a_crash():
    with pytest.raises(RegistryError, match="no training run"):
        experiment.champion_settings(_client({}, run_id=None))


# ------------------------------------------------------------------- the whole thing


def test_run_evaluates_the_warehouse_with_the_champions_settings(monkeypatch):
    panel = _panel(_dates(7), seed=5)
    monkeypatch.setattr(tracking, "configure", lambda: None)
    monkeypatch.setattr(
        experiment, "champion_settings", lambda client: ("logistic_regression", {"C": 1.0})
    )

    @contextmanager
    def connection():
        yield None

    monkeypatch.setattr(database, "get_connection", connection)
    monkeypatch.setattr(features, "load_snapshots", lambda conn: panel)

    table, outcome = experiment.run()

    assert set(table["variant"]) == set(experiment.VARIANTS)
    assert set(outcome) == {"full", "half", "avg2"}


def test_the_command_prints_the_table_and_the_verdict_and_exits_cleanly(monkeypatch, capsys):
    panel = _labelled(_panel(_dates(7), seed=6))
    table = experiment.evaluate(panel, _factory, {})
    monkeypatch.setattr(experiment, "run", lambda: (table, experiment.verdict(table)))

    assert experiment.main() == 0

    shown = capsys.readouterr().out
    assert "verdict" in shown and "full:" in shown and "half:" in shown and "avg2:" in shown
    assert "PR-AUC is the same for every variant" in shown
