"""Model selection, grid search and segment naming: the behaviour behind the numbers in MLflow.

Several of these tests exist because a one-off mutation-testing run on the core logic showed
that earlier tests would not have noticed a wrong validation fold, boundary or ranking.
"""

import numpy as np
import pandas as pd
import pytest

from src import features, model_search, segmentation

FITTED_ON: list[int] = []  # how many rows each fit saw, in order


class _Constant:
    """A model that records how much data it was fitted on and predicts zero."""

    def fit(self, X, y):
        FITTED_ON.append(len(X))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _select(candidates, score_fn, higher_is_better=True):
    dates = pd.Series(["d1"] * 4 + ["d2"] * 3 + ["d3"] * 2)
    X = pd.DataFrame({"a": np.arange(9.0)})
    y = np.arange(9)
    return model_search.select_model(
        candidates,
        X,
        y,
        dates,
        score_fn=score_fn,
        predict_fn=lambda m, data: m.predict(data),
        higher_is_better=higher_is_better,
    )


# ------------------------------------------------------------ select_model


def test_each_fold_trains_on_earlier_snapshots_and_validates_on_the_next_one_only():
    FITTED_ON.clear()
    validated = []

    def score_fn(y_true, y_pred):
        validated.append(list(y_true))
        return 0.5

    _select([model_search.Candidate("const", lambda p: _Constant(), {})], score_fn)

    assert FITTED_ON == [4, 7]  # fold 1: d1; fold 2: d1 + d2
    assert validated == [[4, 5, 6], [7, 8]]  # fold 1 validates on d2 only, fold 2 on d3 only


def test_selection_reports_the_winning_parameters_score_and_every_candidate():
    scripted = iter([0.1234567, 0.9, 0.5, 0.5])  # candidate a: two folds, then candidate b

    candidates = [
        model_search.Candidate("a", lambda p: _Constant(), {"p": [1]}),
        model_search.Candidate("b", lambda p: _Constant(), {"p": [2]}),
    ]
    selection = _select(candidates, lambda y_true, y_pred: next(scripted))

    assert selection.algorithm == "a"
    assert selection.params == {"p": 1}
    assert selection.cv_score == pytest.approx((0.1234567 + 0.9) / 2)
    results = selection.results
    assert list(results.columns) == ["algorithm", "params", "cv_score", "fold_scores"]
    assert list(results.index) == [0, 1]  # a clean 0..n-1 index, not an extra column
    assert results["algorithm"].tolist() == ["a", "b"]
    assert results.loc[0, "params"] == '{"p": 1}'
    assert results.loc[0, "fold_scores"] == "[0.12346, 0.9]"  # rounded to five decimals


def test_selection_picks_the_lowest_score_when_lower_is_better():
    scripted = iter([0.3, 0.3, 0.2, 0.2])

    candidates = [
        model_search.Candidate("a", lambda p: _Constant(), {}),
        model_search.Candidate("b", lambda p: _Constant(), {}),
    ]
    selection = _select(candidates, lambda y_true, y_pred: next(scripted), higher_is_better=False)

    assert selection.algorithm == "b"
    assert selection.cv_score == pytest.approx(0.2)


def test_selection_needs_at_least_two_snapshots_with_a_clear_message():
    X = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ValueError) as caught:
        model_search.select_model(
            [model_search.Candidate("c", lambda p: _Constant(), {})],
            X,
            np.array([0, 1]),
            pd.Series(["d1", "d1"]),
            score_fn=lambda a, b: 0.0,
            predict_fn=lambda m, d: m.predict(d),
            higher_is_better=True,
        )

    # the exact text, so a changed or decorated message is noticed
    assert (
        str(caught.value) == "Need at least two training snapshots for time-based cross-validation"
    )


# ------------------------------------------------------- grid_combinations


def test_grid_combinations_enumerates_every_combination_with_sorted_keys():
    combos = list(model_search.grid_combinations({"b": [1, 2], "a": [10]}))

    assert combos == [{"a": 10, "b": 1}, {"a": 10, "b": 2}]


def test_an_empty_grid_has_one_combination_with_no_parameters():
    assert list(model_search.grid_combinations({})) == [{}]


# ----------------------------------------------------------- name_segments


def test_segment_names_follow_value_rank_and_the_recency_boundaries():
    # median days since last purchase sit exactly on the boundaries: <= 60 Active, <= 180 Cooling
    profile = pd.DataFrame({"median_recency_days": [60.0, 61.0, 180.0, 181.0]}, index=[0, 1, 2, 3])
    # scaled centroids [recency, frequency, monetary]; frequency and monetary deliberately differ,
    # so ranking on frequency + monetary differs from ranking on either one alone
    centroids = np.array(
        [
            [0.0, 3.0, 0.0],  # frequency + monetary = 3.0   -> rank 0 (High)
            [0.0, 0.0, 2.0],  # 2.0                           -> rank 1 (High)
            [0.0, -1.0, -1.0],  # -2.0                        -> rank 3 (Low)
            [0.0, 1.0, 0.5],  # 1.5                           -> rank 2 (Mid)
        ]
    )

    names = segmentation.name_segments(profile, centroids)

    assert names == {
        0: "High value / Active",  # 60 days is still Active
        1: "High value / Cooling",  # 61 days is Cooling
        2: "Low value / Cooling",  # 180 days is still Cooling
        3: "Mid value / Lapsed",  # 181 days is Lapsed
    }


# ------------------------------------------------------- prepare_features


def _feature_frame(**overrides):
    frame = pd.DataFrame({column: [1, 2] for column in features.BASE_FEATURES})  # integers
    for column, values in overrides.items():
        frame[column] = values
    return frame


def test_prepare_features_returns_floats_even_for_integer_input():
    out = features.prepare_features(_feature_frame())

    assert (out.dtypes == "float64").all()


def test_a_ratio_with_a_zero_denominator_becomes_missing_in_both_directions():
    frame = _feature_frame(recency_days=[10, -10], avg_days_between_orders_365d=[0, 0])

    out = features.prepare_features(frame)

    assert out["recency_ratio"].isna().all()  # +inf and -inf are both replaced
