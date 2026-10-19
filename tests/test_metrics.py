"""The metric functions, checked against hand-computed values and scikit-learn.

Several of these tests exist because a one-off mutation-testing run on the core logic showed
that earlier tests would not have noticed a wrong formula, default or tie-break.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_poisson_deviance,
    roc_auc_score,
)

from src import measures, reporting

# ----------------------------------------------------------- ranking metrics


def test_ranking_metrics_known_example():
    y, score = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]  # the scikit-learn documentation example

    out = measures.ranking_metrics(y, score, "x_")

    assert set(out) == {
        "x_roc_auc",
        "x_pr_auc",
        "x_recall_at_top20pct",
        "x_precision_at_top20pct",
    }
    assert out["x_roc_auc"] == pytest.approx(0.75)
    assert out["x_pr_auc"] == pytest.approx(5 / 6)
    assert out["x_recall_at_top20pct"] == pytest.approx(0.5)  # top 1 of 4: the 0.8 positive


def test_recall_uses_a_20_percent_default():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = list(range(10, 0, -1))  # the first two customers are the highest scored

    assert measures.recall_at_top_fraction(y, score) == pytest.approx(0.5)  # 1 of 2 positives


def test_recall_always_takes_at_least_one_customer():
    y, score = [0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]

    # a tiny fraction still takes the single top customer (a positive), never two
    assert measures.recall_at_top_fraction(y, score, fraction=0.01) == pytest.approx(0.5)


def test_recall_is_undefined_without_positives():
    assert math.isnan(measures.recall_at_top_fraction([0, 0, 0], [3, 2, 1]))


def test_recall_splits_a_tie_evenly_so_row_order_cannot_matter():
    y = np.zeros(100)
    y[0] = y[99] = 1
    score = np.ones(100)  # every customer tied

    # taking 10 of 100 tied customers at random finds 10% of the positives on average; the old
    # row-order rule gave 0.5 here because a positive happened to be first (F13 in the review)
    assert measures.recall_at_top_fraction(y, score, 0.1) == pytest.approx(0.1)
    assert measures.recall_at_top_fraction(y[::-1], score, 0.1) == pytest.approx(0.1)


# ------------------------------------------------------------ churn metrics


def test_churn_metrics_match_scikit_learn_and_use_the_default_prefix():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 300)
    y = (rng.random(300) < p).astype(int)

    out = measures.churn_metrics(y, p)

    assert set(out) == {
        "test_roc_auc",
        "test_pr_auc",
        "test_recall_at_top20pct",
        "test_precision_at_top20pct",
        "test_brier",
        "test_log_loss",
        "test_ece",
    }
    assert out["test_roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert out["test_pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert out["test_brier"] == pytest.approx(brier_score_loss(y, p))
    assert out["test_log_loss"] == pytest.approx(log_loss(y, p))
    assert out["test_ece"] == pytest.approx(measures.expected_calibration_error(y, p))


def test_churn_metrics_use_a_custom_prefix():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 100)
    y = (rng.random(100) < p).astype(int)

    assert all(key.startswith("val_") for key in measures.churn_metrics(y, p, "val_"))


def test_log_loss_stays_finite_for_certain_wrong_predictions():
    y = [1] * 5 + [0] * 5
    p = [0.0] * 5 + [1.0] * 5  # every prediction is certain and wrong

    out = measures.churn_metrics(y, p)

    assert out["test_log_loss"] == pytest.approx(-math.log(1e-6))  # probabilities clipped at 1e-6


# ------------------------------------------------------------- calibration


def test_expected_calibration_error_known_value():
    # two bins: predictions of about 0.15 that are always 0, and of about 0.85 that are always 1
    out = measures.expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], bins=2)

    assert out == pytest.approx(0.15)


def test_expected_calibration_error_defaults_to_ten_bins():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 200)
    y = (rng.random(200) < p).astype(int)

    assert measures.expected_calibration_error(y, p) == measures.expected_calibration_error(
        y, p, bins=10
    )
    assert measures.expected_calibration_error(y, p) != measures.expected_calibration_error(
        y, p, bins=11
    )


# ---------------------------------------------------------- value metrics


def test_value_metrics_known_example():
    actual = [0, 10, 20, 30]
    predicted = [0, 20, 10, 40]  # errors 0, +10, -10, +10

    out = measures.value_metrics(actual, predicted)

    assert set(out) == {
        "test_rmse",
        "test_mae",
        "test_wape",
        "test_spearman",
        "test_top_decile_capture",
        "test_bias_pct",
    }
    assert out["test_rmse"] == pytest.approx(math.sqrt(75))
    assert out["test_mae"] == pytest.approx(7.5)
    assert out["test_wape"] == pytest.approx(0.5)  # 30 absolute error / 60 actual
    assert out["test_spearman"] == pytest.approx(0.8)
    assert out["test_top_decile_capture"] == pytest.approx(0.5)  # top 1 predicted holds 30 of 60
    assert out["test_bias_pct"] == pytest.approx(100 * 10 / 60)


def test_value_metrics_rank_correlation_does_not_need_a_wide_prediction_range():
    out = measures.value_metrics([0, 10, 20, 30], [0, 0.2, 0.1, 0.4])  # predictions all within 0.4

    assert out["test_spearman"] == pytest.approx(0.8)


def test_value_metrics_use_a_custom_prefix():
    out = measures.value_metrics([1, 2, 3], [1, 2, 4], prefix="baseline_")

    assert all(key.startswith("baseline_") for key in out)


# ------------------------------------------------------ top-fraction capture


def test_top_fraction_capture_uses_a_10_percent_default():
    actual = np.arange(20, 0, -1.0)  # 20, 19, ... 1: the best customers come first
    predicted = actual.copy()

    assert measures.top_fraction_capture(actual, predicted) == pytest.approx((20 + 19) / 210)


def test_top_fraction_capture_is_undefined_without_spend():
    assert math.isnan(measures.top_fraction_capture(np.zeros(10), np.arange(10.0)))


def test_top_fraction_capture_splits_a_tie_evenly():
    actual = np.arange(100.0)
    predicted = np.ones(100)  # every customer tied

    # a random 10% of customers hold 10% of the spend on average, whatever the row order
    assert measures.top_fraction_capture(actual, predicted, 0.1) == pytest.approx(0.1)
    assert measures.top_fraction_capture(actual[::-1], predicted, 0.1) == pytest.approx(0.1)


# --------------------------------------------------------- Poisson deviance


def test_poisson_deviance_matches_scikit_learn():
    y, p = [0, 1, 2, 5], [0.5, 1.0, 3.0, 4.0]

    assert measures.poisson_deviance(y, p) == pytest.approx(mean_poisson_deviance(y, p))


def test_poisson_deviance_clips_predictions_of_zero():
    # a prediction of exactly zero would make the deviance infinite; it is clipped to 1e-3
    assert measures.poisson_deviance([1.0], [0.0]) == pytest.approx(
        mean_poisson_deviance([1.0], [1e-3])
    )


# --------------------------------------------------------- data fingerprint


def test_data_fingerprint_is_stable_and_sensitive_to_values():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})

    same = reporting.data_fingerprint(frame.copy())
    changed = frame.copy()
    changed.loc[1, "a"] = 2.5

    assert reporting.data_fingerprint(frame) == same
    assert reporting.data_fingerprint(changed) != same
    assert len(same) == 16
    assert int(same, 16) >= 0  # hexadecimal


def test_data_fingerprint_ignores_the_row_labels():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    relabelled = frame.set_axis([10, 20, 30])

    assert reporting.data_fingerprint(frame) == reporting.data_fingerprint(relabelled)


# ------------------------------------------------- ties between different values


def _expected_top_sum(values, score, k):
    """Reference answer: average the top-k sum over many random orders of the tied customers."""
    rng = np.random.default_rng(0)
    sums = []
    for _ in range(4000):
        order = rng.permutation(len(score))
        top = order[np.argsort(-score[order], kind="stable")[:k]]
        sums.append(values[top].sum())
    return float(np.mean(sums))


def test_recall_with_many_tied_scores_equals_the_average_over_random_tie_breaks():
    rng = np.random.default_rng(3)
    score = rng.integers(0, 4, 300)  # only four distinct scores: heavy ties
    y = (rng.random(300) < 0.5).astype(float)

    expected = _expected_top_sum(y, score, 30) / y.sum()

    assert measures.recall_at_top_fraction(y, score, 0.1) == pytest.approx(expected, abs=0.005)


def test_top_fraction_capture_with_many_tied_scores_equals_the_average_over_random_tie_breaks():
    rng = np.random.default_rng(4)
    predicted = rng.integers(0, 4, 300)
    actual = rng.uniform(0, 100, 300)

    expected = _expected_top_sum(actual, predicted, 30) / actual.sum()

    assert measures.top_fraction_capture(actual, predicted, 0.1) == pytest.approx(
        expected, abs=0.005
    )


def test_row_order_never_changes_a_top_fraction_metric():
    rng = np.random.default_rng(5)
    score = rng.integers(0, 5, 200)
    y = (rng.random(200) < 0.4).astype(float)
    shuffled = rng.permutation(200)

    assert measures.recall_at_top_fraction(y, score, 0.2) == pytest.approx(
        measures.recall_at_top_fraction(y[shuffled], score[shuffled], 0.2)
    )
    assert measures.precision_at_top_fraction(y, score, 0.2) == pytest.approx(
        measures.precision_at_top_fraction(y[shuffled], score[shuffled], 0.2)
    )


def test_precision_at_top_fraction_is_the_hit_rate_of_the_contacted_customers():
    y = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    assert measures.precision_at_top_fraction(y, score, 0.2) == pytest.approx(1.0)  # top 2: both
    assert measures.precision_at_top_fraction(y, score, 0.5) == pytest.approx(0.6)  # top 5: 3 of 5
