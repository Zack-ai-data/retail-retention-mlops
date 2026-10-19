"""Platt calibration: probabilities that match observed rates, with the ranking untouched."""

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src import calibration
from src.measures import expected_calibration_error


def _overconfident(n=4000, seed=0):
    """True churn probability p, but the model reports a much more extreme value."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.1, 0.9, n)
    y = (rng.random(n) < true_p).astype(int)
    reported = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))  # logit scaled by 3
    return y, reported


def test_platt_scaling_pulls_overconfident_probabilities_back_to_observed_rates():
    y, reported = _overconfident()

    slope, intercept = calibration.fit_platt(y, reported)

    assert slope == pytest.approx(1 / 3, abs=0.06)  # undoes the x3 stretch
    assert abs(intercept) < 0.15
    fixed = 1 / (1 + np.exp(-(slope * np.log(reported / (1 - reported)) + intercept)))
    assert expected_calibration_error(y, fixed) < expected_calibration_error(y, reported) / 2


def test_calibration_needs_both_classes():
    with pytest.raises(ValueError, match="both churned and retained"):
        calibration.fit_platt([1, 1, 1], [0.2, 0.5, 0.9])


def test_probabilities_at_the_extremes_do_not_break_the_logit():
    slope, intercept = calibration.fit_platt([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])

    assert np.isfinite(slope) and np.isfinite(intercept)


class _FixedProbabilities:
    def __init__(self, prob):
        self.prob = np.asarray(prob)

    def predict_proba(self, X):
        return np.column_stack([1 - self.prob, self.prob])


def test_the_calibrated_classifier_keeps_the_ranking_and_returns_valid_probabilities():
    raw = np.linspace(0.02, 0.98, 50)
    model = calibration.PlattCalibratedClassifier(_FixedProbabilities(raw), 0.4, -0.1)

    proba = model.predict_proba(pd.DataFrame({"x": range(50)}))

    assert proba.shape == (50, 2)
    assert proba.sum(axis=1) == pytest.approx(np.ones(50))
    assert ((proba >= 0) & (proba <= 1)).all()
    assert spearmanr(proba[:, 1], raw).statistic == pytest.approx(1.0)


def test_predict_uses_a_half_threshold_on_the_calibrated_probability():
    raw = np.array([0.1, 0.4, 0.6, 0.9])
    model = calibration.PlattCalibratedClassifier(_FixedProbabilities(raw), 1.0, 0.0)

    assert model.predict(pd.DataFrame({"x": range(4)})).tolist() == [0, 0, 1, 1]


def test_the_calibrated_classifier_survives_pickling_and_cloning():
    base = LogisticRegression().fit(np.arange(20.0).reshape(-1, 1), np.arange(20) % 2)
    model = calibration.PlattCalibratedClassifier(base, 0.8, 0.1)
    X = np.arange(5.0).reshape(-1, 1)

    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert restored.predict_proba(X) == pytest.approx(model.predict_proba(X))
    assert clone(model).slope == 0.8
    assert list(model.classes_) == [0, 1]


# ------------------------------------------------------------ fit_if_it_helps


def _three_snapshots(seed=0, distort_last=True):
    """Out-of-fold predictions for three snapshots; the model is overconfident in each."""
    ys, ps, gs = [], [], []
    for index, group in enumerate(["2011-01", "2011-04", "2011-07"]):
        y, reported = _overconfident(n=2000, seed=seed + index)
        if group == "2011-07" and not distort_last:
            true_p = 1 / (1 + np.exp(-np.log(reported / (1 - reported)) / 3))
            reported = true_p  # the latest snapshot is already well calibrated
            y = (np.random.default_rng(50).random(len(reported)) < reported).astype(int)
        ys.append(y)
        ps.append(reported)
        gs.append(np.full(len(y), group, dtype=object))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)


def test_calibration_is_kept_when_it_improves_the_latest_snapshot():
    y, prob, groups = _three_snapshots()

    slope, _, applied = calibration.fit_if_it_helps(y, prob, groups)

    assert applied is True
    assert slope == pytest.approx(1 / 3, abs=0.06)


def test_calibration_is_dropped_when_it_makes_the_latest_snapshot_worse():
    # earlier snapshots are overconfident but the latest is already calibrated: a mapping
    # learned from the past would damage it, so the identity mapping is returned
    y, prob, groups = _three_snapshots(distort_last=False)

    assert calibration.fit_if_it_helps(y, prob, groups) == (1.0, 0.0, False)


def test_with_a_single_snapshot_there_is_nothing_to_check_it_on():
    y, prob = _overconfident()

    assert calibration.fit_if_it_helps(y, prob, np.full(len(y), "only", dtype=object)) == (
        1.0,
        0.0,
        False,
    )


def test_when_earlier_snapshots_have_one_class_calibration_is_skipped_not_fatal():
    y = np.array([1, 1, 1, 0, 1, 0])
    prob = np.array([0.6, 0.7, 0.8, 0.3, 0.7, 0.4])
    groups = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)

    assert calibration.fit_if_it_helps(y, prob, groups) == (1.0, 0.0, False)


def test_apply_platt_with_slope_one_and_no_intercept_is_the_identity():
    prob = np.array([0.05, 0.3, 0.5, 0.9])

    assert calibration.apply_platt(prob, 1.0, 0.0) == pytest.approx(prob)
