"""Probability calibration for the churn model.

Customers are ranked by expected value at risk = P(churn) x predicted spend x margin - cost, so
the churn score has to be a probability, not just an ordering. A Platt (sigmoid) calibrator
maps the model's probabilities onto observed churn rates. It is fitted on out-of-fold
predictions (each made by a model that had only seen earlier snapshots), never on the data the
model was trained on or the test snapshot. The mapping is increasing, so rankings are unchanged.

Calibration is not always an improvement: churn depends on the season, so a mapping learned on
earlier snapshots can fit a later one worse. fit_if_it_helps() therefore checks it on the latest
out-of-fold snapshot first, and falls back to the identity mapping when it does not help.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

_EPSILON = 1e-6


def _logit(prob: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(logit(np.clip(np.asarray(prob, dtype=float), _EPSILON, 1 - _EPSILON)))


def fit_platt(y_true: npt.ArrayLike, prob: npt.ArrayLike) -> tuple[float, float]:
    """Slope and intercept of the sigmoid that maps probabilities to observed rates."""
    y = np.asarray(y_true, dtype=int)
    if y.min() == y.max():
        raise ValueError("Calibration needs both churned and retained customers")
    # a very large C leaves the two-parameter sigmoid effectively unpenalised
    fitted = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(prob).reshape(-1, 1), y)
    return float(fitted.coef_[0, 0]), float(fitted.intercept_[0])


def apply_platt(prob: npt.ArrayLike, slope: float, intercept: float) -> npt.NDArray[np.float64]:
    return np.asarray(expit(slope * _logit(prob) + intercept))


def fit_if_it_helps(
    y_true: npt.ArrayLike, prob: npt.ArrayLike, groups: npt.ArrayLike
) -> tuple[float, float, bool]:
    """Fit a Platt mapping, but only keep it if it improves log loss on the latest group.

    groups says which snapshot each out-of-fold prediction belongs to. The mapping is fitted on
    all earlier snapshots and tested on the latest one; if log loss does not improve, the
    identity mapping (slope 1, intercept 0) is returned. The mapping that is returned when it
    helps is refitted on every snapshot. Returns slope, intercept and whether it was applied.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    g = np.asarray(groups, dtype=object)
    ordered = sorted(set(g.tolist()))
    if len(ordered) < 2:
        return 1.0, 0.0, False
    latest = g == ordered[-1]
    try:
        slope, intercept = fit_platt(y[~latest], p[~latest])
    except ValueError:
        return 1.0, 0.0, False
    raw_loss = log_loss(y[latest], np.clip(p[latest], _EPSILON, 1 - _EPSILON), labels=[0, 1])
    calibrated_loss = log_loss(y[latest], apply_platt(p[latest], slope, intercept), labels=[0, 1])
    if calibrated_loss >= raw_loss:
        return 1.0, 0.0, False
    return (*fit_platt(y, p), True)


class PlattCalibratedClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """A fitted classifier whose probabilities are passed through a fitted sigmoid."""

    def __init__(self, base: Any, slope: float, intercept: float) -> None:
        self.base = base
        self.slope = slope
        self.intercept = intercept
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        calibrated = apply_platt(self.base.predict_proba(X)[:, 1], self.slope, self.intercept)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int64]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
