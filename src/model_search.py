"""Candidate model families and the time-ordered cross-validation that chooses between them.

Supervised models are selected by cross-validation over the training snapshots only (each fold
trains on earlier snapshots and validates on the next one), so the test snapshot never takes
part in choosing a model or its hyperparameters.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import (
    config,
    transforms,
)

Params = dict[str, Any]  # hyperparameters of one candidate model


def linear_preprocessing() -> list[Any]:
    return [
        SimpleImputer(strategy="median"),
        FunctionTransformer(transforms.signed_log1p),
        StandardScaler(),
    ]


@dataclass(frozen=True)
class Candidate:
    """One algorithm family with the hyperparameter grid to search."""

    name: str
    factory: Callable[[Params], Any]
    grid: dict[str, list[Any]]


CHURN_CANDIDATES = [
    Candidate(
        "hist_gradient_boosting",
        lambda p: HistGradientBoostingClassifier(
            early_stopping=False, random_state=config.RANDOM_STATE, **p
        ),
        {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [8, 31],
            "max_iter": [100, 300],
            "l2_regularization": [0.0, 1.0],
        },
    ),
    Candidate(
        "logistic_regression",
        lambda p: make_pipeline(*linear_preprocessing(), LogisticRegression(max_iter=1000, **p)),
        {"C": [0.1, 1.0, 10.0]},
    ),
]

# Spend is zero-heavy and skewed, so both candidates use a Poisson (log-link) loss, which
# estimates the mean directly. Training on log(spend) was tried first and underestimated
# total spend by ~78% in cross-validation.
VALUE_CANDIDATES = [
    Candidate(
        "hist_gradient_boosting_poisson",
        lambda p: HistGradientBoostingRegressor(
            loss="poisson", early_stopping=False, random_state=config.RANDOM_STATE, **p
        ),
        {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [8, 31],
            "max_iter": [100, 300],
            "l2_regularization": [0.0, 1.0],
        },
    ),
    Candidate(
        "poisson_glm",
        lambda p: make_pipeline(*linear_preprocessing(), PoissonRegressor(max_iter=1000, **p)),
        {"alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
]

# Second stage of the value model: what a customer spends *given that they buy*. Only buyers are
# used, so the target is strictly positive and a gamma (log-link) loss applies.
SPEND_CANDIDATES = [
    Candidate(
        "hist_gradient_boosting_gamma",
        lambda p: HistGradientBoostingRegressor(
            loss="gamma", early_stopping=False, random_state=config.RANDOM_STATE, **p
        ),
        {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [8, 31],
            "max_iter": [100, 300],
            "l2_regularization": [0.0, 1.0],
        },
    ),
    Candidate(
        "gamma_glm",
        lambda p: make_pipeline(*linear_preprocessing(), GammaRegressor(max_iter=1000, **p)),
        {"alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
]


def grid_combinations(grid: dict[str, list[Any]]) -> Iterator[Params]:
    keys = sorted(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def expanding_window_folds(snapshot_dates: Sequence[Any]) -> list[tuple[list[Any], Any]]:
    """Time-ordered folds: train on all earlier snapshots, validate on the next one."""
    dates = sorted(set(snapshot_dates))
    return [(dates[:i], dates[i]) for i in range(1, len(dates))]


@dataclass(frozen=True)
class Selection:
    algorithm: str
    params: Params
    cv_score: float
    results: pd.DataFrame  # every algorithm/parameter combination with its CV score


def select_model(
    candidates: Sequence[Candidate],
    X: pd.DataFrame,
    y: np.ndarray,
    snapshot_dates: pd.Series,
    score_fn: Callable[[Any, Any], float],
    predict_fn: Callable[[Any, pd.DataFrame], Any],
    higher_is_better: bool,
) -> Selection:
    """Pick the best algorithm and hyperparameters by expanding-window cross-validation."""
    dates = pd.Series(snapshot_dates).reset_index(drop=True)
    folds = expanding_window_folds(dates)
    if not folds:
        raise ValueError("Need at least two training snapshots for time-based cross-validation")

    rows = []
    for candidate in candidates:
        for params in grid_combinations(candidate.grid):
            fold_scores = []
            for train_dates, validation_date in folds:
                train_mask = dates.isin(train_dates).to_numpy()
                validation_mask = (dates == validation_date).to_numpy()
                model = candidate.factory(params).fit(X[train_mask], y[train_mask])
                fold_scores.append(
                    score_fn(y[validation_mask], predict_fn(model, X[validation_mask]))
                )
            rows.append(
                {
                    "algorithm": candidate.name,
                    "params": json.dumps(params, sort_keys=True),
                    "cv_score": float(np.mean(fold_scores)),
                    "fold_scores": json.dumps([round(float(s), 5) for s in fold_scores]),
                }
            )

    results = (
        pd.DataFrame(rows)
        .sort_values("cv_score", ascending=not higher_is_better, kind="stable")
        .reset_index(drop=True)
    )
    best = results.iloc[0]
    return Selection(
        best["algorithm"], json.loads(best["params"]), float(best["cv_score"]), results
    )


def candidate_factory(candidates: Sequence[Candidate], algorithm: str) -> Callable[[Params], Any]:
    return next(c.factory for c in candidates if c.name == algorithm)


def out_of_fold_predictions(
    factory: Callable[[Params], Any],
    params: Params,
    X: pd.DataFrame,
    y: np.ndarray,
    snapshot_dates: pd.Series,
    predict_fn: Callable[[Any, pd.DataFrame], Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions for every snapshot after the first, each from a model that saw only earlier ones.

    Returns the true labels, the predictions and the snapshot each belongs to, in the same
    order. Unlike predictions on the training data, these are honest: the model never saw the
    customers it is scoring.
    """
    dates = pd.Series(snapshot_dates).reset_index(drop=True)
    labels, predictions, snapshots = [], [], []
    for train_dates, validation_date in expanding_window_folds(dates):
        train_mask = dates.isin(train_dates).to_numpy()
        validation_mask = (dates == validation_date).to_numpy()
        fitted = factory(params).fit(X[train_mask], y[train_mask])
        labels.append(y[validation_mask])
        predictions.append(np.asarray(predict_fn(fitted, X[validation_mask])))
        snapshots.append(np.full(int(validation_mask.sum()), validation_date, dtype=object))
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(snapshots)
