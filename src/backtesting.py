"""Evaluation over several time origins, and the statistical baseline it is compared with."""

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
)

from src import (
    baselines,
    evaluation,
    features,
    measures,
    model_search,
)

log = logging.getLogger(__name__)


MIN_BACKTEST_TRAIN_SNAPSHOTS = 3


def churn_backtest(
    df: pd.DataFrame, factory: Callable[[model_search.Params], Any], params: model_search.Params
) -> pd.DataFrame:
    """Evaluate the chosen churn model at every origin that has enough earlier history.

    For each origin snapshot the model is trained only on snapshots whose label window had ended
    by then (a training snapshot whose window overlaps the origin is dropped, "purging"), and
    scored against ranking customers by recency. The last origin is the test snapshot. The
    algorithm and hyperparameters were chosen on the training snapshots, so the earlier origins
    show how stable the result is over time, not an independent estimate for each.
    """
    frame = df.reset_index(drop=True)
    X = features.prepare_features(frame)
    y = frame[features.CHURN_TARGET].to_numpy(dtype=int)
    dates = sorted(frame["snapshot_date"].unique())
    horizon = timedelta(days=features.HORIZON_DAYS)

    rows = []
    for origin in dates:
        usable = [d for d in dates if d + horizon <= origin]
        if len(usable) < MIN_BACKTEST_TRAIN_SNAPSHOTS:
            continue
        train_mask = frame["snapshot_date"].isin(usable).to_numpy()
        origin_mask = (frame["snapshot_date"] == origin).to_numpy()
        fitted = factory(params).fit(X[train_mask], y[train_mask])
        prob = fitted.predict_proba(X[origin_mask])[:, 1]
        recency = X.loc[origin_mask, "recency_days"].to_numpy()
        y_origin = y[origin_mask]
        gain = evaluation.paired_bootstrap(y_origin, prob, recency, average_precision_score)
        rows.append(
            {
                "origin": str(origin),
                "train_snapshots": len(usable),
                "customers": int(origin_mask.sum()),
                "base_rate": float(y_origin.mean()),
                "model_pr_auc": gain.candidate,
                "recency_pr_auc": gain.reference,
                "gain": gain.diff,
                "gain_ci_low": gain.low,
                "gain_ci_high": gain.high,
                "significant": gain.excludes_zero_above(),
                "model_recall_top20": measures.recall_at_top_fraction(y_origin, prob, 0.2),
                "recency_recall_top20": measures.recall_at_top_fraction(y_origin, recency, 0.2),
                "model_precision_top20": measures.precision_at_top_fraction(y_origin, prob, 0.2),
                "recency_precision_top20": measures.precision_at_top_fraction(
                    y_origin, recency, 0.2
                ),
            }
        )
    return pd.DataFrame(rows)


def backtest_summary(table: pd.DataFrame) -> dict[str, float]:
    if table.empty:
        raise ValueError("The backtest has no origins with enough earlier snapshots")
    significant = int(table["significant"].sum())
    return {
        "backtest_origins": float(len(table)),
        "backtest_origins_significant": float(significant),
        "backtest_share_significant": significant / len(table),
    }


def bgnbd_churn_scores(df: pd.DataFrame, test: pd.DataFrame) -> np.ndarray | None:
    """Churn scores (1 - P(alive)) from a BG/NBD model fitted before the test snapshot.

    The model is fitted to every customer's history as of the latest training snapshot, so the
    test snapshot is not seen. Returns None if the fit fails: a baseline that cannot be fitted
    must not stop the real model from being trained.
    """
    train_rows = df[df["split"] == "train"]
    cohort = train_rows[train_rows["snapshot_date"] == train_rows["snapshot_date"].max()]
    try:
        fitted = baselines.BgNbd.fit(
            cohort["frequency"], cohort["recency_days"], cohort["tenure_days"]
        )
    except ValueError as exc:
        log.warning("BG/NBD baseline skipped: %s", exc)
        return None
    alive = fitted.probability_alive(test["frequency"], test["recency_days"], test["tenure_days"])
    return 1.0 - alive
