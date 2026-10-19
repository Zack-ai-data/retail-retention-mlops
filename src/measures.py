"""Metrics for churn scores and spend forecasts. Pure functions, no MLflow or database."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_gamma_deviance,
    mean_poisson_deviance,
    roc_auc_score,
)

from src import (
    features,
)


def top_fraction_total(
    values: npt.ArrayLike, score: npt.ArrayLike, fraction: float
) -> tuple[float, int]:
    """Sum of values over the highest-scored fraction of customers, and how many that is.

    Scores often tie (recency is a whole number of days). A tie that straddles the cut-off is
    split evenly, which is the expected result of breaking it at random, so the order of the
    rows cannot change the answer.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(score, dtype=float)
    k = max(1, round(len(v) * fraction))
    cutoff = np.sort(s)[::-1][k - 1]
    above = s > cutoff
    tied = s == cutoff
    total = v[above].sum() + v[tied].mean() * (k - int(above.sum()))
    return float(total), k


def recall_at_top_fraction(
    y_true: npt.ArrayLike, score: npt.ArrayLike, fraction: float = 0.2
) -> float:
    """Share of all positives found in the highest-scored fraction of customers."""
    found, _ = top_fraction_total(y_true, score, fraction)
    total = float(np.asarray(y_true, dtype=float).sum())
    return found / total if total else float("nan")


def precision_at_top_fraction(
    y_true: npt.ArrayLike, score: npt.ArrayLike, fraction: float = 0.2
) -> float:
    """Share of the highest-scored fraction of customers who are positive (a campaign hit rate)."""
    found, k = top_fraction_total(y_true, score, fraction)
    return found / k


def expected_calibration_error(y_true: npt.ArrayLike, prob: npt.ArrayLike, bins: int = 10) -> float:
    """Average gap between predicted probability and observed rate, over equal-size bins."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(prob, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((grouped["n"] * (grouped["y"] - grouped["p"]).abs()).sum() / len(frame))


def ranking_metrics(y_true: npt.ArrayLike, score: npt.ArrayLike, prefix: str) -> dict[str, float]:
    """Metrics for any churn score (higher = more likely to churn), probability or not."""
    return {
        f"{prefix}roc_auc": float(roc_auc_score(y_true, score)),
        f"{prefix}pr_auc": float(average_precision_score(y_true, score)),
        f"{prefix}recall_at_top20pct": recall_at_top_fraction(y_true, score, 0.2),
        f"{prefix}precision_at_top20pct": precision_at_top_fraction(y_true, score, 0.2),
    }


def churn_metrics(
    y_true: npt.ArrayLike, prob: npt.ArrayLike, prefix: str = "test_"
) -> dict[str, float]:
    """Ranking metrics plus probability quality (Brier, log loss, calibration error)."""
    p = np.asarray(prob, dtype=float)
    return {
        **ranking_metrics(y_true, p, prefix),
        f"{prefix}brier": float(brier_score_loss(y_true, p)),
        f"{prefix}log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        f"{prefix}ece": expected_calibration_error(y_true, p),
    }


# Recency bands for reporting how a churn score does within each part of the actionable population
RECENCY_BANDS = {"le90": (0, 90), "91_180": (90, features.ACTIONABLE_RECENCY_DAYS)}


def recency_band_metrics(
    y_true: npt.ArrayLike, score: npt.ArrayLike, recency_days: npt.ArrayLike, prefix: str
) -> dict[str, float]:
    """Customers, churn rate and PR-AUC inside each recency band.

    Overall figures mix easy and hard customers; this shows where a score actually helps.
    PR-AUC is left out for a band where every customer has the same label.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    r = np.asarray(recency_days, dtype=float)
    out: dict[str, float] = {}
    for band, (low, high) in RECENCY_BANDS.items():
        inside = (r > low) & (r <= high) if low else r <= high
        out[f"{prefix}band_{band}_n"] = float(inside.sum())
        if not inside.any():
            continue
        out[f"{prefix}band_{band}_base_rate"] = float(y[inside].mean())
        if 0 < y[inside].sum() < inside.sum():
            out[f"{prefix}band_{band}_pr_auc"] = float(
                average_precision_score(y[inside], s[inside])
            )
    return out


def top_fraction_capture(
    actual: npt.ArrayLike, predicted: npt.ArrayLike, fraction: float = 0.1
) -> float:
    """Share of total actual spend made by the customers with the highest predicted spend."""
    captured, _ = top_fraction_total(actual, predicted, fraction)
    total = float(np.asarray(actual, dtype=float).sum())
    return captured / total if total else float("nan")


def value_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, prefix: str = "test_"
) -> dict[str, float]:
    """Spend prediction quality on the original (currency) scale.

    The model's job is the *expected* spend per customer, so RMSE (which rewards getting
    the mean right) is the main metric. MAE is reported too, but note that with many
    zero-spend customers it is minimised by predicting zero, so it is not used to choose models.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = predicted - actual
    rho = spearmanr(actual, predicted).statistic if np.ptp(predicted) > 0 else float("nan")
    return {
        f"{prefix}rmse": float(np.sqrt((errors**2).mean())),
        f"{prefix}mae": float(np.abs(errors).mean()),
        f"{prefix}wape": float(np.abs(errors).sum() / actual.sum()),
        f"{prefix}spearman": float(rho),
        f"{prefix}top_decile_capture": top_fraction_capture(actual, predicted, 0.1),
        f"{prefix}bias_pct": float(100 * errors.sum() / actual.sum()),  # total predicted vs actual
    }


def poisson_deviance(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean Poisson deviance: a scoring rule that rewards predicting the mean of skewed spend."""
    return float(
        mean_poisson_deviance(y_true, np.clip(np.asarray(y_pred, dtype=float), 1e-3, None))
    )


def gamma_deviance(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean gamma deviance: a scoring rule for positive, right-skewed amounts (spend of buyers)."""
    return float(mean_gamma_deviance(y_true, np.clip(np.asarray(y_pred, dtype=float), 1e-3, None)))
