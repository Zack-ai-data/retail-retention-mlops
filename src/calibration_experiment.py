"""A trial of one cheap way to keep the churn probabilities' level right: rolling recalibration.

The churn model ranks customers well but its average probability can sit well away from the share
that really churns (the churn rate swings from 39% to 65% with the seasons, and there is not enough
history to learn that). This asks whether shifting the probabilities by what the most recent known
outcomes say would shrink that gap. Snapshots are 90 days apart and the outcome window is 90 days,
so at each origin the outcomes of the previous snapshot are already known: nothing from the future
is used (a snapshot whose window has not ended by the origin is never looked at).

Four variants, fixed before the first run so they cannot be tuned to the answer. Each moves the
average probability to a target by one constant on the log-odds scale, so the order of the customers
(the ranking, and so PR-AUC and the gain over the recency rule) is exactly what it was:

  none   no recalibration (the model as it is)
  full   the average goes to the churn rate of the previous snapshot (the level is assumed to stay)
  half   halfway between the model's average and that churn rate
  avg2   the model's average minus the mean of its last two gaps (its recent errors carry over)

The verdict is fixed too: a variant helps only if its average absolute gap, over the origins where
every variant can be computed, is smaller than that of `none` AND no origin gets worse by more than
MAX_ORIGIN_WORSENING. Run it with: python -m src.calibration_experiment

It reads the warehouse and MLflow and writes nothing. It is a measurement, not a change to the
model: the live model, the promotion gate and the pinned values are untouched.
"""

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss

from src import backtesting, config, database, features, measures, model, model_search, tracking
from src.errors import RegistryError

log = logging.getLogger(__name__)

VARIANTS = ("none", "full", "half", "avg2")
MAX_ORIGIN_WORSENING = 0.05  # no origin may get worse by more than 5 points of average probability
MIN_TARGET, MAX_TARGET = 0.01, 0.99  # a target outside this is not a plausible churn rate
RESULT_COLUMNS = [
    "origin",
    "variant",
    "customers",
    "actual",
    "mean_p",
    "gap",
    "abs_gap",
    "ece",
    "brier",
    "pr_auc",
    "comparable",
]


def shift_to_mean(p: npt.ArrayLike, target: float) -> npt.NDArray[np.float64]:
    """Probabilities moved by one constant on the log-odds scale so that their mean is `target`.

    The same constant is added to every customer's log-odds, which can never change their order.
    The constant is found by bisection (the mean rises with it).
    """
    if not 0 < target < 1:
        raise ValueError(f"The target mean must be between 0 and 1, not {target}")
    clipped = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    log_odds = np.log(clipped / (1 - clipped))
    low, high = -60.0, 60.0
    for _ in range(200):
        middle = (low + high) / 2
        if (1 / (1 + np.exp(-(log_odds + middle)))).mean() < target:
            low = middle
        else:
            high = middle
    return np.asarray(1 / (1 + np.exp(-(log_odds + (low + high) / 2))), dtype=float)


def _targets(
    mean_p: float, previous_rate: float, known_gaps: list[float]
) -> dict[str, float | None]:
    """The mean each variant moves the probabilities to (None where it cannot be computed)."""
    recent = known_gaps[:2]
    return {
        "none": mean_p,
        "full": previous_rate,
        "half": mean_p + 0.5 * (previous_rate - mean_p),
        "avg2": mean_p - float(np.mean(recent)) if recent else None,
    }


def evaluate(
    df: pd.DataFrame, factory: Callable[[model_search.Params], Any], params: model_search.Params
) -> pd.DataFrame:
    """Every variant at every backtest origin, trained and purged as backtesting.churn_backtest
    does it. One row per origin and variant; `comparable` is True where all four variants could be
    computed (the origin has at least one earlier gap to carry over)."""
    frame = df.reset_index(drop=True)
    X = features.prepare_features(frame)
    y = frame[features.CHURN_TARGET].to_numpy(dtype=int)
    dates = sorted(frame["snapshot_date"].unique())
    horizon = timedelta(days=features.HORIZON_DAYS)
    rate_of = {d: float(y[(frame["snapshot_date"] == d).to_numpy()].mean()) for d in dates}
    gap_at: dict[Any, float] = {}  # the model's own gap at each origin already evaluated

    rows: list[dict[str, Any]] = []
    for origin in dates:
        usable = [d for d in dates if d + horizon <= origin]  # their outcomes are known by now
        if len(usable) < backtesting.MIN_BACKTEST_TRAIN_SNAPSHOTS:
            continue
        train_mask = frame["snapshot_date"].isin(usable).to_numpy()
        origin_mask = (frame["snapshot_date"] == origin).to_numpy()
        p = factory(params).fit(X[train_mask], y[train_mask]).predict_proba(X[origin_mask])[:, 1]
        y_origin = y[origin_mask]
        actual = float(y_origin.mean())
        known_gaps = [gap_at[d] for d in sorted(gap_at, reverse=True) if d + horizon <= origin]
        targets = _targets(float(p.mean()), rate_of[usable[-1]], known_gaps)
        comparable = all(target is not None for target in targets.values())

        for variant in VARIANTS:
            target = targets[variant]
            if target is None:
                continue
            shifted = (
                p
                if variant == "none"
                else shift_to_mean(p, float(np.clip(target, MIN_TARGET, MAX_TARGET)))
            )
            rows.append(
                {
                    "origin": str(origin),
                    "variant": variant,
                    "customers": int(origin_mask.sum()),
                    "actual": actual,
                    "mean_p": float(shifted.mean()),
                    "gap": float(shifted.mean() - actual),
                    "abs_gap": abs(float(shifted.mean() - actual)),
                    "ece": measures.expected_calibration_error(y_origin, shifted),
                    "brier": float(brier_score_loss(y_origin, shifted)),
                    "pr_auc": float(average_precision_score(y_origin, shifted)),
                    "comparable": comparable,
                }
            )
        gap_at[origin] = float(p.mean() - actual)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def verdict(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each variant other than `none`: its average absolute gap against that of `none` over the
    comparable origins, the worst any one origin got, and whether it helps (see the module note)."""
    comparable = table[table["comparable"]]
    baseline = comparable[comparable["variant"] == "none"].set_index("origin")["abs_gap"]
    result: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS[1:]:
        gaps = comparable[comparable["variant"] == variant].set_index("origin")["abs_gap"]
        if baseline.empty or gaps.empty:
            result[variant] = {"origins": 0, "helps": False}
            continue
        worsening = float((gaps - baseline).max())
        result[variant] = {
            "origins": len(gaps),
            "mean_abs_gap": float(gaps.mean()),
            "baseline_mean_abs_gap": float(baseline.mean()),
            "worst_origin_worsening": worsening,
            "helps": bool(gaps.mean() < baseline.mean() and worsening <= MAX_ORIGIN_WORSENING),
        }
    return result


def champion_settings(client: MlflowClient) -> tuple[str, model_search.Params]:
    """The algorithm and hyperparameters the live churn champion was trained with, with the types
    the model expects (MLflow stores every parameter as text)."""
    version = client.get_model_version_by_alias(config.CHURN_MODEL_NAME, tracking.CHAMPION_ALIAS)
    if version.run_id is None:
        raise RegistryError(f"Churn champion version {version.version} has no training run")
    logged = client.get_run(version.run_id).data.params
    algorithm = logged["algorithm"]
    (candidate,) = [c for c in model_search.CHURN_CANDIDATES if c.name == algorithm]
    params: model_search.Params = {}
    for name, grid in candidate.grid.items():
        params[name] = type(grid[0])(logged[f"hp_{name}"])
    return algorithm, params


def run() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """The experiment on the real warehouse, with the live champion's own settings."""
    client = tracking.configure()
    algorithm, params = champion_settings(client)
    with database.get_connection() as conn:
        snapshots = features.load_snapshots(conn)
    train, test = model.split_labelled(snapshots)
    factory = model_search.candidate_factory(model_search.CHURN_CANDIDATES, algorithm)
    table = evaluate(pd.concat([train, test]), factory, params)
    return table, verdict(table)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table, outcome = run()
    pd.set_option("display.width", 200)
    shown = table.assign(
        **{c: table[c].round(3) for c in ("actual", "mean_p", "gap", "abs_gap", "ece", "brier")}
    )
    print(shown.drop(columns=["pr_auc"]).to_string(index=False))
    print("\nPR-AUC is the same for every variant at each origin:")
    print(table.groupby("origin")["pr_auc"].agg(lambda s: round(float(s.max() - s.min()), 12)))
    print("\nverdict (a variant helps only if it lowers the mean absolute gap AND no origin")
    print(f"gets worse by more than {MAX_ORIGIN_WORSENING}):")
    for variant, result in outcome.items():
        print(f"  {variant}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
