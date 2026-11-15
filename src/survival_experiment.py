"""A trial of a survival-style churn scorer, judged where a retention plan would use it.

The churn model asks "will this customer buy in the next 90 days?" as one yes/no question. A
survival model asks "how long until the next purchase?" and reads churn off the answer: a customer
churns if they are still waiting after 90 days. This trial uses the discrete-time version (Singer
and Willett): the 90 days are cut into N_INTERVALS periods of INTERVAL_DAYS days, every customer
contributes one row for each period they were still waiting at the start of (the "person-period"
rows), a classifier learns the chance of a purchase in a period given none before it (the
hazard), and a customer's churn score is the chance of getting through every period without one:

  score = (1 - hazard_1) x (1 - hazard_2) x ... x (1 - hazard_N)

Only scikit-learn is used. The inputs are the churn model's own features (as they were at the
snapshot; they are not updated inside the 90 days) plus the period number.

What it is compared with, at every backtest origin (trained and purged exactly as
backtesting.churn_backtest does, with the live champion's own hyperparameters and nothing tuned):

  current          the live churn model (one yes/no question)
  survival         the discrete-time hazard model above
  recency          rank by days since the last order
  bgnbd            1 - P(alive) from the BG/NBD model (fitted at the latest usable snapshot)
  current_x_spend  current model's probability x spend in the last 365 days, a stand-in for
                   margin at risk (the real one needs the value model, which is not retrained here)

Judged on the customers a plan would contact, the TOP_FRACTION highest scored (about the share
500 is of the 3,471 customers scored). The primary measure is the share of the churners' spend in
the 365 days before the snapshot that the contacted group reaches. What a churner would have spent
had they stayed is not known; their past spend is the stand-in. The window has to be longer than
the 90 days that define churn: a customer who has not bought for 90 days has no spend inside a
90-day window, so such a window gives the likeliest churners no weight at all. Every actionable
customer bought within the last 180 days, so each one falls inside 365. The others are secondary:
the share of churners reached, the hit rate and PR-AUC.

The verdict rule is fixed. The survival scorer is BETTER than the current model
only if its primary measure is higher at at least WIN_SHARE of the origins AND the 95% interval
of its average advantage over the origins (customers resampled within each origin, the two scorers
compared on the same resample) is above zero. WORSE if that interval is entirely below zero.
Otherwise NO BETTER. A result of "no better" is a result, and nothing in production changes
either way: adopting a winner would be a separate change, through the promotion gate.

One simplification to know about: the event is the customer's first SALE in the window, while the
churn label is zero NET spend (a customer whose only purchase was returned in full counts as
churned). Evaluation is against the real label; how often the two differ is reported.

Run it with: python -m src.survival_experiment
It reads the warehouse and MLflow and writes nothing.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import psycopg
from sklearn.metrics import average_precision_score

from src import (
    backtesting,
    baselines,
    calibration_experiment,
    database,
    evaluation,
    features,
    measures,
    model_search,
    tracking,
)

log = logging.getLogger(__name__)

INTERVAL_DAYS = 30
N_INTERVALS = 3
TOP_FRACTION = 0.15
WIN_SHARE = 0.75  # the survival scorer must lead at 3 of 4 origins
INTERVAL_COLUMN = "interval"
SPEND_COLUMN = "monetary_365d"  # what a churner leaves behind: net spend over the last 365 days
METHODS = ("current", "survival", "recency", "bgnbd", "current_x_spend")
RESULT_COLUMNS = [
    "origin",
    "method",
    "customers",
    "churners",
    "spend_share_top",
    "recall_top",
    "precision_top",
    "pr_auc",
    "label_mismatch",
]

if INTERVAL_DAYS * N_INTERVALS != features.HORIZON_DAYS:
    raise RuntimeError("The hazard intervals must add up to the churn horizon")


# --------------------------------------------------------------- the event times


_EVENT_QUERY = """
    SELECT s.snapshot_date,
           t.customer_id,
           (min(t.invoice_date)::date - s.snapshot_date) AS days_to_first_sale
    FROM marts.snapshot_dates s
    JOIN staging.transactions t
      ON t.line_type = 'sale'
     AND t.customer_id IS NOT NULL
     AND t.invoice_date >= s.snapshot_date
     AND t.invoice_date <  s.snapshot_date + s.horizon_days
    WHERE s.split <> 'score'
    GROUP BY s.snapshot_date, t.customer_id
"""


def load_event_days(conn: psycopg.Connection[Any]) -> pd.DataFrame:
    """Days from each labelled snapshot to the customer's first sale inside its label window.

    One row per customer who bought in the window; customers who did not are simply absent (they
    waited the whole window, which is what "churned" means for this model)."""
    return database.read_dataframe(conn, _EVENT_QUERY)


def attach_event_days(frame: pd.DataFrame, events: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Each row's days to the first sale, NaN when there was none, in the frame's row order."""
    merged = frame[["snapshot_date", "customer_id"]].merge(
        events, on=["snapshot_date", "customer_id"], how="left"
    )
    return np.asarray(merged["days_to_first_sale"].to_numpy(dtype=float), dtype=float)


# ------------------------------------------------------------ the hazard model


def person_period(
    X: pd.DataFrame, days: npt.ArrayLike
) -> tuple[pd.DataFrame, npt.NDArray[np.int64]]:
    """One row per customer and period they were still waiting at the start of.

    days is the time to the first sale (NaN: none inside the window). In period j (1-based) a
    customer is at risk if they had not bought before it starts, and the row's target is 1 if
    they bought during it. A customer who never bought is at risk in every period, with target 0.
    """
    d = np.asarray(days, dtype=float)
    parts: list[pd.DataFrame] = []
    targets: list[npt.NDArray[np.int64]] = []
    for j in range(1, N_INTERVALS + 1):
        start, end = (j - 1) * INTERVAL_DAYS, j * INTERVAL_DAYS
        at_risk = np.isnan(d) | (d >= start)
        part = X.loc[at_risk].copy()
        part[INTERVAL_COLUMN] = float(j)
        parts.append(part)
        targets.append(((d[at_risk] >= start) & (d[at_risk] < end)).astype(np.int64))
    return pd.concat(parts, ignore_index=True), np.concatenate(targets)


def fit_hazard(
    factory: Callable[[model_search.Params], Any],
    params: model_search.Params,
    X: pd.DataFrame,
    days: npt.ArrayLike,
) -> Any:
    """The hazard classifier, trained on the person-period rows of these customers."""
    X_pp, y_pp = person_period(X, days)
    return factory(params).fit(X_pp, y_pp)


def survival_scores(fitted: Any, X: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Churn score: the chance of getting through every period without a purchase."""
    surviving = np.ones(len(X))
    for j in range(1, N_INTERVALS + 1):
        rows = X.copy()
        rows[INTERVAL_COLUMN] = float(j)
        surviving *= 1.0 - np.asarray(fitted.predict_proba(rows)[:, 1], dtype=float)
    return surviving


# ------------------------------------------------------------------ the measures


def churner_spend(churned: npt.ArrayLike, spend: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The spend a churner leaves behind (their last 365 days, never below zero); zero for the
    customers who did not churn."""
    spend = np.clip(np.nan_to_num(np.asarray(spend, dtype=float)), 0.0, None)
    return np.asarray(np.asarray(churned, dtype=float) * spend, dtype=float)


def spend_share_top(
    churned: npt.ArrayLike,
    spend: npt.ArrayLike,
    score: npt.ArrayLike,
    fraction: float = TOP_FRACTION,
) -> float:
    """Share of the churners' spend that is held by the highest-scored fraction of customers."""
    at_stake = churner_spend(churned, spend)
    total = float(at_stake.sum())
    if total <= 0:
        return float("nan")
    return measures.top_fraction_total(at_stake, score, fraction)[0] / total


def _bootstrap_metric(y: npt.NDArray[np.float64], score: npt.NDArray[np.float64]) -> float:
    """spend_share_top for evaluation's bootstrap, which resamples one array: column 0 is the
    churn label and column 1 the recent spend."""
    value = spend_share_top(y[:, 0], y[:, 1], score)
    if not np.isfinite(value):
        raise ValueError("No churner spend in this resample")
    return value


@dataclass(frozen=True)
class OriginData:
    """What one origin was scored on: the outcome, the spend, and every method's scores."""

    origin: str
    churned: npt.NDArray[np.float64]
    spend: npt.NDArray[np.float64]
    scores: dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class Trial:
    table: pd.DataFrame
    origins: list[OriginData]


def evaluate(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    factory: Callable[[model_search.Params], Any],
    params: model_search.Params,
) -> Trial:
    """Every method at every backtest origin. snapshots is every labelled row of the snapshot
    table (all customers, because BG/NBD is fitted on them all); the models are trained and scored
    on the actionable customers only, as the production backtest does."""
    frame = snapshots.reset_index(drop=True)
    dates = sorted(frame["snapshot_date"].unique())
    horizon = timedelta(days=features.HORIZON_DAYS)
    actionable = frame[frame["is_actionable"].astype(bool)].reset_index(drop=True)
    X = features.prepare_features(actionable)
    y = actionable[features.CHURN_TARGET].to_numpy(dtype=int)
    days = attach_event_days(actionable, events)

    rows: list[dict[str, Any]] = []
    origins: list[OriginData] = []
    for origin in dates:
        usable = [d for d in dates if d + horizon <= origin]  # their outcomes are known by now
        if len(usable) < backtesting.MIN_BACKTEST_TRAIN_SNAPSHOTS:
            continue
        train_mask = actionable["snapshot_date"].isin(usable).to_numpy()
        origin_mask = (actionable["snapshot_date"] == origin).to_numpy()
        X_origin, y_origin = X[origin_mask], y[origin_mask]
        spend = actionable.loc[origin_mask, SPEND_COLUMN].to_numpy(dtype=float)

        current = factory(params).fit(X[train_mask], y[train_mask]).predict_proba(X_origin)[:, 1]
        survival = survival_scores(
            fit_hazard(factory, params, X[train_mask], days[train_mask]), X_origin
        )
        scores: dict[str, npt.NDArray[np.float64]] = {
            "current": np.asarray(current, dtype=float),
            "survival": survival,
            "recency": X_origin["recency_days"].to_numpy(dtype=float),
            "current_x_spend": np.asarray(current, dtype=float)
            * np.clip(np.nan_to_num(spend), 0.0, None),
        }
        bgnbd = _bgnbd_scores(frame, usable[-1], actionable.loc[origin_mask])
        if bgnbd is not None:
            scores["bgnbd"] = bgnbd

        no_sale = np.isnan(days[origin_mask])
        mismatch = float((no_sale != y_origin.astype(bool)).mean())
        for method in METHODS:
            if method not in scores:
                continue
            score = scores[method]
            rows.append(
                {
                    "origin": str(origin),
                    "method": method,
                    "customers": int(origin_mask.sum()),
                    "churners": int(y_origin.sum()),
                    "spend_share_top": spend_share_top(y_origin, spend, score),
                    "recall_top": measures.recall_at_top_fraction(y_origin, score, TOP_FRACTION),
                    "precision_top": measures.precision_at_top_fraction(
                        y_origin, score, TOP_FRACTION
                    ),
                    "pr_auc": float(average_precision_score(y_origin, score)),
                    "label_mismatch": mismatch,
                }
            )
        origins.append(OriginData(str(origin), y_origin.astype(float), spend, scores))
    return Trial(pd.DataFrame(rows, columns=RESULT_COLUMNS), origins)


def _bgnbd_scores(
    frame: pd.DataFrame, fitted_at: Any, origin_rows: pd.DataFrame
) -> npt.NDArray[np.float64] | None:
    """1 - P(alive) at the origin from BG/NBD fitted on every customer at the latest usable
    snapshot. None when the fit fails: a baseline that cannot be fitted must not stop the rest."""
    cohort = frame[frame["snapshot_date"] == fitted_at]
    try:
        fitted = baselines.BgNbd.fit(
            cohort["frequency"], cohort["recency_days"], cohort["tenure_days"]
        )
    except ValueError as exc:
        log.warning("BG/NBD baseline skipped: %s", exc)
        return None
    alive = fitted.probability_alive(
        origin_rows["frequency"], origin_rows["recency_days"], origin_rows["tenure_days"]
    )
    return np.asarray(1.0 - alive, dtype=float)


# ------------------------------------------------------------------ the verdict


def pooled_gain(
    trial: Trial,
    candidate: str,
    reference: str,
    n_resamples: int | None = None,
    seed: int = 0,
) -> evaluation.Gain:
    """The average over the origins of the candidate's primary measure minus the reference's, with
    a 95% interval from resampling the customers within every origin, both scored on the same
    resample. Origins where either method has no scores are left out."""
    n_resamples = evaluation.N_BOOTSTRAP if n_resamples is None else n_resamples
    usable = [o for o in trial.origins if candidate in o.scores and reference in o.scores]
    if not usable:
        raise ValueError("No origin has scores for both methods")
    stacked = [np.column_stack([o.churned, o.spend]) for o in usable]
    rng = np.random.default_rng(seed)

    differences: list[float] = []
    for _ in range(n_resamples):
        per_origin = []
        try:
            for o, y in zip(usable, stacked, strict=True):
                picked = rng.integers(0, len(y), len(y))
                per_origin.append(
                    _bootstrap_metric(y[picked], o.scores[candidate][picked])
                    - _bootstrap_metric(y[picked], o.scores[reference][picked])
                )
        except ValueError:
            continue
        differences.append(float(np.mean(per_origin)))
    if len(differences) < n_resamples // 2:
        raise ValueError("The measure was undefined in too many resamples")

    point_candidate = float(
        np.mean([spend_share_top(o.churned, o.spend, o.scores[candidate]) for o in usable])
    )
    point_reference = float(
        np.mean([spend_share_top(o.churned, o.spend, o.scores[reference]) for o in usable])
    )
    tail = (1 - evaluation.CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    return evaluation.Gain(
        candidate=point_candidate,
        reference=point_reference,
        diff=point_candidate - point_reference,
        low=float(low),
        high=float(high),
        resamples=len(differences),
    )


def origins_won(trial: Trial, candidate: str, reference: str) -> tuple[int, int]:
    """At how many origins the candidate's primary measure is strictly higher, and how many origins
    there are to compare."""
    wins = comparable = 0
    for o in trial.origins:
        if candidate not in o.scores or reference not in o.scores:
            continue
        comparable += 1
        a = spend_share_top(o.churned, o.spend, o.scores[candidate])
        b = spend_share_top(o.churned, o.spend, o.scores[reference])
        wins += int(a > b)
    return wins, comparable


def decide(wins: int, origins: int, gain: evaluation.Gain) -> str:
    """The fixed rule (see the module note): "better", "worse" or "no better"."""
    if origins and wins >= math.ceil(WIN_SHARE * origins) and gain.low > 0:
        return "better"
    if gain.high < 0:
        return "worse"
    return "no better"


def compare(
    trial: Trial, candidate: str, reference: str, n_resamples: int | None = None
) -> dict[str, Any]:
    wins, origins = origins_won(trial, candidate, reference)
    gain = pooled_gain(trial, candidate, reference, n_resamples)
    return {
        "candidate": candidate,
        "reference": reference,
        "origins_won": wins,
        "origins": origins,
        "mean_advantage": gain.diff,
        "interval": (gain.low, gain.high),
        "verdict": decide(wins, origins, gain),
    }


def run() -> tuple[Trial, dict[str, Any], dict[str, Any]]:
    """The trial on the real warehouse, with the live champion's own settings. Returns the trial,
    the survival-against-current comparison (the verdict) and the stand-in for margin at risk
    against the current model on its own (for information; it has no verdict of its own)."""
    client = tracking.configure()
    algorithm, params = calibration_experiment.champion_settings(client)
    with database.get_connection() as conn:
        snapshots = features.load_snapshots(conn)
        events = load_event_days(conn)
    labelled = snapshots[snapshots["split"] != "score"]
    factory = model_search.candidate_factory(model_search.CHURN_CANDIDATES, algorithm)
    trial = evaluate(labelled, events, factory, params)
    return (
        trial,
        compare(trial, "survival", "current"),
        compare(trial, "current_x_spend", "current"),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trial, verdict, stand_in = run()
    pd.set_option("display.width", 200)
    shown = trial.table.round(3)
    print(shown.to_string(index=False))
    print(f"\nTop {TOP_FRACTION:.0%} of customers; primary measure: spend_share_top.")
    print("\nverdict (fixed in advance; see the module note):")
    print(f"  {verdict}")
    print("\nfor information, the margin-at-risk stand-in against the current model alone:")
    print(f"  {stand_in}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
