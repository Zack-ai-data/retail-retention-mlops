"""Scoring: the one place that turns customer features into a churn probability, expected spend
and the margin at risk.

Batch scoring (the quarterly outreach list in ml.predictions) and the API both call
score_features(), so a customer cannot get one answer from the table and another from the API for
the same model versions (a test checks this).

margin_at_risk = P(churn) x spend_if_retained x margin

spend_if_retained is what the customer is expected to spend if they do buy (the spend stage of the
value model). It is not multiplied by P(buy) again: P(churn) already carries the chance that they
do not buy. margin_at_risk is the margin expected to be lost, an upper bound on what contacting
the customer could save. The project has no campaign data, so it ranks risk and claims no uplift.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg
from mlflow import MlflowClient

from src import config, database, features, hurdle, tracking
from src.errors import EmptyTableError, RegistryError, ScoringError

log = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "p_churn",
    "spend_if_retained",
    "expected_spend",
    "margin_at_risk",
    "margin_at_risk_after_cost",
]


@dataclass(frozen=True)
class Champions:
    """The models that are live, with the versions that scores must be attributed to."""

    churn_model: Any
    value_model: hurdle.HurdleValueModel
    churn_version: str
    value_version: str


def _load(client: MlflowClient, name: str) -> tracking.Champion:
    champion = tracking.load_champion(client, name, features.FEATURE_COLUMNS)
    if champion is None:
        raise RegistryError(
            f"There is no champion for {name!r} yet. Train the models first: "
            "python -m src.main --stage train"
        )
    if champion.model is None:
        raise RegistryError(
            f"The champion for {name!r} (version {champion.version}) cannot score the current "
            f"features because {champion.incompatible_reason}. Retrain: "
            "python -m src.main --stage train"
        )
    return champion


def load_champions(client: MlflowClient) -> Champions:
    """Load the live churn and value models from the registry."""
    churn = _load(client, config.CHURN_MODEL_NAME)
    value = _load(client, config.VALUE_MODEL_NAME)
    if not isinstance(value.model, hurdle.HurdleValueModel):
        raise RegistryError(
            f"The champion for {config.VALUE_MODEL_NAME!r} (version {value.version}) is not a "
            "two-part value model, so the spend if the customer buys is not available. Retrain."
        )
    return Champions(churn.model, value.model, churn.version, value.version)


def score_features(
    frame: pd.DataFrame,
    champions: Champions,
    margin: float = config.MARGIN,
    contact_cost: float = config.CONTACT_COST_GBP,
) -> pd.DataFrame:
    """Score customers. frame needs the model input columns (features.BASE_FEATURES); the result
    has SCORE_COLUMNS and the same index."""
    X = features.prepare_features(frame)
    p_churn = champions.churn_model.predict_proba(X)[:, 1]
    spend_if_retained = champions.value_model.predict_spend_if_buying(X)
    expected_spend = champions.value_model.predict(X)
    margin_at_risk = p_churn * spend_if_retained * margin

    scores = pd.DataFrame(
        {
            "p_churn": p_churn,
            "spend_if_retained": spend_if_retained,
            "expected_spend": expected_spend,
            "margin_at_risk": margin_at_risk,
            "margin_at_risk_after_cost": margin_at_risk - contact_cost,
        },
        index=frame.index,
    )
    if (
        not scores.notna().all().all()
        or not ((scores["p_churn"] >= 0) & (scores["p_churn"] <= 1)).all()
    ):
        bad = scores[scores.isna().any(axis=1) | ~scores["p_churn"].between(0, 1)]
        raise ScoringError(
            f"The models produced invalid scores for {len(bad)} customers (first: "
            f"{list(bad.index[:3])}). Refusing to use them."
        )
    return scores


@dataclass(frozen=True)
class ScoringRun:
    scoring_run_id: int
    snapshot_date: Any
    customers: int
    churn_version: str
    value_version: str


def score_batch(conn: psycopg.Connection[Any], client: MlflowClient) -> ScoringRun:
    """Score every customer in the models' population and store the result in ml.predictions.

    Only customers who bought within the last 180 days are scored; lapsed customers are outside
    what the models were trained for. The run and its predictions are written in one transaction.
    """
    frame = features.load_score_snapshot(conn)
    if frame.empty:
        raise EmptyTableError(
            "marts.customer_snapshots has no scoring snapshot. Run: "
            "python -m src.main --stage transform"
        )
    population = frame[frame["is_actionable"]].reset_index(drop=True)
    if population.empty:
        raise EmptyTableError("No customer in the scoring snapshot is in the models' population")
    snapshot_date = population["snapshot_date"].iloc[0]

    champions = load_champions(client)
    scores = score_features(population, champions)

    with conn.transaction():
        run_id = conn.execute(
            """
            INSERT INTO ml.scoring_runs
                (snapshot_date, churn_model_version, value_model_version, margin, contact_cost,
                 customers)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING scoring_run_id
            """,
            (
                snapshot_date,
                champions.churn_version,
                champions.value_version,
                config.MARGIN,
                config.CONTACT_COST_GBP,
                len(scores),
            ),
        ).fetchone()
        assert run_id is not None
        rows = scores.assign(
            scoring_run_id=run_id[0], customer_id=population["customer_id"].astype(int)
        )
        database.copy_dataframe(
            conn, rows[["scoring_run_id", "customer_id", *SCORE_COLUMNS]], "ml.predictions"
        )
    conn.commit()
    log.info(
        "Scored %s customers as of %s with churn v%s and value v%s (scoring run %s)",
        f"{len(scores):,}",
        snapshot_date,
        champions.churn_version,
        champions.value_version,
        run_id[0],
    )
    return ScoringRun(
        run_id[0], snapshot_date, len(scores), champions.churn_version, champions.value_version
    )
