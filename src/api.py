"""HTTP API for the retention models (FastAPI).

Two kinds of answer, deliberately both:

  stored   what batch scoring wrote to ml.predictions: reliable and auditable, it is the list the
           outreach plan was made from.
  online   the live champion models applied now to a customer's features from the latest snapshot
           (or to features you send, for what-if questions).

Both use scoring.score_features(), so for the same model versions they give the same numbers. The
features are only as fresh as the latest snapshot: scoring on request does not make a customer's
inputs more current. Every answer says which model versions produced it.

Only customers in the models' population (bought within the last 180 days) can be scored; others
are lapsed customers for a win-back list, and asking about them is an explained error, not a
meaningless number. There is no authentication: this is a local service bound to 127.0.0.1.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import pandas as pd
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from src import config, database, features, optimizer, scoring, tracking
from src.errors import DatabaseUnavailableError, PipelineError

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ schemas

# Allowed values for the inputs of a what-if request. Anything not listed is unrestricted
# (money can be negative when returns exceed purchases).
_BOUNDS: dict[str, dict[str, Any]] = {
    "recency_days": {"ge": 0},
    "tenure_days_capped": {"ge": 0, "le": 365},
    "history_days_available": {"ge": 0, "le": 365},
    "frequency_365d": {"ge": 0},
    "total_units_365d": {"ge": 0},
    "distinct_products_365d": {"ge": 0},
    "orders_last_90d": {"ge": 0},
    "orders_prev_90d": {"ge": 0},
    "return_invoices_365d": {"ge": 0},
    "return_value_365d": {"ge": 0},
    "is_uk": {"ge": 0, "le": 1},
    "season_sin": {"ge": -1, "le": 1},
    "season_cos": {"ge": -1, "le": 1},
}
# Undefined (null) when a customer has no orders in the 365-day window, or one order
_NULLABLE = {"avg_order_value_365d", "avg_days_between_orders_365d", "return_rate_365d"}


def _request_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in features.BASE_FEATURES:
        nullable = name in _NULLABLE
        default: Any = None if nullable else ...
        fields[name] = (
            float | None if nullable else float,
            Field(default, **_BOUNDS.get(name, {})),
        )
    return fields


# The request has exactly the model inputs, so a change to the feature contract changes the API
# (a test checks the two stay in step).
FeaturesRequest = create_model(
    "FeaturesRequest", __config__=ConfigDict(extra="forbid"), **_request_fields()
)


class ScoreResponse(BaseModel):
    source: Literal["online", "stored", "what-if"]
    customer_id: int | None = None
    snapshot_date: str | None = None
    segment: str | None = None
    p_churn: float = Field(description="chance of no net-positive spend in the next 90 days")
    spend_if_retained: float = Field(description="expected 90-day spend if the customer buys")
    expected_spend: float = Field(description="expected 90-day spend = P(buy) x spend_if_retained")
    margin_at_risk: float = Field(description="p_churn x spend_if_retained x margin")
    margin_at_risk_after_cost: float
    worth_contacting: bool = Field(description="margin at risk is more than the contact cost")
    churn_model_version: str
    value_model_version: str
    scored_at: str | None = None


class OutreachRow(BaseModel):
    customer_id: int
    rank: int
    group: Literal["contact", "holdout", "not_selected"]
    p_churn: float
    spend_if_retained: float
    margin_at_risk: float


class OutreachPlan(BaseModel):
    plan_id: int
    created_at: str
    scoring_run_id: int
    k: int
    holdout_fraction: float
    seed: int
    margin: float
    contact_cost: float
    contact_customers: int
    holdout_customers: int
    customers: list[OutreachRow]


# ------------------------------------------------------------------ the app


def _default_loader() -> scoring.Champions:
    return scoring.load_champions(tracking.configure())


def create_app(load_champions: Callable[[], scoring.Champions] = _default_loader) -> FastAPI:
    """Build the API. load_champions is replaceable so tests can run without a model registry."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        _load(app)
        yield

    def _load(app: FastAPI) -> None:
        try:
            app.state.champions = load_champions()
            app.state.load_error = None
            log.info(
                "Loaded churn v%s and value v%s",
                app.state.champions.churn_version,
                app.state.champions.value_version,
            )
        except PipelineError as exc:  # the registry is down or has no usable champion
            app.state.champions = None
            app.state.load_error = str(exc)
            log.error("Models not loaded: %s", exc)

    app = FastAPI(
        title="Retail retention API",
        description="Churn risk, expected spend and margin at risk for customers of the UCI "
        "Online Retail II dataset. See docs/problem_statement.md for what the numbers mean "
        "and what they do not claim.",
        lifespan=lifespan,
    )

    def champions(request: Request) -> scoring.Champions:
        loaded = request.app.state.champions
        if loaded is None:
            raise HTTPException(503, f"The models are not loaded: {request.app.state.load_error}")
        return loaded  # type: ignore[no-any-return]

    def connection() -> Iterator[psycopg.Connection[Any]]:
        try:
            conn = database.get_connection(retries=1)
        except DatabaseUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            yield conn
        finally:
            conn.close()

    Models = Annotated[scoring.Champions, Depends(champions)]
    Conn = Annotated[psycopg.Connection[Any], Depends(connection)]

    def response(
        source: str,
        scores: pd.Series,
        models: scoring.Champions,
        customer_id: int | None = None,
        snapshot_date: Any = None,
        segment: str | None = None,
    ) -> ScoreResponse:
        return ScoreResponse(
            source=source,  # type: ignore[arg-type]
            customer_id=customer_id,
            snapshot_date=str(snapshot_date) if snapshot_date is not None else None,
            segment=segment,
            p_churn=float(scores["p_churn"]),
            spend_if_retained=float(scores["spend_if_retained"]),
            expected_spend=float(scores["expected_spend"]),
            margin_at_risk=float(scores["margin_at_risk"]),
            margin_at_risk_after_cost=float(scores["margin_at_risk_after_cost"]),
            worth_contacting=bool(scores["margin_at_risk_after_cost"] > 0),
            churn_model_version=models.churn_version,
            value_model_version=models.value_version,
        )

    def segment_of(conn: psycopg.Connection[Any], customer_id: int) -> str | None:
        row = conn.execute(
            "SELECT segment_name FROM marts.customer_segments WHERE customer_id = %s "
            "ORDER BY snapshot_date DESC LIMIT 1",
            (customer_id,),
        ).fetchone()
        conn.rollback()
        return row[0] if row else None

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Whether the service can answer: models loaded and the database reachable."""
        loaded = request.app.state.champions
        report: dict[str, Any] = {
            "status": "ok" if loaded is not None else "degraded",
            "models_loaded": loaded is not None,
        }
        if loaded is not None:
            report.update(
                churn_model_version=loaded.churn_version, value_model_version=loaded.value_version
            )
        else:
            report["problem"] = request.app.state.load_error
        try:
            with database.get_connection(retries=1) as conn:
                run = conn.execute(
                    "SELECT scoring_run_id, scored_at::text, snapshot_date::text "
                    "FROM ml.scoring_runs ORDER BY scoring_run_id DESC LIMIT 1"
                ).fetchone()
            report["database"] = "ok"
            report["latest_scoring_run"] = (
                dict(zip(["scoring_run_id", "scored_at", "snapshot_date"], run, strict=True))
                if run
                else None
            )
        except PipelineError as exc:
            report["database"] = "unavailable"
            report["status"] = "degraded"
            report["problem"] = str(exc)
        if report["status"] != "ok":
            raise HTTPException(503, detail=report)
        return report

    @app.post("/models/reload")
    def reload_models(request: Request) -> dict[str, Any]:
        """Pick up newly promoted champion models without restarting the service."""
        _load(request.app)
        if request.app.state.champions is None:
            raise HTTPException(503, f"The models are not loaded: {request.app.state.load_error}")
        loaded = request.app.state.champions
        return {
            "churn_model_version": loaded.churn_version,
            "value_model_version": loaded.value_version,
        }

    def scorable_customer(conn: psycopg.Connection[Any], customer_id: int) -> pd.DataFrame:
        """The customer's features from the latest snapshot, or an explained error."""
        frame = features.load_score_customer(conn, customer_id)
        if frame.empty:
            raise HTTPException(404, f"Customer {customer_id} is not in the latest snapshot.")
        if not bool(frame["is_actionable"].iloc[0]):
            days = int(frame["recency_days"].iloc[0])
            raise HTTPException(
                422,
                f"Customer {customer_id} last bought {days} days before the snapshot, more than "
                f"{features.ACTIONABLE_RECENCY_DAYS}: outside the population the models were "
                "trained for (a win-back customer). No score is given.",
            )
        return frame

    @app.get("/customers/{customer_id}/features")
    def customer_features(customer_id: int, conn: Conn) -> dict[str, Any]:
        """The model inputs for a customer: a starting point for what-if questions (POST /score)."""
        frame = scorable_customer(conn, customer_id)
        row = frame.iloc[0]
        return {
            "customer_id": customer_id,
            "snapshot_date": str(row["snapshot_date"]),
            "features": {
                name: None if pd.isna(row[name]) else float(row[name])
                for name in features.BASE_FEATURES
            },
        }

    @app.get("/customers/{customer_id}/score", response_model=ScoreResponse)
    def score_customer(customer_id: int, models: Models, conn: Conn) -> ScoreResponse:
        """Score a customer now with the live models, from their latest-snapshot features."""
        frame = scorable_customer(conn, customer_id)
        scores = scoring.score_features(frame, models).iloc[0]
        return response(
            "online",
            scores,
            models,
            customer_id,
            frame["snapshot_date"].iloc[0],
            segment_of(conn, customer_id),
        )

    @app.get("/customers/{customer_id}/prediction", response_model=ScoreResponse)
    def stored_prediction(customer_id: int, conn: Conn) -> ScoreResponse:
        """What batch scoring stored for a customer in the latest scoring run."""
        row = conn.execute(
            "SELECT p_churn, spend_if_retained, expected_spend, margin_at_risk, "
            "margin_at_risk_after_cost, churn_model_version, value_model_version, "
            "snapshot_date::text, scored_at::text "
            "FROM ml.latest_predictions WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()
        conn.rollback()
        if row is None:
            raise HTTPException(
                404,
                f"No stored prediction for customer {customer_id} in the latest scoring run. "
                "They may be outside the models' population, or scoring has not run.",
            )
        return ScoreResponse(
            source="stored",
            customer_id=customer_id,
            snapshot_date=row[7],
            segment=segment_of(conn, customer_id),
            p_churn=row[0],
            spend_if_retained=row[1],
            expected_spend=row[2],
            margin_at_risk=row[3],
            margin_at_risk_after_cost=row[4],
            worth_contacting=row[4] > 0,
            churn_model_version=row[5],
            value_model_version=row[6],
            scored_at=row[8],
        )

    @app.post("/score", response_model=ScoreResponse)
    def what_if(body: FeaturesRequest, models: Models) -> ScoreResponse:  # type: ignore[valid-type]
        """Score features you supply, for what-if questions. Nothing is stored."""
        frame = pd.DataFrame([body.model_dump()]).astype(float)  # type: ignore[attr-defined]
        return response("what-if", scoring.score_features(frame, models).iloc[0], models)

    @app.get("/outreach/latest", response_model=OutreachPlan)
    def latest_outreach_plan(
        conn: Conn,
        group: Annotated[Literal["contact", "holdout", "not_selected"] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=5000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> OutreachPlan:
        """The most recent outreach plan: who to contact, and who is deliberately held out."""
        plan = conn.execute(
            "SELECT plan_id, created_at::text, scoring_run_id, k, holdout_fraction, seed, margin, "
            "contact_cost, contact_customers, holdout_customers "
            "FROM ml.outreach_plans ORDER BY plan_id DESC LIMIT 1"
        ).fetchone()
        if plan is None:
            conn.rollback()
            raise HTTPException(
                404, "There is no outreach plan yet: python -m src.main --stage plan"
            )
        rows = conn.execute(
            """
            SELECT a.customer_id, a.rank, a.grp, p.p_churn, p.spend_if_retained, p.margin_at_risk
            FROM ml.outreach_assignments a
            JOIN ml.outreach_plans o USING (plan_id)
            JOIN ml.predictions p
              ON p.scoring_run_id = o.scoring_run_id AND p.customer_id = a.customer_id
            WHERE a.plan_id = %s AND (%s::text IS NULL OR a.grp = %s)
            ORDER BY a.rank
            LIMIT %s OFFSET %s
            """,
            (plan[0], group, group, limit, offset),
        ).fetchall()
        conn.rollback()
        return OutreachPlan(
            **dict(
                zip(
                    [
                        "plan_id",
                        "created_at",
                        "scoring_run_id",
                        "k",
                        "holdout_fraction",
                        "seed",
                        "margin",
                        "contact_cost",
                        "contact_customers",
                        "holdout_customers",
                    ],
                    plan,
                    strict=True,
                )
            ),
            customers=[
                OutreachRow(
                    customer_id=r[0],
                    rank=r[1],
                    group=r[2],
                    p_churn=r[3],
                    spend_if_retained=r[4],
                    margin_at_risk=r[5],
                )
                for r in rows
            ],
        )

    @app.get("/outreach/sensitivity")
    def outreach_sensitivity(
        conn: Conn,
        k: Annotated[int, Query(ge=1, le=5000)] = optimizer.DEFAULT_OUTREACH_SIZE,
        margins: Annotated[list[float], Query(max_length=10)] = [0.2, 0.3, 0.4],  # noqa: B006
        contact_costs: Annotated[list[float], Query(max_length=10)] = [3.0, 5.0, 8.0],  # noqa: B006
    ) -> list[dict[str, Any]]:
        """How the selected list changes if the margin or contact-cost assumptions change."""
        try:
            scores, _ = optimizer.load_scores(conn)
        except PipelineError as exc:
            raise HTTPException(404, str(exc)) from exc
        if any(m <= 0 for m in margins) or any(c < 0 for c in contact_costs):
            raise HTTPException(422, "Margins must be above 0 and contact costs at least 0.")
        table = optimizer.sensitivity(scores, k, margins, contact_costs)
        return [{str(key): value for key, value in row.items()} for row in table.to_dict("records")]

    return app


app = create_app()

__all__ = ["FeaturesRequest", "ScoreResponse", "app", "config", "create_app"]
