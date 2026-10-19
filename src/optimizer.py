"""Outreach optimizer: who to contact this quarter, within a fixed budget.

Customers are ranked by margin at risk (see scoring.py). The plan takes the top k, but only those
whose margin at risk is more than the cost of contacting them, so the budget is not spent on
customers who cannot pay for it even if every one of them were saved. A random share of the
selected customers is held out and not contacted: the project has no campaign history, so without
a holdout group nobody could ever tell whether contacting customers changed anything.

Groups:
  contact       selected and contacted
  holdout       selected, deliberately NOT contacted (a random sample, fixed by the seed)
  not_selected  not in the top k, or not worth the contact cost

The ranking is by risk, not by who can be saved. Nothing here claims an uplift.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from src import config, database
from src.errors import EmptyTableError

log = logging.getLogger(__name__)

# Customers selected per quarter. The contact budget is not in the data; this is a stated
# assumption (about 15% of the customers the models score) and a parameter everywhere it is used.
DEFAULT_OUTREACH_SIZE = 500
DEFAULT_HOLDOUT_FRACTION = 0.10

GROUPS = ("contact", "holdout", "not_selected")


def plan_outreach(
    scores: pd.DataFrame,
    k: int = DEFAULT_OUTREACH_SIZE,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Choose customers to contact. scores needs customer_id, margin_at_risk and
    margin_at_risk_after_cost; the result has those plus rank and group, one row per customer."""
    if k < 1:
        raise ValueError("The outreach size k must be at least 1")
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be at least 0 and below 1")
    if scores["customer_id"].duplicated().any():
        raise ValueError("Each customer may appear only once in the scores")

    plan = scores.sort_values(
        ["margin_at_risk", "customer_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    plan["rank"] = np.arange(1, len(plan) + 1)

    worth_contacting = plan["margin_at_risk_after_cost"] > 0
    selected = plan.index[(plan["rank"] <= k) & worth_contacting]

    plan["group"] = "not_selected"
    plan.loc[selected, "group"] = "contact"
    holdout_size = round(len(selected) * holdout_fraction)
    if holdout_fraction > 0 and len(selected) >= 2:
        holdout_size = max(1, holdout_size)  # a holdout of zero would defeat its purpose
    if holdout_size:
        picked = np.random.default_rng(seed).choice(selected, size=holdout_size, replace=False)
        plan.loc[picked, "group"] = "holdout"
    return plan


def summarize(plan: pd.DataFrame, contact_cost: float = config.CONTACT_COST_GBP) -> dict[str, Any]:
    """What a plan commits to: customers, cost and the margin at risk it covers."""
    contact = plan[plan["group"] == "contact"]
    holdout = plan[plan["group"] == "holdout"]
    return {
        "customers_scored": len(plan),
        "contact": len(contact),
        "holdout": len(holdout),
        "not_selected": int((plan["group"] == "not_selected").sum()),
        "cost": float(len(contact) * contact_cost),
        "margin_at_risk_contacted": float(contact["margin_at_risk"].sum()),
        "margin_at_risk_held_out": float(holdout["margin_at_risk"].sum()),
    }


def sensitivity(
    scores: pd.DataFrame,
    k: int,
    margins: Sequence[float],
    contact_costs: Sequence[float],
    baseline_margin: float = config.MARGIN,
    baseline_cost: float = config.CONTACT_COST_GBP,
) -> pd.DataFrame:
    """How the selected list changes when the margin and contact-cost assumptions change.

    scores needs customer_id, p_churn and spend_if_retained. Margin scales every customer's
    margin at risk by the same factor, so it cannot change the ranking, only which customers clear
    the contact cost; cost only moves that cutoff. overlap is the share of the baseline list that
    is still selected.
    """

    def selected_under(margin: float, cost: float) -> set[int]:
        at_risk = scores["p_churn"] * scores["spend_if_retained"] * margin
        frame = pd.DataFrame(
            {
                "customer_id": scores["customer_id"],
                "margin_at_risk": at_risk,
                "margin_at_risk_after_cost": at_risk - cost,
            }
        )
        plan = plan_outreach(frame, k, holdout_fraction=0.0)
        return set(plan.loc[plan["group"] == "contact", "customer_id"])

    baseline = selected_under(baseline_margin, baseline_cost)
    rows = []
    for margin in margins:
        for cost in contact_costs:
            chosen = selected_under(margin, cost)
            rows.append(
                {
                    "margin": margin,
                    "contact_cost": cost,
                    "selected": len(chosen),
                    "overlap_with_baseline": len(chosen & baseline) / len(baseline)
                    if baseline
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- database


def load_scores(
    conn: psycopg.Connection[Any], scoring_run_id: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The predictions of a scoring run (the latest by default) and the run's details."""
    run = conn.execute(
        """
        SELECT scoring_run_id, snapshot_date, margin, contact_cost, churn_model_version,
               value_model_version
        FROM ml.scoring_runs
        WHERE scoring_run_id = coalesce(%s, (SELECT max(scoring_run_id) FROM ml.scoring_runs))
        """,
        (scoring_run_id,),
    ).fetchone()
    conn.rollback()
    if run is None:
        raise EmptyTableError(
            "There are no scores yet. Score the customers first: python -m src.main --stage score"
        )
    scores = database.read_dataframe(
        conn,
        "SELECT customer_id, p_churn, spend_if_retained, expected_spend, margin_at_risk, "
        "margin_at_risk_after_cost FROM ml.predictions WHERE scoring_run_id = %s ORDER BY 1",
        (run[0],),
    )
    details = dict(
        zip(
            [
                "scoring_run_id",
                "snapshot_date",
                "margin",
                "contact_cost",
                "churn_model_version",
                "value_model_version",
            ],
            run,
            strict=True,
        )
    )
    return scores, details


def save_plan(
    conn: psycopg.Connection[Any],
    plan: pd.DataFrame,
    run: dict[str, Any],
    k: int,
    holdout_fraction: float,
    seed: int,
) -> int:
    """Store a plan so the holdout group is fixed and can be audited later."""
    counts = plan["group"].value_counts()
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO ml.outreach_plans
                (scoring_run_id, k, holdout_fraction, seed, margin, contact_cost,
                 contact_customers, holdout_customers)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING plan_id
            """,
            (
                run["scoring_run_id"],
                k,
                holdout_fraction,
                seed,
                run["margin"],
                run["contact_cost"],
                int(counts.get("contact", 0)),
                int(counts.get("holdout", 0)),
            ),
        ).fetchone()
        assert row is not None
        assignments = pd.DataFrame(
            {
                "plan_id": row[0],
                "customer_id": plan["customer_id"].astype(int),
                "rank": plan["rank"].astype(int),
                "grp": plan["group"],
            }
        )
        database.copy_dataframe(conn, assignments, "ml.outreach_assignments")
    conn.commit()
    return int(row[0])


def make_plan(
    conn: psycopg.Connection[Any],
    k: int = DEFAULT_OUTREACH_SIZE,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = config.RANDOM_STATE,
) -> tuple[int, pd.DataFrame]:
    """Plan outreach from the latest scores and store the plan. Returns its id and the plan."""
    scores, run = load_scores(conn)
    plan = plan_outreach(scores, k, holdout_fraction, seed)
    plan_id = save_plan(conn, plan, run, k, holdout_fraction, seed)
    summary = summarize(plan, run["contact_cost"])
    log.info(
        "Outreach plan %s from scoring run %s: contact %s customers (cost %.0f), hold out %s, "
        "margin at risk covered %.0f",
        plan_id,
        run["scoring_run_id"],
        summary["contact"],
        summary["cost"],
        summary["holdout"],
        summary["margin_at_risk_contacted"],
    )
    return plan_id, plan
