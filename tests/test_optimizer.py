"""The outreach optimizer: ranking, the contact-cost cutoff, the random holdout and saved plans."""

import numpy as np
import pandas as pd
import pytest

from src import config, optimizer
from src.errors import EmptyTableError


def make_scores(n=200, seed=0, margin=0.30, cost=5.0):
    rng = np.random.default_rng(seed)
    p_churn = rng.uniform(0.05, 0.95, n)
    spend = rng.lognormal(5, 1, n)
    at_risk = p_churn * spend * margin
    return pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "p_churn": p_churn,
            "spend_if_retained": spend,
            "expected_spend": spend * (1 - p_churn),
            "margin_at_risk": at_risk,
            "margin_at_risk_after_cost": at_risk - cost,
        }
    )


def groups(plan):
    return {g: set(plan.loc[plan["group"] == g, "customer_id"]) for g in optimizer.GROUPS}


# ---------------------------------------------------------------- selection


def test_customers_are_ranked_by_margin_at_risk_highest_first():
    plan = optimizer.plan_outreach(make_scores(), k=20)

    assert plan["rank"].tolist() == list(range(1, 201))
    assert plan["margin_at_risk"].is_monotonic_decreasing


def test_equal_margins_at_risk_are_ranked_by_customer_id_so_plans_are_reproducible():
    scores = make_scores(10)
    scores["margin_at_risk"] = 100.0
    scores["margin_at_risk_after_cost"] = 95.0
    shuffled = scores.sample(frac=1, random_state=3)

    plan = optimizer.plan_outreach(shuffled, k=10, holdout_fraction=0)

    assert plan["customer_id"].tolist() == sorted(scores["customer_id"])


def test_the_top_k_are_selected_and_everyone_else_is_not():
    scores = make_scores()
    plan = optimizer.plan_outreach(scores, k=50, holdout_fraction=0.1)

    selected = groups(plan)["contact"] | groups(plan)["holdout"]
    top_50 = set(scores.nlargest(50, "margin_at_risk")["customer_id"])
    assert selected == top_50
    assert len(groups(plan)["not_selected"]) == 150


def test_customers_not_worth_the_contact_cost_are_never_selected():
    scores = make_scores()
    scores.loc[scores["margin_at_risk"].nlargest(3).index, "margin_at_risk_after_cost"] = -1.0
    unprofitable = set(scores.loc[scores["margin_at_risk_after_cost"] <= 0, "customer_id"])

    plan = optimizer.plan_outreach(scores, k=len(scores))

    assert not (unprofitable & (groups(plan)["contact"] | groups(plan)["holdout"]))
    assert unprofitable <= groups(plan)["not_selected"]


def test_the_budget_is_not_filled_with_customers_who_cannot_pay_for_it():
    scores = make_scores(50)
    worth_it = int((scores["margin_at_risk_after_cost"] > 0).sum())
    assert 0 < worth_it < 50

    plan = optimizer.plan_outreach(scores, k=50)

    assert len(groups(plan)["contact"]) + len(groups(plan)["holdout"]) == worth_it


def test_a_customer_exactly_at_the_contact_cost_is_not_worth_contacting():
    scores = make_scores(5)
    scores["margin_at_risk"] = [50.0, 40.0, 30.0, 20.0, 5.0]
    scores["margin_at_risk_after_cost"] = scores["margin_at_risk"] - 5.0  # the last is exactly 0

    plan = optimizer.plan_outreach(scores, k=5, holdout_fraction=0)

    assert len(groups(plan)["contact"]) == 4


def test_the_input_scores_are_not_modified():
    scores = make_scores()
    before = scores.copy()

    optimizer.plan_outreach(scores, k=30)

    pd.testing.assert_frame_equal(scores, before)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0}, "at least 1"),
        ({"k": 5, "holdout_fraction": 1.0}, "below 1"),
        ({"k": 5, "holdout_fraction": -0.1}, "at least 0"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        optimizer.plan_outreach(make_scores(), **kwargs)


def test_a_customer_appearing_twice_is_rejected():
    scores = pd.concat([make_scores(5), make_scores(5)])

    with pytest.raises(ValueError, match="only once"):
        optimizer.plan_outreach(scores, k=5)


# ----------------------------------------------------------------- holdout


def test_the_holdout_is_the_requested_share_of_the_selected_customers():
    plan = optimizer.plan_outreach(make_scores(), k=100, holdout_fraction=0.10)

    selected = len(groups(plan)["contact"]) + len(groups(plan)["holdout"])
    assert len(groups(plan)["holdout"]) == round(selected * 0.10)


def test_the_holdout_is_fixed_by_the_seed():
    scores = make_scores()

    a = groups(optimizer.plan_outreach(scores, k=100, seed=7))["holdout"]
    b = groups(optimizer.plan_outreach(scores, k=100, seed=7))["holdout"]
    c = groups(optimizer.plan_outreach(scores, k=100, seed=8))["holdout"]

    assert a == b and a != c


def test_the_holdout_is_drawn_at_random_across_ranks_not_from_the_bottom_of_the_list():
    scores = make_scores(400)
    mean_ranks = []
    for seed in range(40):
        plan = optimizer.plan_outreach(scores, k=200, holdout_fraction=0.2, seed=seed)
        mean_ranks.append(plan.loc[plan["group"] == "holdout", "rank"].mean())

    # selected customers have ranks 1..~200 (mean ~100): a random sample has the same mean
    assert np.mean(mean_ranks) == pytest.approx(
        optimizer.plan_outreach(scores, k=200)["rank"].iloc[:200].mean(), rel=0.05
    )


def test_a_holdout_is_never_empty_when_one_is_asked_for():
    plan = optimizer.plan_outreach(
        make_scores(), k=5, holdout_fraction=0.10
    )  # 10% of 5 rounds to 0

    assert len(groups(plan)["holdout"]) == 1


def test_no_holdout_when_none_is_asked_for_or_only_one_customer_is_selected():
    assert (
        groups(optimizer.plan_outreach(make_scores(), k=50, holdout_fraction=0))["holdout"] == set()
    )
    assert (
        groups(optimizer.plan_outreach(make_scores(), k=1, holdout_fraction=0.5))["holdout"]
        == set()
    )


def test_holdout_customers_come_only_from_the_selected_list():
    scores = make_scores()
    plan = optimizer.plan_outreach(scores, k=60, holdout_fraction=0.25)

    top = set(scores.nlargest(60, "margin_at_risk")["customer_id"])
    assert groups(plan)["holdout"] <= top


# -------------------------------------------------------- summary, sensitivity


def test_the_summary_states_the_cost_and_the_margin_at_risk_covered():
    plan = optimizer.plan_outreach(make_scores(), k=50, holdout_fraction=0.10)

    summary = optimizer.summarize(plan, contact_cost=5.0)

    contact = plan[plan["group"] == "contact"]
    assert summary["contact"] == len(contact) and summary["holdout"] == 5
    assert summary["cost"] == pytest.approx(5.0 * len(contact))
    assert summary["margin_at_risk_contacted"] == pytest.approx(contact["margin_at_risk"].sum())
    assert summary["customers_scored"] == 200
    assert summary["contact"] + summary["holdout"] + summary["not_selected"] == 200


def test_the_sensitivity_baseline_overlaps_itself_completely():
    result = optimizer.sensitivity(make_scores(), k=50, margins=[0.3], contact_costs=[5.0])

    assert result["overlap_with_baseline"].tolist() == [1.0]


def test_a_higher_contact_cost_or_a_lower_margin_shrinks_the_list_when_the_budget_is_not_binding():
    scores = make_scores()
    result = optimizer.sensitivity(
        scores, k=len(scores), margins=[0.15, 0.30, 0.60], contact_costs=[2.0, 5.0, 20.0]
    ).set_index(["margin", "contact_cost"])["selected"]

    assert result[(0.30, 2.0)] >= result[(0.30, 5.0)] >= result[(0.30, 20.0)]
    assert result[(0.15, 5.0)] <= result[(0.30, 5.0)] <= result[(0.60, 5.0)]


def test_margin_alone_never_reorders_customers_only_the_cutoff_moves():
    scores = make_scores()
    result = optimizer.sensitivity(scores, k=60, margins=[0.20, 0.60], contact_costs=[0.0])

    # every selected customer under the smaller margin is also selected under the larger one
    small, large = result["selected"].tolist()
    assert small <= large
    assert result["overlap_with_baseline"].between(0, 1).all()


# ----------------------------------------------------------------- database


def insert_run(conn, scores, margin=0.30, cost=5.0):
    run_id = conn.execute(
        "INSERT INTO ml.scoring_runs (snapshot_date, churn_model_version, value_model_version, "
        "margin, contact_cost, customers) VALUES ('2011-12-10', '3', '4', %s, %s, %s) "
        "RETURNING scoring_run_id",
        (margin, cost, len(scores)),
    ).fetchone()[0]
    for row in scores.itertuples():
        conn.execute(
            "INSERT INTO ml.predictions VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                int(row.customer_id),
                float(row.p_churn),
                float(row.spend_if_retained),
                float(row.expected_spend),
                float(row.margin_at_risk),
                float(row.margin_at_risk_after_cost),
            ),
        )
    conn.commit()
    return run_id


def test_planning_before_scoring_says_to_score(scratch_conn):
    with pytest.raises(EmptyTableError, match="--stage score"):
        optimizer.make_plan(scratch_conn)


def test_a_plan_is_stored_with_every_customers_group(scratch_conn):
    scores = make_scores(120)
    run_id = insert_run(scratch_conn, scores)

    plan_id, plan = optimizer.make_plan(scratch_conn, k=40, holdout_fraction=0.10, seed=5)

    stored = dict(
        scratch_conn.execute(
            "SELECT customer_id, grp FROM ml.outreach_assignments WHERE plan_id = %s", (plan_id,)
        ).fetchall()
    )
    assert stored == dict(zip(plan["customer_id"], plan["group"], strict=True))
    header = scratch_conn.execute(
        "SELECT scoring_run_id, k, holdout_fraction, seed, margin, contact_cost, "
        "contact_customers, holdout_customers FROM ml.outreach_plans WHERE plan_id = %s",
        (plan_id,),
    ).fetchone()
    contact, holdout = (plan["group"] == "contact").sum(), (plan["group"] == "holdout").sum()
    assert header == (run_id, 40, 0.10, 5, 0.30, 5.0, contact, holdout)


def test_the_same_settings_give_the_same_holdout_every_time(scratch_conn):
    insert_run(scratch_conn, make_scores(150))

    _, first = optimizer.make_plan(scratch_conn, k=60, seed=11)
    _, second = optimizer.make_plan(scratch_conn, k=60, seed=11)

    assert groups(first) == groups(second)


def test_planning_uses_the_latest_scoring_run(scratch_conn):
    insert_run(scratch_conn, make_scores(30, seed=1))
    latest = insert_run(scratch_conn, make_scores(60, seed=2))

    plan_id, plan = optimizer.make_plan(scratch_conn, k=10)

    assert len(plan) == 60
    assert (
        scratch_conn.execute(
            "SELECT scoring_run_id FROM ml.outreach_plans WHERE plan_id = %s", (plan_id,)
        ).fetchone()[0]
        == latest
    )


def test_load_scores_can_read_an_earlier_run(scratch_conn):
    first = insert_run(scratch_conn, make_scores(30, seed=1))
    insert_run(scratch_conn, make_scores(60, seed=2))

    scores, run = optimizer.load_scores(scratch_conn, first)

    assert len(scores) == 30 and run["scoring_run_id"] == first


def test_the_plan_uses_the_assumptions_the_scores_were_computed_with(scratch_conn):
    insert_run(scratch_conn, make_scores(50, margin=0.5, cost=8.0), margin=0.5, cost=8.0)

    plan_id, _ = optimizer.make_plan(scratch_conn, k=20)

    margin, cost = scratch_conn.execute(
        "SELECT margin, contact_cost FROM ml.outreach_plans WHERE plan_id = %s", (plan_id,)
    ).fetchone()
    assert (margin, cost) == (0.5, 8.0)


def test_the_default_outreach_size_and_holdout_are_the_documented_assumptions():
    assert optimizer.DEFAULT_OUTREACH_SIZE == 500
    assert optimizer.DEFAULT_HOLDOUT_FRACTION == 0.10
    assert config.CONTACT_COST_GBP == 5.0
