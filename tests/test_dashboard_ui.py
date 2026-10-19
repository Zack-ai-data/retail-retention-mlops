"""The Streamlit dashboard, driven the way a person would use it, against a fake API.

These need Streamlit, which only the dashboard image has, so they are skipped in the main quality
run. Run them with: docker compose run --rm dashboard-tests
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from src import dashboard_client as dc

APP = Path(__file__).resolve().parent.parent / "services" / "dashboard" / "app.py"

HEALTH = {
    "status": "ok",
    "models_loaded": True,
    "churn_model_version": "3",
    "value_model_version": "3",
    "database": "ok",
    "latest_scoring_run": {
        "scoring_run_id": 1,
        "scored_at": "2026-10-19 08:59:11.6+00",
        "snapshot_date": "2011-12-10",
    },
}

SCORE = {
    "source": "online",
    "customer_id": 1,
    "snapshot_date": "2011-12-10",
    "segment": "High value / Active",
    "p_churn": 0.70,
    "spend_if_retained": 900.0,
    "expected_spend": 270.0,
    "margin_at_risk": 189.0,
    "margin_at_risk_after_cost": 184.0,
    "worth_contacting": True,
    "churn_model_version": "3",
    "value_model_version": "3",
    "scored_at": None,
}

PLAN = {
    "plan_id": 1,
    "created_at": "2026-10-19",
    "scoring_run_id": 1,
    "k": 3,
    "holdout_fraction": 0.5,
    "seed": 42,
    "margin": 0.3,
    "contact_cost": 5.0,
    "contact_customers": 2,
    "holdout_customers": 1,
    "customers": [
        {
            "customer_id": 1,
            "rank": 1,
            "group": "contact",
            "p_churn": 0.7,
            "spend_if_retained": 900.0,
            "margin_at_risk": 189.0,
        },
        {
            "customer_id": 2,
            "rank": 2,
            "group": "holdout",
            "p_churn": 0.6,
            "spend_if_retained": 800.0,
            "margin_at_risk": 144.0,
        },
        {
            "customer_id": 3,
            "rank": 3,
            "group": "contact",
            "p_churn": 0.5,
            "spend_if_retained": 700.0,
            "margin_at_risk": 105.0,
        },
    ],
}

FEATURES = {
    "customer_id": 1,
    "snapshot_date": "2011-12-10",
    "features": {
        "recency_days": 10.0,
        "tenure_days_capped": 365.0,
        "history_days_available": 365.0,
        "frequency_365d": 4.0,
        "monetary_365d": 900.0,
        "avg_order_value_365d": 225.0,
        "total_units_365d": 100.0,
        "distinct_products_365d": 30.0,
        "avg_days_between_orders_365d": 60.0,
        "orders_last_90d": 2.0,
        "orders_prev_90d": 1.0,
        "spend_last_90d": 400.0,
        "return_invoices_365d": 0.0,
        "return_value_365d": 0.0,
        "return_rate_365d": 0.0,
        "is_uk": 1.0,
        "season_sin": 0.1,
        "season_cos": 0.9,
    },
}

SENSITIVITY = [
    {"margin": m, "contact_cost": c, "selected": 3, "overlap_with_baseline": 1.0}
    for m in (0.2, 0.3, 0.4)
    for c in (3.0, 5.0, 8.0)
]


def default_what_if(features):
    """Churn chance rises 1 point for every 5 days since the last order (a stand-in model)."""
    shift = (features["recency_days"] - 10.0) / 500
    return SCORE | {
        "source": "what-if",
        "p_churn": 0.70 + shift,
        "margin_at_risk": 189.0 * (1 + shift),
    }


def fake_api(**overrides):
    """A stand-in for ApiClient. Each override is a value to return or an exception to raise."""
    answers = {
        "health": HEALTH,
        "reload_models": {"churn_model_version": "3", "value_model_version": "3"},
        "outreach_plan": PLAN,
        "score": SCORE,
        "prediction": SCORE | {"source": "stored"},
        "customer_features": FEATURES,
        "sensitivity": SENSITIVITY,
        "what_if": default_what_if,
    } | overrides
    calls: list[tuple[str, tuple]] = []

    def respond(name):
        def method(self, *args, **kwargs):
            calls.append((name, args))
            answer = answers[name]
            if isinstance(answer, Exception):
                raise answer
            return answer(*args, **kwargs) if callable(answer) else answer

        return method

    attributes = {name: respond(name) for name in answers} | {
        "__init__": lambda self, *a, **k: None
    }
    api = type("FakeApi", (), attributes)
    api.calls = calls
    return api


def start(monkeypatch, **overrides):
    api = fake_api(**overrides)
    monkeypatch.setattr(dc, "ApiClient", api)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception, app.exception
    return app, api


def metrics(app):
    return {m.label: m.value for m in app.metric}


# ---------------------------------------------------------------- the page and the service


def test_the_sidebar_shows_the_models_and_the_latest_scoring_run(monkeypatch):
    app, _ = start(monkeypatch)

    success = [s.value for s in app.sidebar.success]
    assert any("Churn model v3" in s and "value model v3" in s for s in success)
    assert any("2011-12-10" in c.value for c in app.sidebar.caption)


def test_the_plan_page_shows_the_numbers_a_planner_needs(monkeypatch):
    app, _ = start(monkeypatch)

    shown = metrics(app)
    assert shown["Customers to contact"] == "2"
    assert shown["Held out (do not contact)"] == "1"
    assert shown["Contact cost"] == "£10"
    assert shown["Margin at risk covered"] == "£294"
    assert any("Do not contact them" in i.value for i in app.info)
    assert len(app.get("download_button")) == 1


def test_the_plan_table_can_be_filtered_by_group(monkeypatch):
    app, _ = start(monkeypatch)
    assert len(app.dataframe[0].value) == 2  # contact is the default view

    app.selectbox(key="plan_group").select("holdout").run()

    assert list(app.dataframe[0].value["customer_id"]) == [2]
    app.selectbox(key="plan_group").select("all").run()
    assert len(app.dataframe[0].value) == 3


def test_a_degraded_api_is_explained_in_the_sidebar(monkeypatch):
    degraded = {"status": "degraded", "models_loaded": False, "problem": "no champion yet"}

    app, _ = start(monkeypatch, health=degraded)

    assert any("no champion yet" in w.value for w in app.sidebar.warning)


def test_an_unreachable_api_stops_the_page_with_the_fix(monkeypatch):
    app, _ = start(
        monkeypatch,
        health=dc.ApiError("The API is not reachable. Start it with: docker compose up -d api"),
    )

    assert any("docker compose up -d api" in e.value for e in app.error)
    assert len(app.tabs) == 0  # nothing else is attempted


def test_without_a_plan_the_page_says_how_to_make_one(monkeypatch):
    app, _ = start(
        monkeypatch, outreach_plan=dc.ApiError("There is no outreach plan yet: --stage plan", 404)
    )

    assert any("--stage plan" in i.value for i in app.info)
    assert "Customers to contact" not in metrics(app)


# --------------------------------------------------------------------------- customer


def look_up(app, customer_id):
    app.number_input(key="customer_id").set_value(customer_id)
    app.button(key="lookup").click().run()
    return app


def test_looking_up_a_customer_shows_the_score_and_that_it_matches_the_stored_one(monkeypatch):
    app, api = start(monkeypatch)

    look_up(app, 1)

    shown = metrics(app)
    assert shown["Chance of churning"] == "70%"
    assert shown["Margin at risk"] == "£189"
    assert shown["Worth contacting"] == "Yes"
    assert any("agree" in s.value for s in app.success)
    assert ("score", (1,)) in api.calls and ("prediction", (1,)) in api.calls
    # numbers and model versions are in the same columns, so they are shown as text (a table
    # column with both cannot be sent to the browser as one type)
    comparison = next(d.value for d in app.dataframe if "measure" in d.value.columns)
    assert all(str(dtype) == "string" for dtype in comparison[["online", "stored"]].dtypes)
    assert "churn_model_version" in set(comparison["measure"])


def test_a_lapsed_customer_gets_the_apis_explanation_not_a_score(monkeypatch):
    refusal = dc.ApiError(
        "Customer 5 last bought 326 days before the snapshot: a win-back customer.", 422
    )
    app, _ = start(monkeypatch, score=refusal)

    look_up(app, 5)

    assert any("win-back" in w.value for w in app.warning)
    assert "Chance of churning" not in metrics(app)


def test_scores_that_differ_are_flagged_with_the_reason(monkeypatch):
    stale = SCORE | {"source": "stored", "p_churn": 0.5, "churn_model_version": "2"}
    app, _ = start(monkeypatch, prediction=stale)

    look_up(app, 1)

    assert any("retrained" in w.value for w in app.warning)


def test_a_customer_with_no_stored_score_is_still_shown(monkeypatch):
    app, _ = start(monkeypatch, prediction=dc.ApiError("No stored prediction", 404))

    look_up(app, 1)

    assert metrics(app)["Chance of churning"] == "70%"
    assert any("No stored score" in i.value for i in app.info)


# --------------------------------------------------------------------------- what-if


def test_changing_an_input_shows_how_the_score_moves(monkeypatch):
    app, api = start(monkeypatch)
    app.button(key="load_inputs").click().run()
    assert app.number_input(key="what_if_recency_days").value == 10.0

    app.number_input(key="what_if_recency_days").set_value(60.0).run()
    app.button(key="score_what_if").click().run()

    sent = [args[0] for name, args in api.calls if name == "what_if"]
    assert sent[0]["recency_days"] == 10.0 and sent[1]["recency_days"] == 60.0  # before, after
    assert sent[1]["tenure_days_capped"] == 365.0  # inputs not offered for editing are kept
    changed = {m.label: (m.value, m.delta) for m in app.metric if m.delta}
    assert changed["Chance of churning"] == ("80%", "+10.0 points")
    assert changed["Margin at risk"] == ("£208", "+19")


def test_what_if_for_a_customer_outside_the_population_explains_why(monkeypatch):
    app, _ = start(
        monkeypatch, customer_features=dc.ApiError("Customer 9 ... win-back customer", 422)
    )

    app.button(key="load_inputs").click().run()

    assert any("win-back" in w.value for w in app.warning)
    assert "what_if_recency_days" not in [n.key for n in app.number_input]  # nothing to edit


# ------------------------------------------------------------------------- assumptions


def test_the_assumptions_page_says_when_they_make_no_difference(monkeypatch):
    app, api = start(monkeypatch)

    assert any("same customers are selected" in s.value for s in app.success)
    _, args = next(c for c in api.calls if c[0] == "sensitivity")
    assert args[0] == 500 and args[1] == [0.2, 0.3, 0.4] and args[2] == [3.0, 5.0, 8.0]


def test_the_assumptions_page_explains_a_change_when_there_is_one(monkeypatch):
    changed = [r | {"overlap_with_baseline": 0.8} for r in SENSITIVITY]

    app, _ = start(monkeypatch, sensitivity=changed)

    assert any("never reorders customers" in i.value for i in app.info)


def test_choosing_no_margin_asks_for_one_instead_of_calling_the_api(monkeypatch):
    app, api = start(monkeypatch)
    before = len([c for c in api.calls if c[0] == "sensitivity"])

    app.multiselect(key="margins").set_value([]).run()

    assert any("Choose at least one" in i.value for i in app.info)
    assert len([c for c in api.calls if c[0] == "sensitivity"]) == before
