"""The HTTP API: answers, errors, and that stored and online scores agree."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api, config, features, optimizer, processor, scoring
from src.errors import RegistryError
from tests.test_scoring import RAW, stub_champions

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def db(scratch_conn, insert_raw, monkeypatch):
    """A database with marts built from RAW; the API is pointed at it."""
    insert_raw(RAW)
    processor.build_staging(scratch_conn)
    processor.build_marts(scratch_conn)
    monkeypatch.setattr(config, "POSTGRES_DB", scratch_conn.info.dbname)
    monkeypatch.setattr(scoring, "load_champions", lambda client: stub_champions())
    return scratch_conn


@pytest.fixture
def scored(db):
    scoring.score_batch(db, client=None)
    db.execute(
        "INSERT INTO marts.customer_segments "
        "VALUES ('2011-12-10', 1, 0, 'High value / Active', '3')"
    )
    db.commit()
    return db


@pytest.fixture
def client(db):
    with TestClient(api.create_app(stub_champions)) as test_client:
        yield test_client


def _plan(conn):
    # the stand-in models give both customers less margin at risk than the contact cost, so the
    # optimizer would rightly select nobody; give them realistic stored scores
    conn.execute(
        "UPDATE ml.predictions SET margin_at_risk = CASE customer_id WHEN 1 THEN 100 ELSE 80 END, "
        "margin_at_risk_after_cost = CASE customer_id WHEN 1 THEN 95 ELSE 75 END"
    )
    conn.commit()
    return optimizer.make_plan(conn, k=2, holdout_fraction=0.5)


# ---------------------------------------------------------- the request contract


def test_the_request_has_exactly_the_model_inputs():
    assert list(api.FeaturesRequest.model_fields) == features.BASE_FEATURES


def test_only_features_that_can_be_undefined_may_be_null():
    nullable = {n for n, f in api.FeaturesRequest.model_fields.items() if not f.is_required()}

    assert nullable == {"avg_order_value_365d", "avg_days_between_orders_365d", "return_rate_365d"}


# --------------------------------------------------------------------- health


def test_health_reports_the_models_the_database_and_the_latest_scoring_run(client, scored):
    body = client.get("/health").json()

    assert body["status"] == "ok" and body["database"] == "ok"
    assert (body["churn_model_version"], body["value_model_version"]) == ("7", "9")
    assert body["latest_scoring_run"]["snapshot_date"] == "2011-12-10"


def test_health_is_ok_before_any_scoring_has_run(client):
    body = client.get("/health").json()

    assert body["latest_scoring_run"] is None


def test_without_models_health_says_degraded_and_why(db):
    def no_models():
        raise RegistryError("There is no champion for 'retail-churn-classifier' yet.")

    with TestClient(api.create_app(no_models)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["models_loaded"] is False and "no champion" in detail["problem"]


def test_scoring_endpoints_explain_that_the_models_are_not_loaded(scored):
    def no_models():
        raise RegistryError("registry down")

    with TestClient(api.create_app(no_models)) as client:
        online = client.get("/customers/1/score")
        what_if = client.post("/score", json={})

    assert online.status_code == 503 and "registry down" in online.json()["detail"]
    assert what_if.status_code in (422, 503)  # validation of the body may come first


def test_models_can_be_reloaded_without_restarting(scored):
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RegistryError("registry down")
        return stub_champions()

    with TestClient(api.create_app(flaky)) as client:
        assert client.get("/customers/1/score").status_code == 503

        reloaded = client.post("/models/reload")

        assert reloaded.json() == {"churn_model_version": "7", "value_model_version": "9"}
        assert client.get("/customers/1/score").status_code == 200


def test_reload_reports_when_the_models_still_cannot_be_loaded(scored):
    def no_models():
        raise RegistryError("still down")

    with TestClient(api.create_app(no_models)) as client:
        response = client.post("/models/reload")

    assert response.status_code == 503 and "still down" in response.json()["detail"]


# ----------------------------------------------------------------- online score


def test_a_customer_is_scored_online_with_the_live_models(client, scored):
    response = client.get("/customers/1/score")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "online" and body["customer_id"] == 1
    assert body["snapshot_date"] == "2011-12-10"
    assert body["segment"] == "High value / Active"
    assert (body["churn_model_version"], body["value_model_version"]) == ("7", "9")
    snapshot = features.load_score_customer(scored, 1)
    expected = scoring.score_features(snapshot, stub_champions()).iloc[0]
    assert body["p_churn"] == pytest.approx(expected["p_churn"])
    assert body["margin_at_risk"] == pytest.approx(expected["margin_at_risk"])
    assert body["worth_contacting"] is bool(expected["margin_at_risk_after_cost"] > 0)


def test_an_unknown_customer_is_not_found(client, scored):
    response = client.get("/customers/999999/score")

    assert response.status_code == 404
    assert "not in the latest snapshot" in response.json()["detail"]


def test_a_lapsed_customer_gets_an_explanation_not_a_score(client, scored):
    response = client.get("/customers/3/score")  # last bought 2010-06-01

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "outside the population" in detail and "win-back" in detail and "180" in detail


@pytest.mark.parametrize("bad_id", ["abc", "1.5", "1%20OR%201=1", "1;DROP TABLE x"])
def test_a_customer_id_that_is_not_an_integer_is_rejected_before_it_reaches_the_database(
    client, scored, bad_id
):
    assert client.get(f"/customers/{bad_id}/score").status_code == 422


def test_a_huge_customer_id_is_handled_not_crashed(client, scored):
    assert client.get("/customers/99999999999999999999/score").status_code in (404, 422)


# --------------------------------------------------------------- stored prediction


def test_the_stored_prediction_is_what_batch_scoring_wrote(client, scored):
    body = client.get("/customers/1/prediction").json()

    assert body["source"] == "stored" and body["scored_at"]
    row = scored.execute(
        "SELECT p_churn, margin_at_risk FROM ml.latest_predictions WHERE customer_id = 1"
    ).fetchone()
    assert (body["p_churn"], body["margin_at_risk"]) == pytest.approx(row)


def test_online_and_stored_scores_agree_for_the_same_model_versions(client, scored):
    """The consistency check: the table and the API must not give different answers."""
    for customer_id in (1, 2):
        online = client.get(f"/customers/{customer_id}/score").json()
        stored = client.get(f"/customers/{customer_id}/prediction").json()

        for field in (
            "p_churn",
            "spend_if_retained",
            "expected_spend",
            "margin_at_risk",
            "margin_at_risk_after_cost",
        ):
            assert online[field] == pytest.approx(stored[field]), (customer_id, field)
        assert online["churn_model_version"] == stored["churn_model_version"]
        assert online["value_model_version"] == stored["value_model_version"]
        assert online["worth_contacting"] == stored["worth_contacting"]


def test_no_stored_prediction_for_a_customer_outside_the_population(client, scored):
    response = client.get("/customers/3/prediction")

    assert response.status_code == 404
    assert "outside the models' population" in response.json()["detail"]


# ---------------------------------------------------------------------- what-if


def _features_of(conn, customer_id):
    row = features.load_score_customer(conn, customer_id).iloc[0]
    return {
        name: (None if pd.isna(row[name]) else float(row[name])) for name in features.BASE_FEATURES
    }


def test_what_if_scoring_matches_the_online_score_for_the_same_features(client, scored):
    payload = _features_of(scored, 1)

    what_if = client.post("/score", json=payload).json()
    online = client.get("/customers/1/score").json()

    assert what_if["source"] == "what-if" and what_if["customer_id"] is None
    assert what_if["p_churn"] == pytest.approx(online["p_churn"])
    assert what_if["margin_at_risk"] == pytest.approx(online["margin_at_risk"])


def test_changing_an_input_changes_the_score(client, scored):
    payload = _features_of(scored, 1)

    base = client.post("/score", json=payload).json()["p_churn"]
    payload["recency_days"] = 150.0
    later = client.post("/score", json=payload).json()["p_churn"]

    assert later > base  # the stand-in model's churn probability rises with days since the order


def test_features_that_are_undefined_may_be_null(client, scored):
    payload = _features_of(scored, 1)
    payload["avg_days_between_orders_365d"] = None
    payload["avg_order_value_365d"] = None
    payload["return_rate_365d"] = None

    assert client.post("/score", json=payload).status_code == 200


def test_a_missing_feature_is_rejected_and_named(client, scored):
    payload = _features_of(scored, 1)
    del payload["recency_days"]

    response = client.post("/score", json=payload)

    assert response.status_code == 422 and "recency_days" in response.text


def test_an_unknown_field_is_rejected(client, scored):
    payload = _features_of(scored, 1) | {"customer_name": "someone"}

    assert client.post("/score", json=payload).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [("is_uk", 2), ("recency_days", -1), ("season_sin", 3), ("history_days_available", 400)],
)
def test_impossible_values_are_rejected(client, scored, field, value):
    payload = _features_of(scored, 1) | {field: value}

    assert client.post("/score", json=payload).status_code == 422


def test_text_where_a_number_belongs_is_rejected(client, scored):
    payload = _features_of(scored, 1) | {"recency_days": "yesterday"}

    assert client.post("/score", json=payload).status_code == 422


# ---------------------------------------------------------------------- outreach


def test_there_is_no_plan_until_one_is_made(client, scored):
    response = client.get("/outreach/latest")

    assert response.status_code == 404 and "--stage plan" in response.json()["detail"]


def test_the_latest_plan_lists_customers_in_rank_order_with_their_scores(client, scored):
    plan_id, plan = _plan(scored)

    body = client.get("/outreach/latest").json()

    assert body["plan_id"] == plan_id and body["k"] == 2 and body["holdout_fraction"] == 0.5
    assert [c["rank"] for c in body["customers"]] == sorted(c["rank"] for c in body["customers"])
    assert {c["customer_id"] for c in body["customers"]} == set(plan["customer_id"])
    assert all(
        c["p_churn"] is not None and c["margin_at_risk"] is not None for c in body["customers"]
    )


def test_the_plan_can_be_filtered_to_one_group_and_paged(client, scored):
    _plan(scored)

    everyone = client.get("/outreach/latest").json()["customers"]
    held_out = client.get("/outreach/latest?group=holdout").json()["customers"]
    second = client.get("/outreach/latest?limit=1&offset=1").json()["customers"]

    assert held_out and all(c["group"] == "holdout" for c in held_out)
    assert len(held_out) < len(everyone)
    assert [c["customer_id"] for c in second] == [everyone[1]["customer_id"]]


def test_an_unknown_group_is_rejected(client, scored):
    _plan(scored)

    assert client.get("/outreach/latest?group=everyone").status_code == 422


@pytest.mark.parametrize("query", ["limit=0", "limit=100000", "offset=-1"])
def test_paging_limits_are_enforced(client, scored, query):
    _plan(scored)

    assert client.get(f"/outreach/latest?{query}").status_code == 422


def test_the_sensitivity_covers_every_margin_and_cost_asked_for(client, scored):
    response = client.get(
        "/outreach/sensitivity?k=2&margins=0.2&margins=0.4&contact_costs=1&contact_costs=5&contact_costs=9"
    )

    rows = response.json()
    assert len(rows) == 6
    assert {r["margin"] for r in rows} == {0.2, 0.4} and {r["contact_cost"] for r in rows} == {
        1,
        5,
        9,
    }
    assert all(0 <= r["overlap_with_baseline"] <= 1 for r in rows)


def test_the_sensitivity_has_sensible_defaults(client, scored):
    assert len(client.get("/outreach/sensitivity").json()) == 9


def test_the_sensitivity_before_any_scoring_says_to_score(client):
    response = client.get("/outreach/sensitivity")

    assert response.status_code == 404 and "--stage score" in response.json()["detail"]


@pytest.mark.parametrize("query", ["margins=0", "margins=-0.1", "contact_costs=-1", "k=0"])
def test_nonsense_assumptions_are_rejected(client, scored, query):
    assert client.get(f"/outreach/sensitivity?{query}").status_code == 422


# -------------------------------------------------------------------- robustness


def test_when_the_database_is_down_the_api_says_so_without_leaking_the_password(
    scored, monkeypatch
):
    with TestClient(api.create_app(stub_champions)) as client:
        monkeypatch.setattr(config, "POSTGRES_DB", "no_such_database")
        monkeypatch.setattr(config, "POSTGRES_PASSWORD", "s3cret-do-not-print")

        response = client.get("/customers/1/score")

    assert response.status_code == 503
    assert "s3cret-do-not-print" not in response.text


def test_the_api_describes_itself(client):
    schema = client.get("/openapi.json").json()

    assert {"/health", "/customers/{customer_id}/score", "/score", "/outreach/latest"} <= set(
        schema["paths"]
    )


# --------------------------------------------------------------- customer features


def test_a_customers_features_can_be_fetched_as_a_what_if_starting_point(client, scored):
    body = client.get("/customers/1/features").json()

    assert body["customer_id"] == 1 and body["snapshot_date"] == "2011-12-10"
    assert list(body["features"]) == features.BASE_FEATURES
    # ...and sent straight back to /score, it gives the customer's own score
    what_if = client.post("/score", json=body["features"]).json()
    online = client.get("/customers/1/score").json()
    assert what_if["p_churn"] == pytest.approx(online["p_churn"])


def test_features_are_not_given_for_unknown_or_lapsed_customers(client, scored):
    assert client.get("/customers/999999/features").status_code == 404
    assert client.get("/customers/3/features").status_code == 422
