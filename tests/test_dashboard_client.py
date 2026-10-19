"""The dashboard's API client and the calculations behind its pages."""

import csv
import io
import json

import pytest
import requests

from src import dashboard_client as dc

# ----------------------------------------------------------------- a fake HTTP session


def reply(status=200, body=None, text=None):
    response = requests.Response()
    response.status_code = status
    response._content = (text if text is not None else json.dumps(body)).encode()
    return response


class FakeSession:
    """Returns canned replies and remembers what it was asked."""

    def __init__(self, *replies, error=None):
        self.replies = list(replies)
        self.error = error
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.replies.pop(0)


def client_with(*replies, error=None):
    session = FakeSession(*replies, error=error)
    return dc.ApiClient("http://api:8000/", session=session), session


# ---------------------------------------------------------------------- requests


def test_the_base_url_comes_from_the_argument_then_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://from-env:9000/")

    assert dc.ApiClient().base_url == "http://from-env:9000"
    assert dc.ApiClient("http://given:1/").base_url == "http://given:1"
    monkeypatch.delenv("API_URL")
    assert dc.ApiClient().base_url == dc.DEFAULT_API_URL


def test_each_call_asks_the_right_endpoint():
    client, session = client_with(*[reply(body={})] * 6)

    client.score(7)
    client.prediction(7)
    client.customer_features(7)
    client.what_if({"recency_days": 1.0})
    client.outreach_plan(group="contact", limit=10, offset=5)
    client.sensitivity(100, [0.2, 0.3], [5.0])

    assert [(m, u.removeprefix("http://api:8000")) for m, u, _ in session.calls] == [
        ("GET", "/customers/7/score"),
        ("GET", "/customers/7/prediction"),
        ("GET", "/customers/7/features"),
        ("POST", "/score"),
        ("GET", "/outreach/latest"),
        ("GET", "/outreach/sensitivity"),
    ]
    assert session.calls[3][2]["json"] == {"recency_days": 1.0}
    assert session.calls[4][2]["params"] == {"limit": 10, "offset": 5, "group": "contact"}
    assert session.calls[5][2]["params"] == {
        "k": 100,
        "margins": [0.2, 0.3],
        "contact_costs": [5.0],
    }


def test_a_customer_id_is_sent_as_an_integer_never_as_free_text():
    client, session = client_with(reply(body={}))

    client.score("12345")

    assert session.calls[0][1].endswith("/customers/12345/score")
    with pytest.raises(ValueError):
        client.score("1 OR 1=1")


def test_the_group_filter_is_left_out_when_not_given():
    client, session = client_with(reply(body={}))

    client.outreach_plan()

    assert "group" not in session.calls[0][2]["params"]


# ----------------------------------------------------------------------- errors


def test_an_unreachable_api_says_how_to_start_it():
    client, _ = client_with(error=requests.ConnectionError("refused"))

    with pytest.raises(
        dc.ApiError, match=r"not reachable at http://api:8000.*docker compose up -d api"
    ):
        client.score(1)


def test_a_slow_api_is_reported():
    client, _ = client_with(error=requests.Timeout("slow"))

    with pytest.raises(dc.ApiError, match="took too long"):
        client.score(1)


def test_an_explained_refusal_is_passed_on_in_the_apis_own_words():
    client, _ = client_with(reply(422, {"detail": "Customer 9 last bought 326 days before"}))

    with pytest.raises(dc.ApiError, match="326 days") as caught:
        client.score(9)
    assert caught.value.status == 422


def test_validation_errors_name_the_field():
    body = {"detail": [{"loc": ["body", "recency_days"], "msg": "Input should be >= 0"}]}
    client, _ = client_with(reply(422, body))

    with pytest.raises(dc.ApiError, match="recency_days: Input should be >= 0"):
        client.what_if({"recency_days": -1.0})


def test_a_reply_that_is_not_json_still_gives_a_message():
    client, _ = client_with(reply(502, text="Bad gateway"))

    with pytest.raises(dc.ApiError, match="Bad gateway"):
        client.score(1)


def test_a_degraded_api_is_reported_by_health_not_raised():
    detail = {"status": "degraded", "models_loaded": False, "problem": "no champion"}
    client, _ = client_with(reply(503, {"detail": detail}))

    assert client.health() == detail


def test_health_raises_for_other_failures():
    client, _ = client_with(reply(500, {"detail": "boom"}))

    with pytest.raises(dc.ApiError, match="boom"):
        client.health()


# ---------------------------------------------------------------- page calculations


def plan(rows=None):
    rows = rows or [
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
        {
            "customer_id": 4,
            "rank": 4,
            "group": "not_selected",
            "p_churn": 0.4,
            "spend_if_retained": 100.0,
            "margin_at_risk": 12.0,
        },
    ]
    return {
        "contact_customers": 2,
        "holdout_customers": 1,
        "contact_cost": 5.0,
        "customers": rows,
    }


def parse(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def test_the_contact_list_never_contains_the_holdout_or_unselected_customers():
    rows = parse(dc.contact_list_csv(plan()))

    assert [r["customer_id"] for r in rows] == ["1", "3"]


def test_the_contact_list_is_in_rank_order_with_the_agreed_columns():
    shuffled = plan(list(reversed(plan()["customers"])))

    data = dc.contact_list_csv(shuffled)

    assert data.decode().splitlines()[0] == ",".join(dc.CONTACT_LIST_COLUMNS)
    assert [r["rank"] for r in parse(data)] == ["1", "3"]


def test_an_empty_contact_list_is_just_the_header():
    only_holdout = plan([plan()["customers"][1]])

    assert parse(dc.contact_list_csv(only_holdout)) == []


def test_plan_totals_state_cost_and_margin_at_risk():
    totals = dc.plan_totals(plan())

    assert totals["contact"] == 2 and totals["holdout"] == 1
    assert totals["cost"] == 10.0
    assert totals["margin_at_risk_contacted"] == pytest.approx(294.0)
    assert totals["margin_at_risk_held_out"] == pytest.approx(144.0)
    assert totals["complete"] is True


def test_plan_totals_warn_when_the_rows_given_are_fewer_than_the_plan_selected():
    partial = plan(plan()["customers"][:1])  # the plan says 2 contacts and 1 holdout

    assert dc.plan_totals(partial)["complete"] is False


def scores(**changes):
    base = dict.fromkeys(dc.SCORE_FIELDS, 1.0) | {
        "churn_model_version": "3",
        "value_model_version": "3",
    }
    return base | changes


def test_scores_that_match_are_marked_as_agreeing():
    table = dc.compare_scores(scores(), scores())

    assert table["agree"].all() and len(table) == 7


def test_a_difference_is_marked_and_the_versions_that_explain_it_are_shown():
    table = dc.compare_scores(scores(p_churn=0.5, churn_model_version="4"), scores()).set_index(
        "measure"
    )

    assert not table.loc["p_churn", "agree"] and not table.loc["churn_model_version", "agree"]
    assert table.loc["margin_at_risk", "agree"]


def test_tiny_float_noise_is_not_a_disagreement():
    table = dc.compare_scores(scores(p_churn=0.7000000000001), scores(p_churn=0.7))

    assert table["agree"].all()


def sensitivity_rows(overlap=1.0):
    return [
        {"margin": m, "contact_cost": c, "selected": 100, "overlap_with_baseline": overlap}
        for m in (0.2, 0.3)
        for c in (3.0, 5.0, 8.0)
    ]


def test_the_sensitivity_is_laid_out_as_margin_by_contact_cost():
    tables = dc.sensitivity_tables(sensitivity_rows())

    assert tables["selected"].shape == (2, 3)
    assert list(tables["selected"].index) == [0.2, 0.3]
    assert list(tables["selected"].columns) == [3.0, 5.0, 8.0]


def test_it_recognises_when_the_assumptions_make_no_difference_to_the_list():
    assert dc.assumptions_make_no_difference(sensitivity_rows(1.0)) is True
    assert dc.assumptions_make_no_difference(sensitivity_rows(0.9)) is False
    assert dc.assumptions_make_no_difference([]) is False


def test_money_and_percent_are_formatted_for_reading():
    assert dc.money(83271.4) == "£83,271"
    assert dc.percent(0.6299) == "63%"
