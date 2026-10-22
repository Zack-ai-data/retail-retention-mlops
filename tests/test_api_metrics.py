"""The API's Prometheus metrics: what is counted, how it is labelled, and the model versions."""

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src import api, metrics
from src.errors import RegistryError
from tests.test_api import client as db_client  # noqa: F401  (a client on a database with data)
from tests.test_api import db  # noqa: F401  (the fixture db_client is built on)
from tests.test_scoring import stub_champions


def _samples(test_client):
    text = test_client.get("/metrics").text
    return [s for family in text_string_to_metric_families(text) for s in family.samples]


def _value(test_client, name, **labels):
    """One sample's value, or None if there is no such sample."""
    for sample in _samples(test_client):
        if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    return None


def _app():
    """An API with stand-in models, plus two routes defined here so no database is needed."""
    app = api.create_app(stub_champions)

    @app.get("/items/{item_id}")
    def item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("deliberate")

    return app


@pytest.fixture
def plain():
    """An API with stand-in models and no database (for requests that do not need one)."""
    with TestClient(_app(), raise_server_exceptions=False) as test_client:
        yield test_client


# ----------------------------------------------------------------------- the endpoint


def test_metrics_are_served_in_the_prometheus_text_format(plain):
    response = plain.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE retail_api_requests_total counter" in response.text


def test_the_metrics_endpoint_is_not_part_of_the_documented_api(plain):
    assert "/metrics" not in plain.get("/openapi.json").json()["paths"]


def test_scraping_does_not_count_itself(plain):
    plain.get("/metrics")
    plain.get("/metrics")

    assert not [s for s in _samples(plain) if s.labels.get("route") == "/metrics"]


# ------------------------------------------------------------------------ requests


def test_a_real_route_is_counted_under_its_template_not_the_customer_url(db_client):  # noqa: F811
    db_client.get("/customers/1/features")

    assert (
        _value(
            db_client,
            "retail_api_requests_total",
            method="GET",
            route="/customers/{customer_id}/features",
            status="200",
        )
        == 1
    )


def test_values_in_the_url_never_appear_in_the_metrics(plain):
    plain.get("/items/987654")

    assert _value(plain, "retail_api_requests_total", route="/items/{item_id}", status="200") == 1
    assert "987654" not in plain.get("/metrics").text


def test_a_request_that_matches_no_route_is_counted_as_unmatched(plain):
    plain.get("/no/such/path/abc123")

    assert _value(plain, "retail_api_requests_total", route=metrics.UNMATCHED, status="404") == 1
    assert "abc123" not in plain.get("/metrics").text  # the path would be unbounded cardinality


def test_error_responses_are_counted_with_their_status(plain):
    plain.get("/items/not-a-number")  # fails validation

    assert _value(plain, "retail_api_requests_total", route="/items/{item_id}", status="422") == 1


def test_an_unhandled_error_is_counted_as_a_500_and_not_lost(plain):
    assert plain.get("/boom").status_code == 500

    assert _value(plain, "retail_api_requests_total", route="/boom", status="500") == 1


def test_every_request_is_timed(plain):
    plain.get("/no/such/path")
    plain.get("/no/such/path")

    assert _value(plain, "retail_api_request_duration_seconds_count", route="unmatched") == 2
    assert (
        _value(plain, "retail_api_request_duration_seconds_bucket", route="unmatched", le="10.0")
        == 2
    )


def test_no_request_is_left_in_progress_after_the_answers_are_sent(plain):
    plain.get("/no/such/path")
    plain.get("/items/not-a-number")
    plain.get("/boom")

    assert _value(plain, "retail_api_requests_in_progress") == 0


# ------------------------------------------------------------------ model versions


def test_the_versions_being_served_are_exposed(plain):
    # stand-in champions are churn 7 and value 9
    assert _value(plain, "retail_api_model_version", model="churn") == 7
    assert _value(plain, "retail_api_model_version", model="value") == 9
    assert _value(plain, "retail_api_model_loads_total", result="ok") == 1


def test_a_reload_that_picks_up_new_champions_changes_the_exposed_versions():
    current = {"churn": "7", "value": "9"}

    def loader():
        champions = stub_champions()
        return type(champions)(
            champions.churn_model,
            champions.value_model,
            current["churn"],
            current["value"],
        )

    with TestClient(api.create_app(loader)) as test_client:
        assert _value(test_client, "retail_api_model_version", model="churn") == 7
        current.update(churn="8")  # a retrain promoted a new churn champion

        assert test_client.post("/models/reload").status_code == 200

        assert _value(test_client, "retail_api_model_version", model="churn") == 8
        assert _value(test_client, "retail_api_model_version", model="value") == 9
        assert _value(test_client, "retail_api_model_loads_total", result="ok") == 2


def test_models_that_fail_to_load_show_as_version_zero_and_a_failed_load():
    def broken():
        raise RegistryError("no champion yet")

    with TestClient(api.create_app(broken)) as test_client:
        assert _value(test_client, "retail_api_model_version", model="churn") == 0
        assert _value(test_client, "retail_api_model_version", model="value") == 0
        assert _value(test_client, "retail_api_model_loads_total", result="failed") == 1


# ------------------------------------------------------------------------ isolation


def test_each_app_has_its_own_metrics_so_building_several_does_not_collide():
    with (
        TestClient(api.create_app(stub_champions)) as first,
        TestClient(api.create_app(stub_champions)) as second,
    ):
        first.get("/no/such/path")

        assert _value(first, "retail_api_requests_total", route="unmatched") == 1
        assert _value(second, "retail_api_requests_total", route="unmatched") is None


# ------------------------------------------------------------------ demo and real traffic


def test_requests_are_real_traffic_unless_they_carry_the_demo_header(plain):
    plain.get("/items/1")
    plain.get("/items/2", headers={metrics.DEMO_TRAFFIC_HEADER: "true"})
    plain.get("/items/3", headers={metrics.DEMO_TRAFFIC_HEADER: "TRUE"})

    real = _value(plain, "retail_api_requests_total", route="/items/{item_id}", traffic="real")
    demo = _value(plain, "retail_api_requests_total", route="/items/{item_id}", traffic="demo")

    assert (real, demo) == (1, 2)


@pytest.mark.parametrize("sent", ["false", "yes", "1", "", "x" * 500, "real", "demo"])
def test_any_other_header_value_is_real_traffic_so_the_label_has_only_two_values(plain, sent):
    plain.get("/items/1", headers={metrics.DEMO_TRAFFIC_HEADER: sent})
    plain.get("/items/2", headers={metrics.DEMO_TRAFFIC_HEADER: "true"})

    values = {s.labels["traffic"] for s in _samples(plain) if "traffic" in s.labels}

    assert values == {"real", "demo"}


def test_latency_is_kept_apart_for_demo_and_real_traffic_too(plain):
    plain.get("/items/1", headers={metrics.DEMO_TRAFFIC_HEADER: "true"})

    assert (
        _value(
            plain,
            "retail_api_request_duration_seconds_count",
            route="/items/{item_id}",
            traffic="demo",
        )
        == 1
    )
    assert (
        _value(
            plain,
            "retail_api_request_duration_seconds_count",
            route="/items/{item_id}",
            traffic="real",
        )
        is None
    )
