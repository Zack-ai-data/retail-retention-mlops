"""API access control: who may call what once a key is set, and that nothing is open by accident."""

import logging

import pytest
import yaml
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src import api, config, dashboard_client
from src.errors import RegistryError
from tests.test_api import (
    _features_of,
    db,  # noqa: F401  (the fixture scored is built on)
    scored,  # noqa: F401  (a database with data and a scoring run)
)
from tests.test_scoring import stub_champions

SHARED = "key-for-visitors"
ADMIN = "key-for-the-owner"
HEADER = {api.API_KEY_HEADER: SHARED}
ADMIN_HEADER = {api.API_KEY_HEADER: ADMIN}


def secured_app(shared=SHARED, admin=ADMIN, loader=stub_champions):
    return api.create_app(loader, api_key=shared, admin_key=admin)


@pytest.fixture
def secured(scored):  # noqa: F811
    with TestClient(secured_app()) as client:
        yield client


def concrete(path):
    return path.replace("{customer_id}", "1")


# --------------------------------------------------------- what is open and what is not


def test_the_open_paths_are_exactly_these():
    assert {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"} == api.OPEN_PATHS
    assert api.API_KEY_HEADER == "X-API-Key"


def test_the_dashboard_sends_the_header_name_the_api_checks():
    # the two modules cannot import each other (the dashboard image has no API code)
    assert dashboard_client.API_KEY_HEADER == api.API_KEY_HEADER


def test_every_route_that_is_not_deliberately_open_refuses_a_call_without_a_key(secured):
    documented = secured.app.openapi()["paths"]
    protected = {p: set(m) for p, m in documented.items() if p not in api.OPEN_PATHS}

    assert set(protected) >= {
        "/customers/{customer_id}/score",
        "/customers/{customer_id}/features",
        "/customers/{customer_id}/prediction",
        "/score",
        "/outreach/latest",
        "/outreach/sensitivity",
        "/models/reload",
    }
    for path, methods in protected.items():
        for method in methods:
            response = secured.request(method.upper(), concrete(path), json={})
            assert response.status_code == 401, (method, path)
            assert api.API_KEY_HEADER in response.json()["detail"]
            assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_a_route_added_straight_to_the_app_must_be_on_the_open_list(secured):
    # routes that need the key are added to the protected router, which the app wraps; anything
    # registered on the app itself (documented or not) is open, so it has to be listed on purpose
    direct = {r.path for r in secured.app.routes if isinstance(r, APIRoute)}

    assert direct == {"/health", "/metrics"}
    assert direct <= api.OPEN_PATHS


def test_the_open_paths_answer_without_a_key(secured):
    for path in api.OPEN_PATHS:
        assert secured.get(path).status_code == 200, path


def test_a_missing_key_is_refused_before_the_body_is_looked_at(secured):
    assert secured.post("/score", json={}).status_code == 401


def test_a_wrong_key_is_refused(secured):
    for headers in [
        {api.API_KEY_HEADER: "wrong"},
        {api.API_KEY_HEADER: ""},
        {api.API_KEY_HEADER: SHARED + "x"},
        {api.API_KEY_HEADER: SHARED[:-1]},
        {api.API_KEY_HEADER: SHARED.upper()},
    ]:
        assert secured.get("/customers/1/score", headers=headers).status_code == 401, headers


def test_a_key_in_the_address_does_not_count(secured):
    assert secured.get("/customers/1/score", params={"api_key": SHARED}).status_code == 401


def test_the_header_name_is_not_case_sensitive(secured):
    assert secured.get("/customers/1/score", headers={"x-api-key": SHARED}).status_code == 200


def test_a_header_that_is_not_ascii_is_just_wrong_not_an_error(secured):
    response = secured.get("/customers/1/score", headers=[(b"x-api-key", "café".encode("latin-1"))])

    assert response.status_code == 401


def test_the_shared_key_opens_every_route_a_visitor_needs(secured, scored):  # noqa: F811
    assert secured.get("/customers/1/score", headers=HEADER).status_code == 200
    assert secured.get("/customers/1/prediction", headers=HEADER).status_code == 200
    assert secured.get("/customers/1/features", headers=HEADER).status_code == 200
    assert secured.post("/score", json=_features_of(scored, 1), headers=HEADER).status_code == 200
    assert secured.get("/outreach/sensitivity", headers=HEADER).status_code == 200
    # no plan has been made, so the answer is the explained 404, not a refusal
    assert secured.get("/outreach/latest", headers=HEADER).status_code == 404


def test_the_admin_key_also_opens_the_ordinary_routes(secured):
    assert secured.get("/customers/1/score", headers=ADMIN_HEADER).status_code == 200


def test_keys_are_compared_in_constant_time_on_bytes(secured, monkeypatch):
    seen = []
    real = api.secrets.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(api.secrets, "compare_digest", spy)

    secured.get("/customers/1/score", headers={api.API_KEY_HEADER: "wrong"})

    assert seen and all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in seen)


def test_the_key_matching_rules():
    assert api._key_matches("a", "a")
    assert not api._key_matches(None, "a")
    assert not api._key_matches("a", "")  # no key configured never matches anything
    assert not api._key_matches("", "")
    assert not api._key_matches("a", "b")
    assert not api._key_matches("café", "cafe")
    assert not api._key_matches("cafe", "café")


# ------------------------------------------------------------------ reloading the models


def test_the_shared_key_cannot_reload_the_models(scored):  # noqa: F811
    loads = []

    def loader():
        loads.append(1)
        return stub_champions()

    with TestClient(secured_app(loader=loader)) as client:
        refused = client.post("/models/reload", headers=HEADER)

        assert refused.status_code == 403 and "admin key" in refused.json()["detail"]
        assert len(loads) == 1  # only the load at start-up


def test_the_admin_key_reloads_the_models(scored):  # noqa: F811
    loads = []

    def loader():
        loads.append(1)
        return stub_champions()

    with TestClient(secured_app(loader=loader)) as client:
        reloaded = client.post("/models/reload", headers=ADMIN_HEADER)

        assert reloaded.status_code == 200
        assert reloaded.json() == {"churn_model_version": "7", "value_model_version": "9"}
        assert len(loads) == 2


def test_reloading_without_any_key_is_refused_as_unauthenticated(secured):
    assert secured.post("/models/reload").status_code == 401


def test_with_no_admin_key_configured_nobody_can_reload(scored):  # noqa: F811
    with TestClient(secured_app(admin="")) as client:
        for headers in [HEADER, {api.API_KEY_HEADER: ""}, {}]:
            response = client.post("/models/reload", headers=headers)
            assert response.status_code in (401, 403)
        assert "switched off" in client.post("/models/reload", headers=HEADER).json()["detail"]


# ------------------------------------------------------------------------- health


def test_health_tells_a_stranger_only_whether_it_works(secured):
    body = secured.get("/health").json()

    assert body == {"status": "ok", "models_loaded": True, "database": "ok"}


def test_health_tells_a_caller_with_a_key_everything(secured):
    body = secured.get("/health", headers=HEADER).json()

    assert body["churn_model_version"] == "7" and body["latest_scoring_run"]["snapshot_date"]


def test_a_failing_health_does_not_tell_a_stranger_why(db):  # noqa: F811
    def down():
        raise RegistryError("registry down at http://mlflow:5000")

    with TestClient(secured_app(loader=down)) as client:
        stranger = client.get("/health")
        owner = client.get("/health", headers=HEADER)

    assert stranger.status_code == 503 and owner.status_code == 503
    assert set(stranger.json()["detail"]) <= {"status", "models_loaded", "database"}
    assert "mlflow" not in stranger.text
    assert "mlflow" in owner.json()["detail"]["problem"]


# ------------------------------------------------------------------- the open mode


def test_without_a_key_set_the_api_is_open_as_before(scored, caplog):  # noqa: F811
    with caplog.at_level(logging.WARNING, logger="src.api"):
        app = api.create_app(stub_champions, api_key="", admin_key="")
    with TestClient(app) as client:
        assert client.get("/customers/1/score").status_code == 200
        assert client.post("/models/reload").status_code == 200
        assert (
            client.get("/customers/1/score", headers={api.API_KEY_HEADER: "any"}).status_code == 200
        )

    assert app.state.auth_enabled is False
    assert "API_KEY is not set" in caplog.text


def test_with_a_key_set_the_state_says_so_and_nothing_warns(scored, caplog):  # noqa: F811
    with caplog.at_level(logging.WARNING, logger="src.api"):
        app = secured_app()

    assert app.state.auth_enabled is True
    assert "API_KEY is not set" not in caplog.text


def test_the_keys_default_to_the_settings(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "from-settings")
    monkeypatch.setattr(config, "API_ADMIN_KEY", "admin-from-settings")

    app = api.create_app(stub_champions)

    assert app.state.auth_enabled is True
    with TestClient(app) as client:
        assert client.post("/score", json={}).status_code == 401
        assert client.post("/models/reload").status_code == 401


def test_an_empty_setting_leaves_the_api_open(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")

    assert api.create_app(stub_champions).state.auth_enabled is False


# ------------------------------------------------------------------- self-description


def test_the_docs_show_which_operations_need_a_key(secured):
    paths = secured.get("/openapi.json").json()["paths"]

    assert "security" in paths["/score"]["post"]
    assert "security" in paths["/models/reload"]["post"]
    assert "security" not in paths["/health"]["get"]  # there the key is an optional header
    assert any(p["name"] == "X-API-Key" for p in paths["/health"]["get"]["parameters"])
    assert "X-API-Key" in secured.get("/openapi.json").text


# ------------------------------------------------------------------ how it is deployed


def _compose():
    path = config.PROJECT_ROOT / "docker-compose.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_only_the_api_gets_the_keys_and_the_dashboard_only_the_shared_one():
    services = _compose()["services"]

    assert {"API_KEY", "API_ADMIN_KEY"} <= set(services["api"]["environment"])
    assert "API_KEY" in services["dashboard"]["environment"]
    assert "API_ADMIN_KEY" not in services["dashboard"]["environment"]
    for name in ("app", "quality"):
        assert not {"API_KEY", "API_ADMIN_KEY"} & set(services[name].get("environment", {})), name


def test_the_api_keeps_the_pipeline_settings_it_needs():
    api_env = _compose()["services"]["api"]["environment"]

    assert api_env["POSTGRES_HOST"] == "postgres"
    assert api_env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_the_keys_are_explained_in_the_example_env_file_and_left_empty():
    lines = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert "API_KEY=" in lines and "API_ADMIN_KEY=" in lines
    assert "DASHBOARD_ALLOW_RELOAD=1" in lines
