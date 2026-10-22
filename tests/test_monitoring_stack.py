"""The monitoring stack: Prometheus, Grafana and the read-only database login.

The services run in Docker, so these check the files that define them. That the stack starts, that
Prometheus scrapes the API, that every Grafana data source connects, and what the read-only login
can and cannot do were shown by running it against the real database (see CONTRIBUTING.md).
"""

import json
import re

import pytest
import yaml

from src import config, metrics

ROOT = config.PROJECT_ROOT
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
READONLY_SQL = (ROOT / "services" / "grafana" / "readonly_user.sql").read_text(encoding="utf-8")
SQL_CODE = re.sub(r"--[^\n]*", "", READONLY_SQL)  # the statements, without the explanations
GRAFANA = ROOT / "services" / "grafana"
DASHBOARDS = sorted((GRAFANA / "dashboards").glob("*.json"))
MONITORING_SERVICES = ["prometheus", "grafana-db-init", "grafana"]


def _service(name):
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|^\S|\Z)", COMPOSE, flags=re.M | re.S)
    assert match, f"service {name} not found in docker-compose.yml"
    return match.group(1)


def _datasources():
    text = (GRAFANA / "provisioning" / "datasources" / "datasources.yml").read_text(
        encoding="utf-8"
    )
    return yaml.safe_load(text)["datasources"]


# ------------------------------------------------------------------ the compose services


def test_monitoring_is_a_profile_so_a_plain_up_does_not_start_it():
    for name in MONITORING_SERVICES:
        assert 'profiles: ["monitoring"]' in _service(name), name


def test_images_are_pinned_to_an_exact_version_never_latest():
    for name, pattern in [
        ("prometheus", r"image: prom/prometheus:v\d+\.\d+\.\d+\n"),
        ("grafana", r"image: grafana/grafana-oss:\d+\.\d+\.\d+\n"),
        ("grafana-db-init", r"image: postgres:\d+\n"),
    ]:
        assert re.search(pattern, _service(name)), name
    assert ":latest" not in COMPOSE


def test_prometheus_and_grafana_are_reachable_from_this_machine_only():
    for name, port in [
        ("prometheus", "PROMETHEUS_PORT:-9090}:9090"),
        ("grafana", "GRAFANA_PORT:-3000}:3000"),
    ]:
        assert f'"127.0.0.1:${{{port}"' in _service(name), name


def test_grafana_refuses_to_start_without_a_password_because_its_default_is_admin_admin():
    service = _service("grafana")

    assert '[ -n "$$GF_SECURITY_ADMIN_PASSWORD" ]' in service and "exit 1" in service
    assert "GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-}" in service
    assert re.search(r"exec /run\.sh", service)


def test_grafana_does_not_allow_sign_up_anonymous_access_or_phoning_home():
    service = _service("grafana")

    assert 'GF_USERS_ALLOW_SIGN_UP: "false"' in service
    assert 'GF_AUTH_ANONYMOUS_ENABLED: "false"' in service
    assert 'GF_ANALYTICS_REPORTING_ENABLED: "false"' in service
    assert 'GF_ANALYTICS_CHECK_FOR_UPDATES: "false"' in service


def test_the_database_login_is_created_before_grafana_starts_and_prometheus_is_up_first():
    service = _service("grafana")

    assert "grafana-db-init:\n        condition: service_completed_successfully" in service
    assert "prometheus:\n        condition: service_healthy" in service


def test_the_login_setup_refuses_an_empty_password_and_never_puts_it_on_a_command_line():
    service = _service("grafana-db-init")

    assert '[ -n "$$GRAFANA_DB_PASSWORD" ]' in service and "exit 1" in service
    assert "GRAFANA_DB_PASSWORD: ${GRAFANA_DB_PASSWORD:-}" in service
    command = [line for line in service.splitlines() if "exec psql" in line]
    # the script reads the password from the environment itself (\getenv), so nothing about it is
    # passed to psql
    assert command and "GRAFANA_DB_PASSWORD" not in command[0] and "ro_password" not in command[0]
    assert 'restart: "no"' in service  # a one-shot job


def test_no_credential_is_written_into_the_compose_file_for_the_monitoring_services():
    for name in MONITORING_SERVICES:
        for line in _service(name).splitlines():
            if re.search(r"PASSWORD|SECRET", line) and ":" in line and "echo" not in line:
                value = line.split(":", 1)[1].strip()
                assert value.startswith("${") or value.startswith("$$") or value == "", line


def test_the_configuration_is_mounted_read_only_and_the_data_has_named_volumes():
    prometheus, grafana = _service("prometheus"), _service("grafana")

    assert "prometheus.yml:/etc/prometheus/prometheus.yml:ro" in prometheus
    assert "./services/grafana/provisioning:/etc/grafana/provisioning:ro" in grafana
    assert "./services/grafana/dashboards:/var/lib/grafana-dashboards:ro" in grafana
    assert (
        "prometheus-data:/prometheus" in prometheus and "grafana-data:/var/lib/grafana" in grafana
    )
    volumes = COMPOSE.split("\nvolumes:\n")[1]
    assert "prometheus-data:" in volumes and "grafana-data:" in volumes


def test_env_example_lists_the_monitoring_settings_without_values():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for key in ("GRAFANA_ADMIN_USER", "GRAFANA_ADMIN_PASSWORD", "GRAFANA_DB_PASSWORD"):
        assert re.search(rf"^{key}=", example, flags=re.M), key
    assert re.search(r"^GRAFANA_ADMIN_PASSWORD=$", example, flags=re.M)
    assert re.search(r"^GRAFANA_DB_PASSWORD=$", example, flags=re.M)


# ------------------------------------------------------------------------- Prometheus


def test_prometheus_scrapes_the_api_where_it_serves_its_metrics():
    config_file = yaml.safe_load(
        (ROOT / "services" / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")
    )
    job = next(j for j in config_file["scrape_configs"] if j["job_name"] == "retail-api")

    assert job["metrics_path"] == metrics.METRICS_PATH
    assert job["static_configs"][0]["targets"] == ["api:8000"]
    assert '--port", "8000"' in _service("api")  # the port the API listens on inside Docker


# ------------------------------------------------------------- the read-only database login


def _grants():
    """Every GRANT statement in the read-only script, with the database it applies to."""
    database = None
    grants = []
    for statement in re.split(r";\s*\n", READONLY_SQL):
        statement = re.sub(r"--[^\n]*", "", statement).strip()
        connect = re.search(r"\\connect\s+(?::\"(\w+)\"|(\w+))", statement)
        if connect:
            database = connect.group(1) or connect.group(2)
        grant = re.search(r"GRANT\b.*", statement, flags=re.S)
        if grant:
            grants.append((database, " ".join(grant.group(0).split())))
    return grants


def test_the_login_has_no_privileges_beyond_reading():
    role = re.search(r"CREATE ROLE grafana_ro ([A-Z ]+)'", READONLY_SQL).group(1)

    for attribute in ("NOSUPERUSER", "NOCREATEDB", "NOCREATEROLE", "NOREPLICATION"):
        assert attribute in role
    assert (
        "SET default_transaction_read_only = on" in READONLY_SQL
    )  # every session starts read-only
    assert "CONNECTION LIMIT" in READONLY_SQL


def test_only_select_and_usage_are_ever_granted():
    grants = _grants()

    assert grants
    for _, grant in grants:
        assert re.match(r"GRANT (SELECT|USAGE)\b", grant), grant
        assert not re.search(_MORE_THAN_READING, grant), grant


_MORE_THAN_READING = (
    r"\b(ALL|INSERT|UPDATE|DELETE|TRUNCATE|REFERENCES|TRIGGER|CREATE|TEMP|EXECUTE)\b"
    r"|GRANT OPTION"
)


def test_nothing_is_granted_by_wildcard_or_by_default_for_future_tables():
    assert "ALL TABLES" not in SQL_CODE.upper()
    assert "ALTER DEFAULT PRIVILEGES" not in SQL_CODE.upper()


def test_in_airflow_and_mlflow_only_named_columns_are_granted():
    for database, grant in _grants():
        if database in ("airflow", "mlflow") and grant.startswith("GRANT SELECT"):
            assert re.match(r"GRANT SELECT \([a-z_, ]+\) ON \w+ TO grafana_ro", grant), grant


@pytest.mark.parametrize(
    "column",
    [
        "conf",
        "executor_config",
        "next_kwargs",
        "hostname",
        "pid",
        "unixname",  # Airflow
        "source",
        "storage_location",
        "run_id",
        "user_id",
        "description",
        "run_link",  # MLflow
    ],
)
def test_columns_that_hold_arguments_paths_or_hosts_are_never_granted(column):
    for database, grant in _grants():
        if database == "mlflow" and "model_versions" in grant and column == "run_id":
            assert not re.search(rf"\b{column}\b", grant), grant
        elif column not in ("run_id",):
            assert not re.search(rf"\b{column}\b", grant), (column, grant)


def test_the_warehouse_login_sees_scoring_runs_only_never_customer_level_data():
    warehouse = [grant for database, grant in _grants() if database not in ("airflow", "mlflow")]

    assert sorted(warehouse) == [
        "GRANT SELECT ON ml.scoring_runs TO grafana_ro",
        "GRANT USAGE ON SCHEMA ml TO grafana_ro",
    ]


def test_the_password_comes_from_the_environment_and_the_script_can_be_run_again():
    assert "\\getenv ro_password GRAFANA_DB_PASSWORD" in READONLY_SQL
    assert re.search(
        r"WHERE NOT EXISTS \(SELECT FROM pg_roles WHERE rolname = 'grafana_ro'\)", READONLY_SQL
    )
    assert "format('ALTER ROLE grafana_ro WITH PASSWORD %L'" in READONLY_SQL  # follows .env
    assert not re.search(r"PASSWORD\s+'[^%]", READONLY_SQL)  # no literal password


# -------------------------------------------------------------------- Grafana data sources


def test_grafana_data_sources_never_contain_a_password_and_never_use_the_owner_login():
    for source in _datasources():
        if source["type"] == "grafana-postgresql-datasource":
            assert source["user"] == "grafana_ro", source["name"]
            assert source["secureJsonData"]["password"] == "$GRAFANA_DB_PASSWORD", source["name"]
    text = (GRAFANA / "provisioning" / "datasources" / "datasources.yml").read_text(
        encoding="utf-8"
    )
    assert "retail_admin" not in text and "POSTGRES_PASSWORD" not in text


def test_data_sources_are_not_editable_and_have_unique_names_and_ids():
    sources = _datasources()

    assert all(source["editable"] is False for source in sources)
    assert len({s["uid"] for s in sources}) == len(sources) == len({s["name"] for s in sources})


def test_the_grafana_login_can_reach_exactly_the_databases_that_have_a_data_source():
    databases = {
        s["jsonData"]["database"] for s in _datasources() if "database" in s.get("jsonData", {})
    }

    assert databases == {"$POSTGRES_DB", "airflow", "mlflow"}
    for name in ("airflow", "mlflow"):
        assert re.search(rf"^\\connect {name}$", READONLY_SQL, flags=re.M)


# ------------------------------------------------------------------------- dashboards


def _panels(dashboard):
    for panel in dashboard["panels"]:
        yield panel
        yield from panel.get("panels", [])


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_every_dashboard_is_valid_and_uses_only_known_data_sources(path):
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    known = {s["uid"] for s in _datasources()}
    ids = [panel["id"] for panel in _panels(dashboard)]

    assert dashboard["uid"] and dashboard["title"]
    assert len(ids) == len(set(ids))
    for panel in _panels(dashboard):
        assert panel["datasource"]["uid"] in known, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] in known, panel["title"]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_every_metric_a_dashboard_queries_exists_in_the_api(path):
    exposed = set()
    for family in metrics.ApiMetrics().registry.collect():
        exposed |= {sample.name for sample in family.samples} | {family.name}
        exposed |= {f"{family.name}{suffix}" for suffix in ("_total", "_bucket", "_count", "_sum")}
    queried = set()
    for panel in _panels(json.loads(path.read_text(encoding="utf-8"))):
        for target in panel["targets"]:
            queried |= set(re.findall(r"\bretail_api_[a-z_]+", target.get("expr", "")))

    assert queried, "the dashboard queries none of the API's metrics"
    assert queried <= exposed, queried - exposed


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_the_models_a_dashboard_asks_the_registry_about_are_the_projects_models(path):
    names = {config.CHURN_MODEL_NAME, config.VALUE_MODEL_NAME, config.SEGMENT_MODEL_NAME}
    asked = set()
    for panel in _panels(json.loads(path.read_text(encoding="utf-8"))):
        for target in panel["targets"]:
            asked |= set(re.findall(r"name = '([^']+)'", target.get("rawSql", "")))

    assert asked <= names, asked - names


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboards_ask_only_for_tables_and_columns_the_login_may_read(path):
    granted = " ".join(grant for _, grant in _grants())
    for panel in _panels(json.loads(path.read_text(encoding="utf-8"))):
        for target in panel["targets"]:
            for table in re.findall(r"\bFROM\s+([a-z_.]+)", target.get("rawSql", ""), flags=re.I):
                assert re.search(rf"\bON {re.escape(table)}\b", granted), (panel["title"], table)
