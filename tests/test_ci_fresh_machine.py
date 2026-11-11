"""The fresh-machine CI job: what a new machine does, from nothing.

GitHub runs the job, so it cannot be run here as a test. What can be checked is what it is set to
do:
that it asserts the right things in the right order, never reloads the models by hand (the point is
that the API recovers by itself), uses only throwaway credentials, and always cleans up. That the
steps themselves work was shown by running them, unchanged, in an isolated Compose project with
empty volumes.
"""

import pytest
import yaml

from src import api, config

ROOT = config.PROJECT_ROOT
CI_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
JOB = yaml.safe_load(CI_TEXT)["jobs"]["fresh-machine"]
STEPS = {step["name"]: step for step in JOB["steps"] if "name" in step}
RUN = {name: step["run"] for name, step in STEPS.items() if "run" in step}
EMPTY_DATABASE_STEP = "With no data the API answers a clean 503 that says what to run"
RECOVERY_STEP = "The API picks the models up by itself and serves a real customer"


def _job_text():
    start = CI_TEXT.index("  fresh-machine:")
    return CI_TEXT[start:]


def test_it_runs_after_a_merge_and_on_the_schedule_not_on_pull_requests():
    assert JOB["if"] == "github.event_name != 'pull_request'"


def test_it_uses_only_throwaway_credentials_and_no_secret():
    assert JOB["env"] == {
        "POSTGRES_USER": "ci_admin",
        "POSTGRES_PASSWORD": "ci_password",
        "POSTGRES_DB": "retail_dw",
    }
    assert "secrets." not in _job_text()


def test_it_cannot_run_for_ever():
    assert 20 <= JOB["timeout-minutes"] <= 60
    assert (
        _job_text().count("seq 1 60") == 2
    )  # both waiting loops stop after a fixed number of tries
    assert "while true" not in _job_text()


def test_the_steps_run_in_the_order_a_new_machine_would():
    names = [step["name"] for step in JOB["steps"] if "name" in step]

    assert names == [
        "Build and start the database and MLflow on empty volumes",
        "Build and start the API",
        EMPTY_DATABASE_STEP,
        "Load, build, train, score, plan and monitor from nothing",
        RECOVERY_STEP,
        "Container logs, if anything failed",
        "Tear down",
    ]


def test_it_waits_for_the_database_and_mlflow_but_not_for_the_api_which_is_meant_to_be_unready():
    assert "--wait" in RUN["Build and start the database and MLflow on empty volumes"]
    assert "postgres mlflow" in RUN["Build and start the database and MLflow on empty volumes"]
    assert "--wait" not in RUN["Build and start the API"]  # with no data its health check must fail


def test_with_no_data_it_demands_a_503_and_not_merely_a_failure():
    script = RUN[EMPTY_DATABASE_STEP]

    assert 'test "$code" = 503' in script  # a 500 (the bug this guards) would fail here
    assert '"database":"not ready"' in script
    assert "python -m src.main --stage all" in script


def test_the_instruction_it_checks_for_is_the_one_the_api_gives():
    assert "python -m src.main --stage all" in api.DATA_NOT_LOADED
    assert "python -m src.main --stage all" in RUN[EMPTY_DATABASE_STEP]


def test_the_pipeline_runs_with_the_projects_own_compose_service_and_every_stage():
    assert (
        RUN["Load, build, train, score, plan and monitor from nothing"]
        == "docker compose run --rm app python -m src.main --stage all"
    )


def test_the_api_must_recover_by_itself_so_nothing_reloads_the_models_by_hand():
    # the commands, not the comments that explain them
    assert not any("models/reload" in script for script in RUN.values())
    script = RUN[RECOVERY_STEP]

    assert 'test "$code" = 200' in script
    assert '"models_loaded":true' in script and '"database":"ok"' in script


def test_the_recovered_api_must_serve_a_real_customer_both_live_and_from_the_stored_scores():
    script = RUN[RECOVERY_STEP]

    assert "/outreach/latest" in script  # where the customer comes from: the plan the pipeline made
    assert "/customers/$id/score" in script and "/customers/$id/prediction" in script
    # -f makes any 4xx or 5xx a failed step: every call that must succeed has it, not just one
    calls = [line.strip() for line in script.splitlines() if line.strip().startswith("curl ")]
    needing_success = [call for call in calls if "health.json" not in call]
    assert len(needing_success) == 3
    assert all(call.startswith("curl -sf ") for call in needing_success)
    assert 'test -n "$id"' in script  # an empty plan must not pass silently


def test_it_prints_the_logs_when_something_fails_and_always_tears_down_its_own_volumes():
    assert STEPS["Container logs, if anything failed"]["if"] == "failure()"
    assert STEPS["Tear down"]["if"] == "always()"
    assert RUN["Tear down"] == "docker compose --profile app down -v"


def test_the_teardown_switches_on_the_profile_of_the_services_it_started():
    # a plain `docker compose down` leaves a service running when its profile is not switched on:
    # the API and the pipeline service inherit the "app" profile from the shared settings
    shared_settings = COMPOSE.split("\nservices:\n", 1)[0]
    assert 'profiles: ["app"]' in shared_settings
    for service in ("api", "app"):
        block = COMPOSE.split(f"\n  {service}:\n", 1)[1].split("\n  # ", 1)[0]
        assert "<<: *app-base" in block
    assert "--profile app" in RUN["Tear down"]


@pytest.mark.parametrize("service", ["postgres", "mlflow", "api", "app"])
def test_every_service_it_uses_exists_in_the_compose_file(service):
    assert f"\n  {service}:\n" in COMPOSE


def test_it_asks_the_api_where_compose_publishes_it():
    assert "127.0.0.1:${API_PORT:-8000}:8000" in COMPOSE
    assert "http://127.0.0.1:8000/health" in RUN[EMPTY_DATABASE_STEP]


def test_the_workflow_header_describes_the_job():
    assert "fresh-machine" in CI_TEXT.split("name: CI")[0]
