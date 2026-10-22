"""The Airflow stack: its settings, the database URL helper and the health-check DAG.

Airflow itself is not installed in the test environment, so these check the files that define the
stack. That the stack really starts and runs a DAG through the Celery worker is shown by running it
(docker compose --profile airflow up -d ...; trigger pipeline_healthcheck).
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest

from src import config

ROOT = config.PROJECT_ROOT
AIRFLOW = ROOT / "services" / "airflow"
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
DOCKERFILE = (AIRFLOW / "Dockerfile").read_text(encoding="utf-8")


def _load_db_url():
    spec = importlib.util.spec_from_file_location("db_url", AIRFLOW / "db_url.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


db_url = _load_db_url()


def _service(name):
    """The text of one service in the compose file."""
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|^\S|\Z)", COMPOSE, flags=re.M | re.S)
    assert match, f"service {name} not found"
    return match.group(1)


AIRFLOW_SERVICES = [
    "airflow-init",
    "airflow-apiserver",
    "airflow-scheduler",
    "airflow-dag-processor",
    "airflow-worker",
    "airflow-triggerer",
]

# ----------------------------------------------------------------- the database URL helper


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "retail_admin")
    monkeypatch.setenv("POSTGRES_PASSWORD", "plain-password")
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5432")


def test_the_url_points_at_the_airflow_database_not_the_warehouse(env):
    assert db_url.database_url() == (
        "postgresql+psycopg2://retail_admin:plain-password@postgres:5432/airflow"
    )


def test_celery_uses_the_db_prefixed_form(env):
    assert db_url.database_url(celery=True).startswith("db+postgresql+psycopg2://")


@pytest.mark.parametrize("password", ["p@ss/word", "a:b#c", "50%off", "with space", "ünï/çödé?"])
def test_a_password_with_special_characters_round_trips(monkeypatch, env, password):
    monkeypatch.setenv("POSTGRES_PASSWORD", password)

    parts = urlsplit(db_url.database_url())

    assert unquote(parts.password) == password  # not split at an @ or /, and recoverable
    assert parts.hostname == "postgres" and parts.path == "/airflow"


def test_run_as_a_script_it_prints_the_url_with_no_trailing_newline(env):
    result = subprocess.run(
        [sys.executable, str(AIRFLOW / "db_url.py")],
        capture_output=True,
        text=True,
        env={**os.environ},
    )

    # Airflow uses the output exactly as printed: a newline would become part of the database name
    assert result.stdout == db_url.database_url()
    assert not result.stdout.endswith("\n")


def test_the_celery_flag_works_when_run_as_a_script(env):
    result = subprocess.run(
        [sys.executable, str(AIRFLOW / "db_url.py"), "--celery"], capture_output=True, text=True
    )

    assert result.stdout == db_url.database_url(celery=True)


def test_a_missing_password_is_an_error_not_an_empty_one(monkeypatch, env):
    monkeypatch.delenv("POSTGRES_PASSWORD")

    with pytest.raises(KeyError):
        db_url.database_url()


# --------------------------------------------------------------------- the compose settings


def test_every_airflow_service_is_in_the_airflow_profile_so_a_plain_up_does_not_start_them():
    assert 'profiles: ["airflow"]' in COMPOSE.split("services:")[0]  # the shared settings
    redis = _service("redis")
    assert 'profiles: ["airflow"]' in redis
    for name in AIRFLOW_SERVICES:
        assert "<<: *airflow-common" in _service(name), name


def test_airflow_runs_with_the_celery_executor_and_redis():
    assert "AIRFLOW__CORE__EXECUTOR: CeleryExecutor" in COMPOSE
    assert "AIRFLOW__CELERY__BROKER_URL: redis://" in COMPOSE
    assert "image: redis:7.2" in COMPOSE


def test_the_web_interface_is_bound_to_localhost_only():
    ports = re.findall(r'- "([^"]*:8080)"', _service("airflow-apiserver"))

    assert ports and all(p.startswith("127.0.0.1:") for p in ports)


def test_redis_is_not_published_to_the_host():
    assert "ports:" not in _service("redis")


def test_no_credentials_are_written_into_the_compose_file():
    assert "airflow:airflow" not in COMPOSE
    assert not re.search(
        r"(PASSWORD|SECRET|FERNET_KEY)[A-Z_]*:\s+[A-Za-z0-9+/=_-]{8,}\s*$", COMPOSE, flags=re.M
    )
    # secrets come from the environment, never as a default
    assert "AIRFLOW__CORE__FERNET_KEY: ${AIRFLOW_FERNET_KEY:-}" in COMPOSE
    assert "AIRFLOW__API_AUTH__JWT_SECRET: ${AIRFLOW_JWT_SECRET:-}" in COMPOSE
    assert "_AIRFLOW_WWW_USER_PASSWORD: ${AIRFLOW_ADMIN_PASSWORD:-}" in COMPOSE


def test_the_database_password_is_not_in_any_connection_string_in_the_compose_file():
    assert "postgresql+psycopg2://" not in COMPOSE  # built by db_url.py from the environment
    assert "SQL_ALCHEMY_CONN_CMD: python /opt/airflow/db_url.py" in COMPOSE
    assert "RESULT_BACKEND_CMD: python /opt/airflow/db_url.py --celery" in COMPOSE


def test_no_setting_is_marked_required_because_that_would_break_every_compose_command():
    # compose reads the whole file for any command, so ${VAR:?...} would stop people who do not
    # use Airflow from starting Postgres; the init service checks the secrets instead
    assert not re.search(r"\$\{[A-Z_]+:\?", COMPOSE)


def test_init_refuses_to_run_without_the_secrets():
    init = _service("airflow-init")

    for name in (
        "AIRFLOW__CORE__FERNET_KEY",
        "AIRFLOW__API_AUTH__JWT_SECRET",
        "_AIRFLOW_WWW_USER_PASSWORD",
    ):
        assert name in init
    assert "these settings are empty" in init and "exit 1" in init


def test_init_checks_that_the_database_and_admin_user_really_exist():
    # the image's entrypoint carries on when migration fails (it once "succeeded" against a
    # missing database), so init verifies the result itself
    init = _service("airflow-init")

    assert "airflow db check" in init
    assert "airflow users list" in init
    assert "the admin user was not created" in init


def test_the_workers_and_scheduler_wait_for_init_to_complete_successfully():
    for name in AIRFLOW_SERVICES[1:]:
        assert "service_completed_successfully" in _service(name), name


def test_the_worker_has_what_the_pipeline_tasks_need():
    shared = COMPOSE.split("services:")[0]

    for variable in (
        "POSTGRES_HOST: postgres",
        "MLFLOW_TRACKING_URI: http://mlflow:5000",
        "DATA_PATH:",
    ):
        assert variable in shared
    assert "./data:/data:ro" in shared  # read-only: the pipeline never writes the source file


def test_example_dags_are_not_loaded_and_new_dags_start_paused():
    assert 'AIRFLOW__CORE__LOAD_EXAMPLES: "false"' in COMPOSE
    assert 'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "true"' in COMPOSE


def test_env_example_lists_the_airflow_settings_without_values():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for name in ("AIRFLOW_ADMIN_PASSWORD", "AIRFLOW_FERNET_KEY", "AIRFLOW_JWT_SECRET"):
        assert re.search(rf"^{name}=$", example, flags=re.M), f"{name} should be present and empty"


def test_the_real_env_file_is_ignored_by_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored


# --------------------------------------------------------------------------- the image


def test_airflow_is_pinned_to_an_exact_version_and_python():
    assert re.search(r"^FROM apache/airflow:\d+\.\d+\.\d+-python3\.12$", DOCKERFILE, flags=re.M)


def test_the_pipeline_has_its_own_environment_built_from_the_hashed_lock_file():
    assert "python -m venv /opt/pipeline-venv" in DOCKERFILE
    assert "--require-hashes" in DOCKERFILE and "requirements.lock" in DOCKERFILE
    assert "PIPELINE_PYTHON=/opt/pipeline-venv/bin/python" in DOCKERFILE


def test_the_pipeline_code_and_sql_are_in_the_image():
    assert "COPY --chown=airflow:0 src /opt/pipeline/src" in DOCKERFILE
    assert "COPY --chown=airflow:0 sql /opt/pipeline/sql" in DOCKERFILE


def test_the_image_does_not_contain_the_dataset_or_secrets():
    copied = re.findall(r"^COPY .*$", DOCKERFILE, flags=re.M)

    assert copied, "the image copies the pipeline code"
    assert not any("data" in line.split()[-2] or ".env" in line for line in copied)


def test_the_commit_can_be_passed_in_so_runs_started_by_airflow_record_it():
    assert "ARG GIT_SHA" in DOCKERFILE and "GIT_SHA=${GIT_SHA}" in DOCKERFILE


# ------------------------------------------------------------------ the health-check DAG

DAG_FILE = AIRFLOW / "dags" / "pipeline_healthcheck.py"
DAG_SOURCE = DAG_FILE.read_text(encoding="utf-8")


def test_the_healthcheck_dag_runs_only_when_started_by_hand():
    assert "schedule=None" in DAG_SOURCE
    assert 'dag_id="pipeline_healthcheck"' in DAG_SOURCE
    assert "catchup=False" in DAG_SOURCE


def test_every_task_uses_the_pipeline_environment_not_airflows_own():
    tasks = re.findall(r'task_id="(\w+)"', DAG_SOURCE)

    assert set(tasks) == {
        "pipeline_environment",
        "pipeline_command_line",
        "database_reachable",
        "mlflow_reachable",
    }
    assert DAG_SOURCE.count("{PIPELINE}") == len(tasks)  # each one starts from the pipeline python
    assert 'PIPELINE = "cd $PIPELINE_HOME && $PIPELINE_PYTHON"' in DAG_SOURCE


def test_the_healthcheck_dag_imports_only_airflow_and_the_standard_library():
    imported = set()
    for node in ast.walk(ast.parse(DAG_SOURCE)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"airflow", "datetime"}, imported


def test_the_healthcheck_dag_never_prints_secrets():
    lowered = DAG_SOURCE.lower()

    assert "password" not in lowered and "secret" not in lowered
    # nothing here dumps the environment, where the secrets are
    assert "os.environ" not in lowered and "printenv" not in lowered
    assert not re.search(r"\benv\b", lowered)


# ------------------------------------------------------- shared helpers and the ETL DAG

DAGS = AIRFLOW / "dags"
COMMON_SOURCE = (DAGS / "pipeline_common.py").read_text(encoding="utf-8")
ETL_SOURCE = (DAGS / "retail_etl.py").read_text(encoding="utf-8")
CHECK_DAGS_SOURCE = (AIRFLOW / "check_dags.py").read_text(encoding="utf-8")


def _imported_modules(source):
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def _task(source, task_id):
    """The text of the stage(...) call that creates one task."""
    match = re.search(rf'stage\(\s*"{task_id}",.*?\n    \)', source, re.DOTALL)
    assert match, f"no task {task_id!r}"
    return match.group(0)


def test_the_stage_names_the_dags_may_use_are_the_ones_the_command_line_accepts():
    main_source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
    cli = set(re.findall(r'"(\w+)"', re.search(r"choices=\[(.*?)\]", main_source).group(1)))
    dag_stages = set(
        re.findall(r'"(\w+)"', re.search(r"STAGES = \((.*?)\)", COMMON_SOURCE).group(1))
    )

    assert dag_stages == cli - {"all"}  # "all" would hide which step failed, so DAGs do not use it


def test_an_unknown_stage_is_rejected_when_the_dag_is_loaded():
    assert "raise ValueError" in COMMON_SOURCE
    assert "if name not in STAGES" in COMMON_SOURCE


def test_every_stage_task_starts_from_the_pipeline_python_not_airflows_own():
    assert 'PIPELINE = "cd $PIPELINE_HOME && $PIPELINE_PYTHON"' in COMMON_SOURCE
    assert 'bash_command=f"{PIPELINE} -m src.main --stage {name}"' in COMMON_SOURCE


def test_dag_modules_import_only_airflow_the_standard_library_and_each_other():
    sibling_modules = {path.stem for path in DAGS.glob("*.py")}

    for source in (COMMON_SOURCE, ETL_SOURCE, RETRAIN_SOURCE):
        assert _imported_modules(source) <= {"airflow", "datetime"} | sibling_modules


def test_the_etl_dag_runs_only_when_started_by_hand_and_never_twice_at_once():
    assert 'dag_id="retail_etl"' in ETL_SOURCE
    assert "schedule=None" in ETL_SOURCE
    assert "catchup=False" in ETL_SOURCE
    assert "max_active_runs=1" in ETL_SOURCE


def test_the_etl_dag_loads_before_it_transforms():
    assert '"load_raw",\n        "load",' in ETL_SOURCE
    assert '"transform_marts",\n        "transform",' in ETL_SOURCE
    assert "load_raw >> transform_marts" in ETL_SOURCE


def test_a_failed_quality_check_is_not_retried_but_a_failed_load_is_retried_once():
    assert "retries=1" in _task(ETL_SOURCE, "load_raw")
    assert "retries=0" in _task(ETL_SOURCE, "transform_marts")


def test_every_etl_task_has_a_time_limit():
    for task_id in ("load_raw", "transform_marts"):
        assert "execution_timeout=" in _task(ETL_SOURCE, task_id)


def test_only_the_transform_reports_the_feature_table_as_refreshed():
    assert "outlets=[CUSTOMER_SNAPSHOTS]" in _task(ETL_SOURCE, "transform_marts")
    assert "outlets" not in _task(ETL_SOURCE, "load_raw")  # raw data alone is not ready to train on
    assert "marts/customer_snapshots" in COMMON_SOURCE


def test_the_dag_check_script_lists_exactly_the_dags_in_the_folder():
    dag_ids = set()
    for path in DAGS.glob("*.py"):
        dag_ids |= set(re.findall(r'dag_id="(\w+)"', path.read_text(encoding="utf-8")))
    expected = set(
        re.findall(r'"(\w+)"', re.search(r"EXPECTED = \{(.*?)\}", CHECK_DAGS_SOURCE).group(1))
    )

    assert expected == dag_ids


# ---------------------------------------------------- the retraining and scoring DAG

RETRAIN_SOURCE = (DAGS / "retail_retrain_score.py").read_text(encoding="utf-8")


def _dependencies(source):
    return re.findall(r"^    (.*>>.*)$", source, flags=re.MULTILINE)


def test_the_retraining_dag_starts_when_the_feature_table_is_refreshed_or_by_hand():
    assert 'dag_id="retail_retrain_score"' in RETRAIN_SOURCE
    assert "schedule=[CUSTOMER_SNAPSHOTS]" in RETRAIN_SOURCE
    assert "catchup=False" in RETRAIN_SOURCE
    assert "max_active_runs=1" in RETRAIN_SOURCE


def test_the_etl_dag_produces_the_asset_the_retraining_dag_waits_for():
    assert "outlets=[CUSTOMER_SNAPSHOTS]" in ETL_SOURCE
    assert "schedule=[CUSTOMER_SNAPSHOTS]" in RETRAIN_SOURCE
    assert COMMON_SOURCE.count("Asset(") == 1  # one definition, so the two cannot name it apart


def test_the_stages_run_in_order_and_the_report_does_not_hold_up_scoring():
    assert _dependencies(RETRAIN_SOURCE) == [
        "train >> [report_promotion, score]",
        "score >> plan >> monitor",
    ]


def test_each_task_runs_the_stage_its_name_says():
    for task_id, name in [
        ("train", "train"),
        ("report_promotion", "report"),
        ("score", "score"),
        ("plan", "plan"),
        ("monitor", "monitor"),
    ]:
        assert f'"{task_id}",\n        "{name}",' in RETRAIN_SOURCE


def test_a_rejected_model_ends_the_report_task_as_skipped_not_failed():
    task = _task(RETRAIN_SOURCE, "report_promotion")
    main_source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")

    assert "skip_on_exit_code=PROMOTION_REJECTED_EXIT_CODE" in task
    assert "retries=0" in task  # the same decision would be read again
    dag_code = re.search(r"^PROMOTION_REJECTED_EXIT_CODE = (\d+)", COMMON_SOURCE, re.MULTILINE)
    cli_code = re.search(r"^PROMOTION_REJECTED_EXIT_CODE = (\d+)", main_source, re.MULTILINE)
    assert dag_code and cli_code and dag_code.group(1) == cli_code.group(1)


def test_training_scoring_planning_and_monitoring_are_retried_once_each_with_time_limits():
    for task_id in ("train", "score", "plan", "monitor"):
        task = _task(RETRAIN_SOURCE, task_id)
        assert "retries=1" in task, task_id
        assert "retry_delay=" in task and "execution_timeout=" in task, task_id
    assert "execution_timeout=" in _task(RETRAIN_SOURCE, "report_promotion")


# ------------------------------------------------------------ the DAG check in CI

CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def _ci_job(name):
    """The text of one job in ci.yml, up to the next job or the comment above it."""
    match = re.search(rf"\n  {name}:\n(.*?)(?=\n  (?:# |[a-z-]+:)|\Z)", CI, re.DOTALL)
    assert match, f"no job {name!r} in ci.yml"
    return match.group(1)


def test_ci_loads_every_dag_in_the_airflow_image_on_pull_requests():
    job = _ci_job("airflow-dags")

    # no condition: it runs on pull requests and on main (the job text starts at its first key)
    assert not re.search(r"^    if:", job, flags=re.MULTILINE)
    assert "file: services/airflow/Dockerfile" in job
    assert "python /check_dags.py" in job
    assert "timeout-minutes:" in job


def test_ci_mounts_the_real_dags_and_the_check_read_only():
    job = _ci_job("airflow-dags")

    assert '-v "$PWD/services/airflow/dags:/opt/airflow/dags:ro"' in job
    assert '-v "$PWD/services/airflow/check_dags.py:/check_dags.py:ro"' in job


def test_ci_builds_the_airflow_image_without_publishing_or_logging_in():
    job = _ci_job("airflow-dags")

    assert "push: false" in job and "load: true" in job
    assert "login-action" not in job and "secrets." not in job


def test_ci_caches_the_airflow_image_layers_apart_from_other_images():
    job = _ci_job("airflow-dags")

    assert "cache-from: type=gha,scope=airflow" in job
    assert "cache-to: type=gha,scope=airflow,mode=max" in job


def test_the_airflow_images_entrypoint_needs_bash_not_sh_to_run_a_shell_command():
    # the image's entrypoint passes only `bash` and `python` straight through; anything else is
    # read as an Airflow command and fails ("invalid choice: 'sh'")
    job = _ci_job("airflow-dags")

    assert "retail-airflow:ci bash -c" in job
    assert "retail-airflow:ci sh " not in job


def test_the_dag_job_does_not_replace_the_required_checks():
    assert _ci_job("quality") and _ci_job("dashboard")
    assert "airflow-dags" not in _ci_job("quality") + _ci_job("dashboard")


def test_the_dags_print_no_secrets():
    lowered = (COMMON_SOURCE + ETL_SOURCE + RETRAIN_SOURCE).lower()

    assert "password" not in lowered and "secret" not in lowered
    assert "os.environ" not in lowered and "printenv" not in lowered
