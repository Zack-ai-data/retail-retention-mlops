"""The MLflow tracking server's settings.

The server runs in Docker, so these check the files that define it. That it still serves tracking,
the registry and artifacts with these settings is shown by running it (see CONTRIBUTING.md).
"""

import ast
import re

from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_JOB_EXECUTION

from src import config

ROOT = config.PROJECT_ROOT
ENTRYPOINT = (ROOT / "services" / "mlflow" / "entrypoint.py").read_text(encoding="utf-8")
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def _server_arguments():
    """The argument list the entrypoint hands to `mlflow server` (os.execvp's second argument)."""
    for node in ast.walk(ast.parse(ENTRYPOINT)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "execvp":
            return [
                element.value if isinstance(element, ast.Constant) else None
                for element in node.args[1].elts
            ]
    raise AssertionError("the entrypoint no longer ends in os.execvp")


def _mlflow_service():
    match = re.search(r"^  mlflow:\n(.*?)(?=^  \S|^\S|\Z)", COMPOSE, flags=re.M | re.S)
    assert match, "service mlflow not found in docker-compose.yml"
    return match.group(1)


def test_the_server_runs_one_web_worker_not_the_default_four():
    arguments = _server_arguments()

    assert arguments[arguments.index("--workers") + 1] == "1"


def test_the_background_job_system_is_switched_off_with_a_setting_this_version_knows():
    # the name is read from the installed MLflow, so a rename in an upgrade fails here instead of
    # silently leaving the job system running
    assert MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name == "MLFLOW_SERVER_ENABLE_JOB_EXECUTION"
    assert f'{MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name}: "false"' in _mlflow_service()


def test_the_memory_settings_do_not_remove_what_the_pipeline_needs_or_its_protections():
    arguments = _server_arguments()

    assert "--serve-artifacts" in arguments  # clients upload artifacts through the server
    assert "--backend-store-uri" in arguments
    assert "--allowed-hosts" in arguments  # DNS-rebinding protection stays on
    assert "--disable-security-middleware" not in arguments
    service = _mlflow_service()
    assert '"127.0.0.1:${MLFLOW_PORT:-5000}:5000"' in service  # reachable from this machine only
    assert "healthcheck:" in service
