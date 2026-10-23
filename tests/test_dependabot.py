"""Dependabot's version updates: what it watches, and what it is forbidden to touch.

GitHub reads .github/dependabot.yml and opens the pull requests itself, so this cannot be run
here. What can be checked is that every Dockerfile and compose image is covered, that nothing
that would break the project (a major Postgres, a different Python) can be proposed, and that the
settings are the ones decided on. That GitHub accepts the file and opens its first pull requests
is shown under Insights > Dependency graph > Dependabot after the change reaches main.
"""

import re

import yaml

from src import config

ROOT = config.PROJECT_ROOT
CONFIG_TEXT = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
CONFIG = yaml.safe_load(CONFIG_TEXT)
UPDATES = CONFIG["updates"]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
MAJOR, MINOR = "version-update:semver-major", "version-update:semver-minor"
SKIPPED_FOLDERS = {".git", ".venv", "node_modules", "docs"}


def _entries(ecosystem):
    return [entry for entry in UPDATES if entry["package-ecosystem"] == ecosystem]


def _ignored(entry):
    return {rule["dependency-name"]: set(rule["update-types"]) for rule in entry.get("ignore", [])}


def _dockerfile_folders():
    folders = set()
    for path in ROOT.rglob("Dockerfile*"):
        relative = path.relative_to(ROOT)
        if not SKIPPED_FOLDERS & set(relative.parts):
            folders.add("/" + "/".join(relative.parent.parts))
    return folders


def test_it_watches_actions_docker_images_and_the_compose_images_and_never_pip():
    assert CONFIG["version"] == 2
    assert {entry["package-ecosystem"] for entry in UPDATES} == {
        "github-actions",
        "docker",
        "docker-compose",
    }
    # the libraries are locked by hash and regenerated on purpose, never by a bot
    assert "pip" not in {entry["package-ecosystem"] for entry in UPDATES}


def test_every_folder_with_a_dockerfile_is_watched_and_every_watched_folder_has_one():
    watched = {entry["directory"] for entry in _entries("docker")}

    assert _dockerfile_folders() == watched
    for folder in watched:
        assert any((ROOT / folder.lstrip("/")).glob("Dockerfile*")), folder


def test_the_compose_file_and_the_workflows_are_watched_from_where_they_are():
    (compose,) = _entries("docker-compose")
    (actions,) = _entries("github-actions")

    assert compose["directory"] == "/" and (ROOT / "docker-compose.yml").exists()
    assert actions["directory"] == "/" and (ROOT / ".github" / "workflows").is_dir()


def test_every_update_runs_weekly_on_tuesday_morning_malaysia_time_a_few_pull_requests_at_most():
    for entry in UPDATES:
        assert entry["schedule"] == {
            "interval": "weekly",
            "day": "tuesday",  # the day after the weekly CI run on main
            "time": "10:00",
            "timezone": "Asia/Kuala_Lumpur",
        }, entry["directory"]
        assert 1 <= entry["open-pull-requests-limit"] <= 5, entry["directory"]


def test_the_titles_follow_the_repositorys_conventional_style():
    assert {entry["commit-message"]["prefix"] for entry in _entries("github-actions")} == {"ci"}
    for entry in _entries("docker") + _entries("docker-compose"):
        assert entry["commit-message"]["prefix"] == "chore"


def test_the_python_base_image_is_never_changed_because_the_locks_are_built_for_one_version():
    for entry in _entries("docker"):
        dockerfile = (ROOT / entry["directory"].lstrip("/") / "Dockerfile").read_text(
            encoding="utf-8"
        )
        if re.search(r"^FROM python:", dockerfile, flags=re.M):
            assert _ignored(entry).get("python") == {MAJOR, MINOR}, entry["directory"]


def test_the_python_version_in_the_images_is_the_one_the_locks_were_built_for():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    built_for = re.search(r"--python-version (\d+\.\d+)", lock).group(1)

    for dockerfile in ROOT.rglob("Dockerfile"):
        if SKIPPED_FOLDERS & set(dockerfile.relative_to(ROOT).parts):
            continue
        for version in re.findall(r"^FROM python:(\d+\.\d+)", dockerfile.read_text(), flags=re.M):
            assert version == built_for, dockerfile


def test_airflow_cannot_jump_a_major_version():
    (airflow,) = [e for e in _entries("docker") if e["directory"] == "/services/airflow"]

    assert _ignored(airflow) == {"apache/airflow": {MAJOR}}


def test_no_image_compose_starts_can_jump_a_major_version():
    (compose,) = _entries("docker-compose")
    images = re.findall(r"^\s+image: ([^\s:]+):", COMPOSE, flags=re.M)
    started = {name for name in images if not name.startswith("retail-")}  # built here: not ours

    # every image pulled from a registry has a decision; a new one fails here until it has one
    assert started == {"postgres", "redis", "prom/prometheus", "grafana/grafana-oss"}
    assert {name for name, kinds in _ignored(compose).items() if kinds == {MAJOR}} == started


def test_a_major_postgres_upgrade_in_particular_is_blocked():
    (compose,) = _entries("docker-compose")

    assert _ignored(compose)["postgres"] == {MAJOR}  # it would not open the existing data volume


def test_actions_updates_arrive_in_one_pull_request_not_one_per_action():
    (actions,) = _entries("github-actions")

    assert actions["groups"]["github-actions"]["patterns"] == ["*"]


def test_the_file_explains_what_it_leaves_out_and_why():
    for explanation in (
        "NOT here",
        "Python 3.12",
        "migration",
        "locked by hash",
        "Dependency graph",
    ):
        assert explanation in CONFIG_TEXT


def test_the_file_uses_no_yaml_anchors_which_dependabot_may_not_accept():
    assert "&" not in re.sub(r"#.*", "", CONFIG_TEXT) and "*docker" not in CONFIG_TEXT
