"""The CI workflow's triggers: the weekly schedule, and that it cannot be cancelled by a push.

GitHub runs a schedule only from the default branch and only on its own clock, so the trigger itself
cannot be exercised before it is merged. What can be checked here is what it is set to, that the
jobs it should run are not skipped, and that nothing else about CI changed.
"""

import re

import pytest
import yaml

from src import config

ROOT = config.PROJECT_ROOT
CI_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
CI = yaml.safe_load(CI_TEXT)
ON = CI[True]  # YAML 1.1 reads the key `on` as the boolean true
MALAYSIA_OFFSET_HOURS = 8  # UTC+8, no daylight saving


def _cron_fields():
    (entry,) = ON["schedule"]
    minute, hour, day_of_month, month, day_of_week = entry["cron"].split()
    return minute, hour, day_of_month, month, day_of_week


def test_there_is_exactly_one_schedule_and_it_is_weekly_on_monday():
    assert len(ON["schedule"]) == 1
    minute, hour, day_of_month, month, day_of_week = _cron_fields()

    assert (day_of_month, month, day_of_week) == ("*", "*", "1")  # every Monday
    assert minute.isdigit() and hour.isdigit()


def test_it_runs_at_ten_in_the_morning_in_malaysia():
    # GitHub reads the time as UTC: 02:00 UTC is 10:00 in Malaysia (UTC+8)
    minute, hour, *_ = _cron_fields()

    assert (int(hour) + MALAYSIA_OFFSET_HOURS) % 24 == 10 and int(minute) == 0


def test_the_other_triggers_are_unchanged():
    assert ON["push"] == {"branches": ["main", "master"]}
    assert "pull_request" in ON and "workflow_dispatch" in ON  # the manual run is how to try it


def test_a_push_to_main_cannot_cancel_the_weekly_run():
    concurrency = CI["concurrency"]

    # both use the main branch as their ref, so the kind of trigger must be part of the group name
    assert "github.event_name" in concurrency["group"] and "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True  # a newer push still replaces an older one


def test_the_slow_jobs_run_on_the_schedule_and_only_pull_requests_skip_them():
    for job in ("full-training", "regression"):
        assert CI["jobs"][job]["if"] == "github.event_name != 'pull_request'"
    for job in ("quality", "dashboard", "airflow-dags"):
        assert "if" not in CI["jobs"][job]  # these run on every trigger, the schedule included


def test_the_weekly_run_has_the_same_five_jobs_as_any_other():
    assert set(CI["jobs"]) == {
        "quality",
        "dashboard",
        "airflow-dags",
        "full-training",
        "regression",
    }


def test_the_workflow_needs_no_secret_and_can_only_read_the_repository():
    assert CI["permissions"] == {"contents": "read"}
    assert "secrets." not in CI_TEXT


@pytest.mark.parametrize(
    "fact", ["02:00 UTC", "60 days", "public", "reproducibility", "NOT pinned"]
)
def test_the_workflow_file_explains_what_the_schedule_is_and_is_not(fact):
    assert fact in CI_TEXT


def test_the_header_and_the_cron_line_agree():
    # the comment says Mondays at 02:00 UTC; the line must say the same
    assert re.search(r'cron: "0 2 \* \* 1"', CI_TEXT)
    assert "Mondays at 02:00 UTC" in CI_TEXT
