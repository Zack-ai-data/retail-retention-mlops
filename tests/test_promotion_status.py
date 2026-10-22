"""Reading back what the promotion gate decided, and the `report` stage built on it.

The decisions are written with tracking.record_decision into a real (throwaway SQLite) model
registry and read back with promotion_status, so the two cannot drift apart.
"""

import logging
from types import SimpleNamespace

import mlflow
import pytest

from src import config, gate, main, promotion_status, tracking
from src.errors import RegistryError

PROMOTE = gate.Decision(True, "passed: gain over the baseline is significant")
REJECT = gate.Decision(False, "failed: gain over the baseline (interval includes zero)")


def _register(client, name, decision=None):
    """Register a new version of a model and record the gate's decision on it (if any)."""
    if client.search_registered_models(filter_string=f"name = '{name}'") == []:
        client.create_registered_model(name)
    with mlflow.start_run() as run:  # the decision is also tagged on the training run
        pass
    run_id = run.info.run_id
    version = client.create_model_version(name, source="file:///not-used", run_id=run_id).version
    if decision is not None:
        tracking.record_decision(client, name, version, run_id, decision)
    return str(version)  # the project treats versions as text, as the registry's alias map does


# ------------------------------------------------------------------ latest_outcome


def test_a_promoted_newest_version_is_reported_as_the_champion(mlflow_client):
    version = _register(mlflow_client, "m", PROMOTE)

    outcome = promotion_status.latest_outcome(mlflow_client, "m")

    assert outcome.promoted is True
    assert (outcome.version, outcome.champion) == (version, version)
    assert outcome.reason == "passed: gain over the baseline is significant"


def test_a_rejected_newest_version_leaves_the_earlier_champion_in_place(mlflow_client):
    first = _register(mlflow_client, "m", PROMOTE)
    second = _register(mlflow_client, "m", REJECT)

    outcome = promotion_status.latest_outcome(mlflow_client, "m")

    assert outcome.promoted is False
    assert (outcome.version, outcome.champion) == (second, first)
    assert "interval includes zero" in outcome.reason


def test_only_the_newest_version_counts_not_an_older_rejection(mlflow_client):
    _register(mlflow_client, "m", REJECT)
    second = _register(mlflow_client, "m", PROMOTE)

    outcome = promotion_status.latest_outcome(mlflow_client, "m")

    assert outcome.promoted is True and outcome.version == second


def test_a_rejected_first_model_means_there_is_no_champion(mlflow_client):
    _register(mlflow_client, "m", REJECT)

    outcome = promotion_status.latest_outcome(mlflow_client, "m")

    assert outcome.promoted is False and outcome.champion is None


def _fake_client(versions, aliases=None):
    """A registry stand-in: versions is a list of (version, tags)."""
    return SimpleNamespace(
        search_model_versions=lambda _filter: [
            SimpleNamespace(version=version, tags=tags) for version, tags in versions
        ],
        get_registered_model=lambda _name: SimpleNamespace(aliases=aliases or {}),
    )


def test_versions_are_compared_as_numbers_not_text():
    # as text "9" sorts after "10"; the newest is 10
    client = _fake_client(
        [("9", {"promotion": "promoted: old"}), ("10", {"promotion": "rejected: new"})],
        {"champion": "9"},
    )

    outcome = promotion_status.latest_outcome(client, "m")

    assert outcome.version == "10" and outcome.promoted is False


def test_a_model_that_was_never_registered_is_an_error_with_the_fix(mlflow_client):
    with pytest.raises(RegistryError, match="Train the models first"):
        promotion_status.latest_outcome(mlflow_client, "never-registered")


def test_a_version_without_a_decision_is_an_error_not_a_guess(mlflow_client):
    version = _register(mlflow_client, "m")  # registered, gate never recorded anything

    with pytest.raises(RegistryError, match=rf"version {version} has no promotion decision"):
        promotion_status.latest_outcome(mlflow_client, "m")


def test_a_decision_in_an_unknown_format_is_an_error():
    client = _fake_client([("1", {"promotion": "maybe: unsure"})])

    with pytest.raises(RegistryError, match="does not recognise"):
        promotion_status.latest_outcome(client, "m")


def test_a_promotion_whose_alias_was_moved_afterwards_is_an_error():
    client = _fake_client([("2", {"promotion": "promoted: ok"})], {"champion": "1"})

    with pytest.raises(RegistryError, match="Someone moved the alias"):
        promotion_status.latest_outcome(client, "m")


def test_the_project_models_are_the_three_the_pipeline_trains():
    assert promotion_status.MODEL_NAMES == (
        config.CHURN_MODEL_NAME,
        config.VALUE_MODEL_NAME,
        config.SEGMENT_MODEL_NAME,
    )


def test_every_project_model_is_reported_in_order(mlflow_client):
    for name in promotion_status.MODEL_NAMES:
        _register(mlflow_client, name, PROMOTE)

    outcomes = promotion_status.latest_outcomes(mlflow_client)

    assert [o.model for o in outcomes] == list(promotion_status.MODEL_NAMES)


# ------------------------------------------------------------------------ describe


def test_a_promotion_is_described_as_such():
    outcome = promotion_status.Outcome("m", "4", True, "4", "passed: fine")

    assert promotion_status.describe(outcome) == "m: version 4 promoted and is now the champion"


def test_a_rejection_names_the_champion_that_stays_and_the_reason():
    outcome = promotion_status.Outcome("m", "4", False, "3", "failed: too weak")

    text = promotion_status.describe(outcome)

    assert text == "m: version 4 REJECTED, the champion stays version 3. failed: too weak"


def test_a_rejection_with_no_champion_says_so():
    outcome = promotion_status.Outcome("m", "1", False, None, "failed: too weak")

    assert "there is no champion" in promotion_status.describe(outcome)


# ------------------------------------------------------------------- the report stage


def _outcomes(*promoted):
    return [
        promotion_status.Outcome(f"model-{i}", "4", ok, "4" if ok else "3", "reason")
        for i, ok in enumerate(promoted)
    ]


def _use(monkeypatch, outcomes):
    monkeypatch.setattr(tracking, "configure", lambda *a, **k: object())
    monkeypatch.setattr(promotion_status, "latest_outcomes", lambda client: outcomes)


def test_report_exits_zero_when_every_model_was_promoted(monkeypatch, caplog):
    _use(monkeypatch, _outcomes(True, True, True))

    with caplog.at_level(logging.INFO, logger="pipeline"):
        code = main.main(["--stage", "report"])

    assert code == 0
    assert "model-0: version 4 promoted" in caplog.text
    assert "Not promoted" not in caplog.text


def test_report_exits_99_and_names_the_models_when_any_was_rejected(monkeypatch, caplog):
    _use(monkeypatch, _outcomes(True, False, False))

    with caplog.at_level(logging.INFO, logger="pipeline"):
        code = main.main(["--stage", "report"])

    assert code == main.PROMOTION_REJECTED_EXIT_CODE == 99
    assert "Not promoted: model-1, model-2" in caplog.text
    assert "model-1: version 4 REJECTED, the champion stays version 3" in caplog.text


def test_report_fails_with_exit_1_when_the_registry_cannot_be_trusted(monkeypatch, caplog):
    monkeypatch.setattr(tracking, "configure", lambda *a, **k: object())

    def broken(client):
        raise RegistryError("version 4 has no promotion decision recorded")

    monkeypatch.setattr(promotion_status, "latest_outcomes", broken)

    with caplog.at_level(logging.ERROR, logger="pipeline"):
        code = main.main(["--stage", "report"])

    assert code == 1
    assert "no promotion decision recorded" in caplog.text


def test_report_is_not_part_of_running_every_stage(monkeypatch):
    calls = []
    for stage in ("load_raw", "transform", "train", "score"):
        monkeypatch.setattr(main, stage, lambda stage=stage: calls.append(stage))
    monkeypatch.setattr(main, "plan", lambda *a: calls.append("plan"))
    monkeypatch.setattr(main, "report_promotions", lambda: calls.append("report") or 0)

    assert main.main(["--stage", "all"]) == 0

    assert calls == ["load_raw", "transform", "train", "score", "plan"]
