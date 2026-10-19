"""Recording the promotion decision, and loading the champion so it can be re-scored.

record_decision is tested against a recording stand-in for MLflow; load_champion against a real
(throwaway SQLite) model registry.
"""

import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from sklearn.linear_model import LogisticRegression

from src import gate, tracking
from src.errors import RegistryError

PROMOTE = gate.Decision(True, "all checks passed")
REJECT = gate.Decision(False, "failed: something")


class RecordingClient:
    """Stands in for MlflowClient: records what it was asked."""

    def __init__(self):
        self.calls = []

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append(("set_alias", name, alias, version))

    def set_model_version_tag(self, name, version, key, value):
        self.calls.append(("version_tag", name, version, key, value))

    def set_tag(self, run_id, key, value):
        self.calls.append(("run_tag", run_id, key, value))


# ---------------------------------------------------------- record_decision


def test_a_promoted_model_becomes_champion_and_the_decision_is_recorded():
    client = RecordingClient()

    tracking.record_decision(client, "m", "3", "run-9", PROMOTE)

    decision = "promoted: all checks passed"
    assert client.calls == [  # the reason is recorded first, then the alias moves
        ("version_tag", "m", "3", "promotion", decision),
        ("run_tag", "run-9", "promotion", decision),
        ("set_alias", "m", "champion", "3"),
    ]


def test_a_rejected_model_is_recorded_without_touching_the_champion_alias():
    client = RecordingClient()

    tracking.record_decision(client, "m", "3", "run-9", REJECT)

    decision = "rejected: failed: something"
    assert client.calls == [
        ("version_tag", "m", "3", "promotion", decision),
        ("run_tag", "run-9", "promotion", decision),
    ]


def test_the_promotion_decision_is_logged_with_the_model_and_version(caplog):
    with caplog.at_level(logging.INFO, logger=tracking.log.name):
        tracking.record_decision(RecordingClient(), "my-model", "7", "run-7", PROMOTE)

    assert "my-model v7 promoted" in caplog.text


class _ClientThatCannotWriteTags(RecordingClient):
    def set_model_version_tag(self, name, version, key, value):
        raise RuntimeError("registry write failed")


def test_if_the_decision_cannot_be_recorded_the_alias_does_not_move():
    client = _ClientThatCannotWriteTags()

    with pytest.raises(RuntimeError, match="registry write failed"):
        tracking.record_decision(client, "m", "5", "run-5", PROMOTE)

    assert not any(
        call[0] == "set_alias" for call in client.calls
    )  # no model goes live unexplained


# ------------------------------------------------------------ load_champion

FEATURES = ["a", "b"]


def _register(client, name, columns, alias=True, with_signature=True):
    """Train a tiny model on the given columns, register it and (optionally) make it champion."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(columns))), columns=columns)
    y = (X.iloc[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X, model.predict_proba(X)[:, 1]) if with_signature else None,
            serialization_format="cloudpickle",
            registered_model_name=name,
        )
    if alias:
        client.set_registered_model_alias(
            name, tracking.CHAMPION_ALIAS, info.registered_model_version
        )
    return info.registered_model_version


def test_no_champion_means_none(mlflow_client):
    assert tracking.load_champion(mlflow_client, "never-registered", FEATURES) is None


def test_a_registered_model_without_the_champion_alias_is_not_a_champion(mlflow_client):
    _register(mlflow_client, "m", FEATURES, alias=False)

    assert tracking.load_champion(mlflow_client, "m", FEATURES) is None


def test_the_champion_is_loaded_so_it_can_score_current_data(mlflow_client):
    version = _register(mlflow_client, "m", FEATURES)

    champion = tracking.load_champion(mlflow_client, "m", FEATURES)

    assert champion is not None and champion.model is not None
    assert champion.version == str(version)
    assert champion.incompatible_reason is None
    scores = champion.model.predict_proba(pd.DataFrame({"a": [1.0, -1.0], "b": [0.0, 0.0]}))
    assert scores.shape == (2, 2)


def test_a_champion_trained_on_other_columns_is_reported_not_loaded(mlflow_client):
    _register(mlflow_client, "m", ["a", "old_feature"])

    champion = tracking.load_champion(mlflow_client, "m", FEATURES)

    assert champion is not None
    assert champion.model is None
    assert "old_feature" in champion.incompatible_reason  # what the champion needs and we lack
    assert "'b'" in champion.incompatible_reason  # what is new


def test_a_champion_without_a_signature_is_an_error_not_no_champion(mlflow_client):
    _register(mlflow_client, "m", FEATURES, with_signature=False)

    with pytest.raises(RegistryError, match="no input signature"):
        tracking.load_champion(mlflow_client, "m", FEATURES)


def test_a_champion_whose_model_cannot_be_read_is_an_error(mlflow_client, monkeypatch):
    _register(mlflow_client, "m", FEATURES)

    def broken(uri):
        raise OSError("artifact store unavailable")

    monkeypatch.setattr(mlflow.models, "get_model_info", broken)

    with pytest.raises(RegistryError, match="Cannot read"):
        tracking.load_champion(mlflow_client, "m", FEATURES)


def test_a_champion_that_cannot_be_unpickled_is_an_error(mlflow_client, monkeypatch):
    _register(mlflow_client, "m", FEATURES)

    def broken(uri):
        raise ValueError("cannot unpickle")

    monkeypatch.setattr(mlflow.sklearn, "load_model", broken)

    with pytest.raises(RegistryError, match="Cannot load"):
        tracking.load_champion(mlflow_client, "m", FEATURES)


class _ClientWhoseLookupFails:
    def __init__(self, code):
        self.code = code

    def get_registered_model(self, name):
        raise MlflowException("simulated", error_code=self.code)


def test_a_model_that_does_not_exist_in_the_registry_means_none():
    client = _ClientWhoseLookupFails(RESOURCE_DOES_NOT_EXIST)

    assert tracking.load_champion(client, "some-model", FEATURES) is None


def test_a_server_error_is_not_mistaken_for_no_champion():
    # Treating "the server failed" as "no champion yet" would let any model be promoted.
    client = _ClientWhoseLookupFails(INTERNAL_ERROR)

    with pytest.raises(MlflowException):
        tracking.load_champion(client, "some-model", FEATURES)
