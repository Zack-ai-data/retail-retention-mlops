"""MLflow helpers: connect to the tracking server and decide which model version goes live."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from src import config, gate
from src.errors import RegistryError, TrackingServerError

log = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"  # the version the API serves: models:/<name>@champion


def configure(
    tracking_uri: str | None = None, artifact_location: str | None = None
) -> MlflowClient:
    """Point MLflow at the tracking server and select the project experiment.

    artifact_location is only needed when the experiment is created outside a tracking
    server (the server otherwise decides where artifacts go). Raises TrackingServerError when the
    server cannot be reached; the client fails after a few quick retries instead of hanging.
    """
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    try:
        mlflow.set_tracking_uri(uri)
        if artifact_location and mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
            mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=artifact_location)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    except (MlflowException, requests.exceptions.RequestException) as exc:
        raise TrackingServerError(
            f"Cannot use the MLflow tracking server at {uri}: {exc}. "
            "Is it running? Try: docker compose up -d mlflow"
        ) from exc
    return MlflowClient(uri)


@dataclass(frozen=True)
class Champion:
    """The model version currently serving, loaded so it can be scored on today's data."""

    version: str
    model: Any | None  # None when it cannot score the current feature columns
    incompatible_reason: str | None = None


def load_champion(
    client: MlflowClient, model_name: str, feature_columns: Sequence[str]
) -> Champion | None:
    """The current champion, ready to be re-scored on the same data as a new candidate.

    A new candidate must be compared with the champion on the data both are being judged on, not
    with the champion's metric as logged when it was trained: that was measured on a different
    evaluation set, so the comparison is unfair as soon as the data moves.

    None means there is no champion (no such model or alias). A champion trained on different
    feature columns than the current ones cannot score today's data; it is returned with
    model=None and the reason. Anything else that goes wrong (a server error, a champion whose
    model cannot be read) is raised: treating it as "no champion" would let any model replace
    the real one.
    """
    try:
        registered = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise
    # Read the alias map instead of asking for the alias: a model that exists without the alias
    # raises an error whose code is shared with genuine mistakes, so it cannot be told apart.
    champion_version = registered.aliases.get(CHAMPION_ALIAS)
    if champion_version is None:
        return None
    number = str(champion_version)
    uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    label = f"{model_name!r} version {number}"
    try:
        signature = mlflow.models.get_model_info(uri).signature
        names = signature.inputs.input_names() if signature and signature.inputs else None
    except Exception as exc:
        raise RegistryError(f"Cannot read the {CHAMPION_ALIAS!r} model {label}: {exc}") from exc
    if names is None:
        raise RegistryError(
            f"The {CHAMPION_ALIAS!r} model {label} has no input signature, so it cannot be "
            "re-scored. Refusing to treat that as 'no champion'; re-train it or move the alias."
        )
    if set(names) != set(feature_columns):
        added = sorted(set(feature_columns) - set(names))
        removed = sorted(set(names) - set(feature_columns))
        return Champion(
            number,
            None,
            f"it was trained on different feature columns (now missing: {removed}; new: {added})",
        )
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception as exc:
        raise RegistryError(f"Cannot load the {CHAMPION_ALIAS!r} model {label}: {exc}") from exc
    return Champion(number, model)


def record_decision(
    client: MlflowClient,
    model_name: str,
    version: str,
    run_id: str,
    decision: gate.Decision,
) -> None:
    """Record the gate's decision and, if it promoted the model, move the champion alias.

    The decision is recorded as a tag on both the model version and the training run, so a
    rejected model stays in the registry with the reason it was rejected. The tags are written
    first: if recording fails, the alias has not moved, so no model goes live without a reason.
    """
    text = f"{'promoted' if decision.promoted else 'rejected'}: {decision.reason}"
    client.set_model_version_tag(model_name, version, "promotion", text)
    client.set_tag(run_id, "promotion", text)
    if decision.promoted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    log.info("%s v%s %s", model_name, version, text)
