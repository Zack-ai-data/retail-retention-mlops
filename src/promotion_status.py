"""What the promotion gate decided about the newest version of each model.

The train stage finishes successfully even when the gate rejects a new model: the previous
champion keeps serving, which is the safe outcome. A scheduler that only sees "exit 0" cannot tell
that from a promotion, so this module reads the decisions back from the registry (they are written
by tracking.record_decision) and says, per model, whether the newest version went live.
"""

from dataclasses import dataclass

from mlflow import MlflowClient

from src import config, tracking
from src.errors import RegistryError

MODEL_NAMES = (config.CHURN_MODEL_NAME, config.VALUE_MODEL_NAME, config.SEGMENT_MODEL_NAME)

_PROMOTED = "promoted:"
_REJECTED = "rejected:"


@dataclass(frozen=True)
class Outcome:
    """The decision on one model's newest version."""

    model: str
    version: str  # the newest registered version
    promoted: bool
    champion: str | None  # the version serving now; None when no version was ever promoted
    reason: str


def latest_outcome(client: MlflowClient, model_name: str) -> Outcome:
    """The gate's decision on the newest version of a model, and which version is serving.

    Raises RegistryError when there is nothing trustworthy to report: no version registered, a
    version with no decision recorded (the training run probably stopped part-way), a decision in
    a format this does not know, or a version recorded as promoted that is not the champion.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RegistryError(
            f"No version of {model_name!r} is registered. Train the models first: "
            "python -m src.main --stage train"
        )
    newest = max(versions, key=lambda v: int(v.version))
    decision = newest.tags.get("promotion")
    if decision is None:
        raise RegistryError(
            f"{model_name!r} version {newest.version} has no promotion decision recorded, so it "
            "is not known whether it went live. The training run may have stopped part-way."
        )
    if decision.startswith(_PROMOTED):
        promoted = True
    elif decision.startswith(_REJECTED):
        promoted = False
    else:
        raise RegistryError(
            f"{model_name!r} version {newest.version} has a promotion decision this code does "
            f"not recognise: {decision!r}"
        )
    aliases = client.get_registered_model(model_name).aliases
    champion = aliases.get(tracking.CHAMPION_ALIAS)
    champion = None if champion is None else str(champion)
    if promoted and champion != str(newest.version):
        raise RegistryError(
            f"{model_name!r} version {newest.version} is recorded as promoted, but the "
            f"{tracking.CHAMPION_ALIAS!r} alias points at version {champion}. "
            "Someone moved the alias after training."
        )
    reason = decision.split(":", 1)[1].strip()
    return Outcome(model_name, str(newest.version), promoted, champion, reason)


def latest_outcomes(client: MlflowClient) -> list[Outcome]:
    """The decision on the newest version of each of the project's models."""
    return [latest_outcome(client, name) for name in MODEL_NAMES]


def describe(outcome: Outcome) -> str:
    """One line saying what happened to a model, for a log."""
    if outcome.promoted:
        return f"{outcome.model}: version {outcome.version} promoted and is now the champion"
    serving = (
        f"the champion stays version {outcome.champion}"
        if outcome.champion is not None
        else "there is no champion"
    )
    return f"{outcome.model}: version {outcome.version} REJECTED, {serving}. {outcome.reason}"
