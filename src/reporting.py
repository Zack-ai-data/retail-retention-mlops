"""What a training run logs to MLflow: figures, importances, parameters and the gate decision."""

import hashlib
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from mlflow.data.pandas_dataset import from_pandas
from sklearn.inspection import permutation_importance

from src import (
    config,
    features,
    gate,
    model_search,
)


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data, logged so a run can be tied to the exact data it saw."""
    hashed = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def log_figure(fig: Figure, name: str) -> None:
    mlflow.log_figure(fig, name)
    plt.close(fig)


def importance_figure(importance: pd.DataFrame, title: str) -> Figure:
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3b6ea5")
    ax.set_xlabel("Drop in score when the feature is shuffled")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def calibration_figure(y_true: npt.ArrayLike, prob: npt.ArrayLike, bins: int = 10) -> Figure:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(prob, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(observed=("y", "mean"), predicted=("p", "mean"))
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="perfectly calibrated")
    ax.plot(grouped["predicted"], grouped["observed"], "o-", color="#3b6ea5", label="model")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on the test snapshot")
    ax.legend()
    fig.tight_layout()
    return fig


def feature_importance(model: Any, X: pd.DataFrame, y: npt.ArrayLike, scoring: Any) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=5, random_state=config.RANDOM_STATE
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def log_common(
    df_used: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selection: model_search.Selection,
    target: str,
    lineage: Mapping[str, str] | None = None,
) -> None:
    """Parameters, lineage tags and the datasets a training run used."""
    mlflow.log_params(
        {
            "algorithm": selection.algorithm,
            "train_snapshots": ",".join(str(d) for d in sorted(train["snapshot_date"].unique())),
            "test_snapshot": str(test["snapshot_date"].iloc[0]),
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "n_features": len(features.FEATURE_COLUMNS),
            "population": f"recency_days<={features.ACTIONABLE_RECENCY_DAYS}",
            **{f"hp_{key}": value for key, value in selection.params.items()},
        }
    )
    mlflow.set_tag("data_fingerprint", data_fingerprint(df_used))
    if lineage:
        mlflow.set_tags(dict(lineage))
    mlflow.log_input(from_pandas(train, name="train snapshots", targets=target), context="training")
    mlflow.log_input(from_pandas(test, name="test snapshot", targets=target), context="testing")
    mlflow.log_dict({"features": features.FEATURE_COLUMNS}, "features.json")
    mlflow.log_table(selection.results, "cv_results.json")


def log_gate(checks: list[gate.Check], decision: gate.Decision) -> None:
    mlflow.log_dict(
        {
            "promoted": decision.promoted,
            "reason": decision.reason,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        },
        "gate.json",
    )
