"""RFM customer segments: unsupervised KMeans groups as of the score snapshot."""

from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import psycopg
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import (
    config,
    database,
    tracking,
)

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
SEGMENT_K_RANGE = range(3, 8)


def segment_pipeline(k: int) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE),
    )


def name_segments(
    profile: pd.DataFrame, centroids_scaled: npt.NDArray[np.float64]
) -> dict[int, str]:
    """Readable segment names built from value level and recency, e.g. 'High value / Active'.

    value level: rank of the cluster's frequency + monetary centroid (top third = High)
    recency:     the cluster's median days since last purchase
                 (<=60 Active, <=180 Cooling, else Lapsed)
    """
    k = len(profile)
    value_score = (
        centroids_scaled[:, 1] + centroids_scaled[:, 2]
    )  # frequency + monetary (scaled, logged)
    rank = (-value_score).argsort().argsort()  # 0 = most valuable cluster
    tiers = ["High", "Mid", "Low"]

    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for cluster in range(k):
        median_recency = profile.loc[cluster, "median_recency_days"]
        recency = (
            "Active" if median_recency <= 60 else "Cooling" if median_recency <= 180 else "Lapsed"
        )
        base = f"{tiers[min(2, int(rank[cluster] * 3 / k))]} value / {recency}"
        used[base] = used.get(base, 0) + 1
        names[cluster] = base if used[base] == 1 else f"{base} #{used[base]}"
    return names


def train_segments(
    df: pd.DataFrame,
    conn: psycopg.Connection[Any] | None,
    client: MlflowClient,
    lineage_tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Cluster customers on recency, frequency and monetary value as of the score snapshot."""
    scoring = df[df["split"] == "score"].reset_index(drop=True)
    if len(scoring) < max(SEGMENT_K_RANGE) * 10:
        raise ValueError("Too few customers in the score snapshot to segment")
    snapshot_date = scoring["snapshot_date"].iloc[0]
    X = scoring[RFM_COLUMNS].astype(float)
    # net monetary value can be zero or negative (returns exceed purchases); the log needs >= 0
    X["monetary"] = X["monetary"].clip(lower=0)

    # Choose k by silhouette in the same log-scaled space the clusters are formed in
    scaled = make_pipeline(FunctionTransformer(np.log1p), StandardScaler()).fit_transform(X)
    silhouettes: dict[int, float] = {}
    for k in SEGMENT_K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(
            scaled
        )
        silhouettes[k] = float(silhouette_score(scaled, labels))
    best_k = max(silhouettes, key=lambda k: silhouettes[k])

    pipeline = segment_pipeline(best_k).fit(X)
    segment_ids = pipeline.predict(X)
    kmeans = pipeline[-1]

    profile = (
        X.assign(segment_id=segment_ids)
        .groupby("segment_id")
        .agg(
            customers=("frequency", "size"),
            median_recency_days=("recency_days", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
            total_monetary=("monetary", "sum"),
        )
    )
    profile["share_of_customers"] = profile["customers"] / profile["customers"].sum()
    profile["share_of_revenue"] = profile["total_monetary"] / profile["total_monetary"].sum()
    names = name_segments(profile, kmeans.cluster_centers_)
    profile.insert(0, "segment_name", profile.index.map(names))

    with mlflow.start_run(run_name="rfm-segments") as run:
        if lineage_tags:
            mlflow.set_tags(dict(lineage_tags))
        mlflow.log_params(
            {
                "algorithm": "kmeans",
                "k": best_k,
                "snapshot_date": str(snapshot_date),
                "n_customers": len(scoring),
                "features": ",".join(RFM_COLUMNS),
            }
        )
        mlflow.log_metrics(
            {"silhouette": silhouettes[best_k], "inertia": float(kmeans.inertia_)}
            | {f"silhouette_k{k}": s for k, s in silhouettes.items()}
        )
        mlflow.log_table(profile.reset_index(), "segment_profile.json")
        mlflow.log_dict({str(i): name for i, name in names.items()}, "segment_names.json")
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X, segment_ids),
            input_example=X.head(5),
            serialization_format=config.SERIALIZATION,
            registered_model_name=config.SEGMENT_MODEL_NAME,
        )
        # Clustering has no ground truth to gate on, so the newest version simply becomes champion
        client.set_registered_model_alias(
            config.SEGMENT_MODEL_NAME, tracking.CHAMPION_ALIAS, info.registered_model_version
        )
        client.set_model_version_tag(
            config.SEGMENT_MODEL_NAME,
            info.registered_model_version,
            "promotion",
            "promoted: unsupervised model, newest version is champion",
        )

    if conn is not None:
        save_segments(
            conn,
            snapshot_date,
            scoring["customer_id"],
            segment_ids,
            names,
            info.registered_model_version,
        )
    return {
        "model": config.SEGMENT_MODEL_NAME,
        "run_id": run.info.run_id,
        "k": best_k,
        "metrics": {"silhouette": silhouettes[best_k]},
        "profile": profile,
    }


def save_segments(
    conn: psycopg.Connection[Any],
    snapshot_date: Any,
    customer_ids: pd.Series,
    segment_ids: np.ndarray,
    names: dict[int, str],
    model_version: str,
) -> None:
    """Replace marts.customer_segments with the latest assignment (all-or-nothing)."""
    frame = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "customer_id": customer_ids.astype(int).to_numpy(),
            "segment_id": segment_ids,
            "segment_name": [names[int(s)] for s in segment_ids],
            "model_version": str(model_version),
        }
    )
    conn.commit()
    with conn.transaction():
        conn.execute("DELETE FROM marts.customer_segments")
        database.copy_dataframe(conn, frame, "marts.customer_segments")
    conn.commit()
