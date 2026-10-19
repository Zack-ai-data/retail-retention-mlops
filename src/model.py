"""Model training and evaluation, tracked in MLflow.

Three models, all trained from marts.customer_snapshots:
  - churn classifier      will the customer buy nothing in the next 90 days?
  - value model           how much will the customer spend in the next 90 days (expected value)?
                          a two-part model: P(buys) from the churn model x E[spend | buys]
  - RFM segments          unsupervised customer groups (recency / frequency / monetary)

Supervised models are selected by cross-validation over the training snapshots only
(each fold trains on earlier snapshots and validates on the next one). The test snapshot
is used once, after selection, so the reported test metrics are an honest estimate.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import psycopg
from mlflow import MlflowClient
from mlflow.models import infer_signature
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
)

from src import (
    backtesting,
    calibration,
    config,
    evaluation,
    features,
    gate,
    hurdle,
    lineage,
    measures,
    model_search,
    reporting,
    segmentation,
    tracking,
)

log = logging.getLogger(__name__)


KEY_METRIC_PREFIXES = ("test_pr_auc", "test_roc", "test_rmse", "test_mae", "silhouette")


def split_labelled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test rows of the actionable population (features.ACTIONABLE_RECENCY_DAYS)."""
    actionable = df[df["is_actionable"].astype(bool)]
    train = actionable[actionable["split"] == "train"].reset_index(drop=True)
    test = actionable[actionable["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Need actionable rows in both the 'train' and 'test' splits")
    return train, test


def champion_reference(
    champion: tracking.Champion | None, score: Callable[[Any], float]
) -> tuple[float | None, list[str]]:
    """The champion's score on the data the candidate is judged on, and notes on why not."""
    if champion is None:
        return None, ["no champion yet"]
    if champion.model is None:
        return None, [
            f"champion v{champion.version} was not compared because {champion.incompatible_reason}"
        ]
    return score(champion.model), [f"champion v{champion.version} re-scored on the same data"]


# ------------------------------------------------------------ churn model


def train_churn(
    df: pd.DataFrame, client: MlflowClient, lineage_tags: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Train, evaluate, log and (if it passes the quality gate) promote the churn classifier.

    The classifier is chosen by cross-validation on the training snapshots, calibrated on its
    out-of-fold predictions, and judged on the test snapshot against ranking by recency, with
    a confidence interval on the gain, a backtest over earlier origins, and the current
    champion re-scored on the same data.
    """
    train, test = split_labelled(df)
    X_train, X_test = features.prepare_features(train), features.prepare_features(test)
    y_train = train[features.CHURN_TARGET].to_numpy(dtype=int)
    y_test = test[features.CHURN_TARGET].to_numpy(dtype=int)

    selection = model_search.select_model(
        model_search.CHURN_CANDIDATES,
        X_train,
        y_train,
        train["snapshot_date"],
        score_fn=average_precision_score,
        predict_fn=lambda model, X: model.predict_proba(X)[:, 1],
        higher_is_better=True,
    )
    factory = model_search.candidate_factory(model_search.CHURN_CANDIDATES, selection.algorithm)
    base = factory(selection.params).fit(X_train, y_train)
    oof_y, oof_prob, oof_snapshots = model_search.out_of_fold_predictions(
        factory,
        selection.params,
        X_train,
        y_train,
        train["snapshot_date"],
        lambda model, X: model.predict_proba(X)[:, 1],
    )
    slope, intercept, calibrated = calibration.fit_if_it_helps(oof_y, oof_prob, oof_snapshots)
    model = calibration.PlattCalibratedClassifier(base, slope, intercept)
    raw_prob = base.predict_proba(X_test)[:, 1]
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_pr_auc": selection.cv_score,
        "test_base_rate": float(y_test.mean()),  # PR-AUC of a model with no skill
        "test_mean_predicted_prob": float(prob.mean()),  # vs base rate: shows calibration shift
        "calibration_applied": float(calibrated),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    metrics.update(measures.churn_metrics(y_test, prob, "test_"))
    metrics.update(
        {
            f"uncalibrated_test_{name}": value
            for name, value in measures.churn_metrics(y_test, raw_prob, "").items()
            if name in ("ece", "brier", "log_loss")
        }
    )
    metrics["test_pr_auc_lift"] = metrics["test_pr_auc"] / metrics["test_base_rate"]
    # Naive baseline: the longer since the last purchase, the likelier the customer is to churn
    recency = X_test["recency_days"]
    metrics.update(measures.ranking_metrics(y_test, recency, "baseline_recency_test_"))
    metrics.update(measures.recency_band_metrics(y_test, prob, recency, "test_"))
    metrics.update(
        measures.recency_band_metrics(y_test, recency, recency, "baseline_recency_test_")
    )
    # Statistical baseline: BG/NBD, which uses only purchase counts and timing
    bgnbd = backtesting.bgnbd_churn_scores(df, test)
    if bgnbd is not None:
        metrics.update(measures.ranking_metrics(y_test, bgnbd, "baseline_bgnbd_test_"))

    gain = evaluation.paired_bootstrap(y_test, prob, recency, average_precision_score)
    metrics.update(gain.to_metrics("test_pr_auc_vs_recency_"))
    backtest = backtesting.churn_backtest(pd.concat([train, test]), factory, selection.params)
    metrics.update(backtesting.backtest_summary(backtest))

    champion = tracking.load_champion(client, config.CHURN_MODEL_NAME, features.FEATURE_COLUMNS)
    champion_pr_auc, notes = champion_reference(
        champion,
        lambda m: float(average_precision_score(y_test, m.predict_proba(X_test)[:, 1])),
    )
    if champion_pr_auc is not None:
        metrics["champion_rescored_test_pr_auc"] = champion_pr_auc
    checks = gate.churn_checks(gain, metrics["backtest_share_significant"], champion_pr_auc)
    decision = gate.decide(checks, notes)

    with mlflow.start_run(run_name="churn-classifier") as run:
        reporting.log_common(
            pd.concat([train, test]), train, test, selection, features.CHURN_TARGET, lineage_tags
        )
        mlflow.log_metrics(metrics)
        mlflow.log_table(backtest, "backtest.json")
        reporting.log_gate(checks, decision)
        importance = reporting.feature_importance(base, X_test, y_test, "average_precision")
        mlflow.log_table(importance, "reporting.feature_importance.json")
        reporting.log_figure(
            reporting.importance_figure(importance, "Churn model: feature importance"),
            "reporting.feature_importance.png",
        )
        reporting.log_figure(reporting.calibration_figure(y_test, prob), "calibration.png")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, prob),
            input_example=X_test.head(5),
            serialization_format=config.SERIALIZATION,
            registered_model_name=config.CHURN_MODEL_NAME,
        )
        tracking.record_decision(
            client,
            config.CHURN_MODEL_NAME,
            info.registered_model_version,
            run.info.run_id,
            decision,
        )
    return {
        "model": config.CHURN_MODEL_NAME,
        "run_id": run.info.run_id,
        "metrics": metrics,
        "algorithm": selection.algorithm,
        "promoted": decision.promoted,
        "reason": decision.reason,
        "estimator": model,  # the value model reuses it as its "will they buy" stage
    }


# ------------------------------------------------------------ value model


def train_value(
    df: pd.DataFrame,
    client: MlflowClient,
    churn_model: Any,
    lineage_tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, log and (if it passes the quality gate) promote the value model.

    The target is the customer's spend over the next 90 days (0 for customers who do not buy).
    The model is two parts: the churn model says how likely the customer is to buy, and a gamma
    regressor trained only on customers who bought says how much they spend when they do (chosen
    by cross-validated gamma deviance). A single-stage Poisson model is trained alongside and
    reported for comparison; it does not take part in the gate. The gate compares the two-part
    model with a naive "next 90 days = last 90 days" forecast on ranking, bias and error, and
    with the current champion re-scored on the same data (see gate.value_checks).
    """
    train, test = split_labelled(df)
    X_train, X_test = features.prepare_features(train), features.prepare_features(test)
    y_train = train[features.VALUE_TARGET].to_numpy(dtype=float)
    spend_test = test[features.VALUE_TARGET].to_numpy(dtype=float)

    buyers = y_train > 0
    selection = model_search.select_model(
        model_search.SPEND_CANDIDATES,
        X_train[buyers],
        y_train[buyers],
        train["snapshot_date"][buyers],
        score_fn=measures.gamma_deviance,
        predict_fn=lambda model, X: model.predict(X),
        higher_is_better=False,
    )
    spend_model = model_search.candidate_factory(
        model_search.SPEND_CANDIDATES, selection.algorithm
    )(selection.params).fit(X_train[buyers], y_train[buyers])
    model = hurdle.HurdleValueModel(churn_model, spend_model)
    predicted = np.clip(model.predict(X_test), 0, None)

    single_selection = model_search.select_model(
        model_search.VALUE_CANDIDATES,
        X_train,
        y_train,
        train["snapshot_date"],
        score_fn=measures.poisson_deviance,
        predict_fn=lambda model, X: model.predict(X),
        higher_is_better=False,
    )
    single_stage = model_search.candidate_factory(
        model_search.VALUE_CANDIDATES, single_selection.algorithm
    )(single_selection.params).fit(X_train, y_train)
    # Naive baseline: assume the next 90 days look like the last 90 days (spend cannot be negative)
    naive = np.clip(X_test["spend_last_90d"].to_numpy(), 0, None)

    metrics = {
        "cv_gamma_deviance": selection.cv_score,
        "single_stage_cv_poisson_deviance": single_selection.cv_score,
        "test_mean_spend": float(spend_test.mean()),
    }
    metrics.update(measures.value_metrics(spend_test, predicted, "test_"))
    metrics.update(measures.value_metrics(spend_test, naive, "baseline_naive_test_"))
    metrics.update(
        measures.value_metrics(
            spend_test, np.clip(single_stage.predict(X_test), 0, None), "single_stage_test_"
        )
    )
    # Where the two parts are right or wrong: how often customers buy, and what buyers spend
    bought = spend_test > 0
    metrics["test_buy_probability_mean"] = float(model.predict_buy_probability(X_test).mean())
    metrics["test_buy_rate"] = float(bought.mean())
    metrics["test_spend_if_buying_predicted_mean"] = float(
        model.predict_spend_if_buying(X_test)[bought].mean()
    )
    metrics["test_spend_if_buying_actual_mean"] = float(spend_test[bought].mean())

    # The error guard judges RMSE on spend capped at a training quantile, so a handful of very
    # large wholesale accounts cannot decide whether the model is usable for ranking.
    cap = float(np.quantile(y_train, gate.WINSORISE_QUANTILE))

    def neg_winsorised_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
        return -float(np.sqrt(np.mean((forecast - np.minimum(actual, cap)) ** 2)))

    def spearman(actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(spearmanr(actual, forecast).statistic)

    ranking = evaluation.paired_bootstrap(spend_test, predicted, naive, spearman)
    capture = evaluation.paired_bootstrap(
        spend_test, predicted, naive, measures.top_fraction_capture
    )
    error_guard = evaluation.paired_bootstrap(spend_test, predicted, naive, neg_winsorised_rmse)
    metrics["winsorise_cap"] = cap
    metrics["test_winsorised_rmse"] = -error_guard.candidate
    metrics["baseline_naive_test_winsorised_rmse"] = -error_guard.reference
    metrics.update(ranking.to_metrics("test_spearman_vs_naive_"))
    metrics.update(capture.to_metrics("test_top_decile_capture_vs_naive_"))
    metrics.update(error_guard.to_metrics("test_neg_winsorised_rmse_vs_naive_"))

    champion = tracking.load_champion(client, config.VALUE_MODEL_NAME, features.FEATURE_COLUMNS)
    champion_spearman, notes = champion_reference(
        champion,
        lambda m: spearman(spend_test, np.clip(m.predict(X_test), 0, None)),
    )
    if champion_spearman is not None:
        metrics["champion_rescored_test_spearman"] = champion_spearman
    checks = gate.value_checks(
        ranking,
        capture,
        bias_pct=metrics["test_bias_pct"],
        naive_bias_pct=metrics["baseline_naive_test_bias_pct"],
        error_guard=error_guard,
        champion_spearman=champion_spearman,
    )
    decision = gate.decide(checks, notes)

    with mlflow.start_run(run_name="value-regressor") as run:
        reporting.log_common(
            pd.concat([train, test]), train, test, selection, features.VALUE_TARGET, lineage_tags
        )
        mlflow.log_params(
            {
                "hurdle_buy_stage": type(churn_model).__name__,
                "single_stage_algorithm": single_selection.algorithm,
            }
        )
        mlflow.log_table(single_selection.results, "single_stage_cv_results.json")
        mlflow.log_metrics(metrics)
        reporting.log_gate(checks, decision)
        # importance in terms of what the model is used for: how well it ranks customers
        ranking_scorer = make_scorer(lambda actual, forecast: spearman(actual, forecast))
        importance = reporting.feature_importance(model, X_test, spend_test, ranking_scorer)
        mlflow.log_table(importance, "reporting.feature_importance.json")
        reporting.log_figure(
            reporting.importance_figure(
                importance, "Value model: feature importance (effect on ranking)"
            ),
            "reporting.feature_importance.png",
        )
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, predicted),
            input_example=X_test.head(5),
            serialization_format=config.SERIALIZATION,
            registered_model_name=config.VALUE_MODEL_NAME,
        )
        tracking.record_decision(
            client,
            config.VALUE_MODEL_NAME,
            info.registered_model_version,
            run.info.run_id,
            decision,
        )
    return {
        "model": config.VALUE_MODEL_NAME,
        "run_id": run.info.run_id,
        "metrics": metrics,
        "algorithm": f"hurdle_{selection.algorithm}",
        "promoted": decision.promoted,
        "reason": decision.reason,
        "estimator": model,
    }


# ----------------------------------------------------------- orchestration


def train_all(
    conn: psycopg.Connection[Any], tracking_uri: str | None = None
) -> list[dict[str, Any]]:
    """Train the churn, value and segmentation models from the warehouse."""
    client = tracking.configure(tracking_uri)
    df = features.load_snapshots(conn)
    log.info(
        "Loaded %s snapshot rows (%s)", f"{len(df):,}", ", ".join(sorted(df["split"].unique()))
    )

    labelled = df[df["split"].isin(["train", "test"])]
    tags = lineage.run_tags(conn)
    log.info(
        "Lineage: git %s, raw file %s..., transform SQL %s...",
        tags["git_sha"][:10],
        tags["raw_file_sha256"][:10],
        tags["transform_sql_sha256"][:10],
    )
    churn = train_churn(labelled, client, tags)
    results = [
        churn,
        train_value(labelled, client, churn["estimator"], tags),
        segmentation.train_segments(df, conn, client, tags),
    ]
    for result in results:
        shown = {
            k: round(v, 4)
            for k, v in result["metrics"].items()
            if k.startswith(KEY_METRIC_PREFIXES)
        }
        log.info("%s: %s", result["model"], shown)
    return results
