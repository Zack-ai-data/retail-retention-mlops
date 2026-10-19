"""The promotion gate: which evidence a new model version needs before it can go live.

Pure rules, no MLflow. Each model type turns its evaluation into a list of Checks and decide()
promotes only if every check passes. The rules follow docs/problem_statement.md:

  churn   the gain over the recency baseline must be real (its confidence interval excludes 0,
          on the test snapshot and on most backtest origins), and the model must not be worse
          than the current champion scored on the same data.
  value   the model is used to rank customers by expected value at risk, so it must rank better
          than the naive forecast, must have a total bias within +-MAX_ABS_BIAS_PCT of actual
          spend, and must not be significantly worse on error (RMSE on winsorised spend, so a
          few wholesale accounts cannot decide the verdict). It must also not rank worse than
          the current champion.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation import Gain

# "Most backtest origins" means strictly more than this share of them
MIN_SHARE_OF_ORIGINS_SIGNIFICANT = 0.5

# The value model's total bias (predicted minus actual spend, as a share of actual spend) must
# stay within this many percent either way. This is a sanity limit, not an accuracy target: it is
# wide enough for the seasonal level shift of predicting a holiday-peak quarter from a year of
# data (-32% at the last test snapshot; the naive last-quarter forecast is -27%) and narrow
# enough to catch a broken model (a log-target version was -78%). The figure was proposed after
# seeing those results and confirmed by the owner on 2026-10-19; see docs/problem_statement.md.
MAX_ABS_BIAS_PCT = 40.0

# Spend above this quantile of the training labels is capped when judging error (RMSE guard)
WINSORISE_QUANTILE = 0.99


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Decision:
    promoted: bool
    reason: str


def decide(checks: Sequence[Check], notes: Sequence[str] = ()) -> Decision:
    """Promote only if there is at least one check and every check passed."""
    if not checks:
        raise ValueError("A promotion decision needs at least one check")
    failed = [check for check in checks if not check.passed]
    passed = [check for check in checks if check.passed]
    if failed:
        reason = "failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed)
        if passed:
            reason += " | passed: " + ", ".join(c.name for c in passed)
    else:
        reason = "passed: " + "; ".join(f"{c.name} ({c.detail})" for c in passed)
    if notes:
        reason += " | " + "; ".join(notes)
    return Decision(promoted=not failed, reason=reason)


def _interval(gain: Gain) -> str:
    return f"gain {gain.diff:+.3f}, 95% CI {gain.low:+.3f} to {gain.high:+.3f}"


def churn_checks(
    vs_recency: Gain, share_of_origins_significant: float, champion_pr_auc: float | None
) -> list[Check]:
    checks = [
        Check(
            "beats the recency baseline",
            vs_recency.excludes_zero_above(),
            f"PR-AUC {vs_recency.candidate:.3f} vs {vs_recency.reference:.3f}, "
            f"{_interval(vs_recency)}",
        ),
        Check(
            "gain holds across backtest origins",
            share_of_origins_significant > MIN_SHARE_OF_ORIGINS_SIGNIFICANT,
            f"significant at {share_of_origins_significant:.0%} of origins, needs more than "
            f"{MIN_SHARE_OF_ORIGINS_SIGNIFICANT:.0%}",
        ),
    ]
    if champion_pr_auc is not None:
        checks.append(
            Check(
                "not worse than the champion",
                vs_recency.candidate >= champion_pr_auc,
                f"PR-AUC {vs_recency.candidate:.3f} vs champion {champion_pr_auc:.3f} "
                "on the same data",
            )
        )
    return checks


def value_checks(
    ranking: Gain,
    capture: Gain,
    bias_pct: float,
    naive_bias_pct: float,
    error_guard: Gain,
    champion_spearman: float | None,
) -> list[Check]:
    """ranking: Spearman gain over the naive forecast. capture: top-decile spend capture gain.
    error_guard: gain in negated RMSE on winsorised spend (positive = the model has less error).
    """
    rmse = -error_guard.candidate
    naive_rmse = -error_guard.reference
    checks = [
        Check(
            "ranks customers better than the naive forecast",
            ranking.excludes_zero_above(),
            f"Spearman {ranking.candidate:.3f} vs {ranking.reference:.3f}, {_interval(ranking)}",
        ),
        Check(
            "top-decile spend capture is not lower",
            capture.diff > 0,
            f"{capture.candidate:.3f} vs {capture.reference:.3f}, {_interval(capture)}",
        ),
        Check(
            "total bias is within the allowed bound",
            abs(bias_pct) <= MAX_ABS_BIAS_PCT,
            f"{bias_pct:+.1f}% of actual spend, bound +-{MAX_ABS_BIAS_PCT:.0f}% "
            f"(naive forecast: {naive_bias_pct:+.1f}%)",
        ),
        Check(
            "error is not significantly worse than the naive forecast",
            error_guard.high >= 0,
            f"winsorised RMSE {rmse:,.0f} vs {naive_rmse:,.0f}, {_interval(error_guard)}",
        ),
    ]
    if champion_spearman is not None:
        checks.append(
            Check(
                "does not rank worse than the champion",
                ranking.candidate >= champion_spearman,
                f"Spearman {ranking.candidate:.3f} vs champion {champion_spearman:.3f} "
                "on the same data",
            )
        )
    return checks
