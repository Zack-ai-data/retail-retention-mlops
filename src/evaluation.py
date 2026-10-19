"""Uncertainty for model comparisons.

A single test score says little about whether one model really beats another: the same
comparison on a different sample of customers can flip. paired_bootstrap() resamples the
customers, scores both models on every resample and reports how large the difference is and
how much it moves, so the promotion gate can require that the gain is not noise.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src import config

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Gain:
    """A metric for a candidate and a reference (higher is better), with the paired difference."""

    candidate: float
    reference: float
    diff: float
    low: float  # lower end of the confidence interval of the difference
    high: float  # upper end
    resamples: int

    def excludes_zero_above(self) -> bool:
        """True when the candidate is better than the reference with confidence."""
        return self.low > 0

    def to_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}gain": self.diff,
            f"{prefix}gain_ci_low": self.low,
            f"{prefix}gain_ci_high": self.high,
        }


def paired_bootstrap(
    y_true: npt.ArrayLike,
    candidate: npt.ArrayLike,
    reference: npt.ArrayLike,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int | None = None,
    seed: int = config.RANDOM_STATE,
    level: float = CONFIDENCE_LEVEL,
) -> Gain:
    """Bootstrap the difference metric(y, candidate) - metric(y, reference) over customers.

    Both scores are evaluated on the same resampled customers, so the interval reflects how
    much the *difference* moves, not how much each score moves. n_resamples defaults to
    N_BOOTSTRAP. metric must be oriented so that higher is better (pass a negated error for
    RMSE). Resamples where the metric is undefined, for example a resample containing a single
    class, are skipped.
    """
    n_resamples = N_BOOTSTRAP if n_resamples is None else n_resamples
    y = np.asarray(y_true)
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if not len(y) == len(a) == len(b):
        raise ValueError("y_true, candidate and reference must have the same length")
    rng = np.random.default_rng(seed)

    differences = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(y), len(y))
        try:
            difference = metric(y[picked], a[picked]) - metric(y[picked], b[picked])
        except ValueError:
            continue
        if np.isfinite(difference):
            differences.append(difference)
    if len(differences) < n_resamples // 2:
        raise ValueError(
            f"The metric was undefined in {n_resamples - len(differences)} of {n_resamples} "
            "resamples; the evaluation set is too small or has too little variety"
        )

    tail = (1 - level) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    candidate_score, reference_score = metric(y, a), metric(y, b)
    return Gain(
        candidate=float(candidate_score),
        reference=float(reference_score),
        diff=float(candidate_score - reference_score),
        low=float(low),
        high=float(high),
        resamples=len(differences),
    )
