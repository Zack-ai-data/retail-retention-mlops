"""Statistical baseline for churn: the BG/NBD model (Fader, Hardie and Lee, 2005).

BG/NBD is the classic model for customers who can leave at any time without telling you. Each
customer buys at their own steady rate while "alive" and, after any purchase, may drop out for
good. Both the rate and the dropout chance vary between customers (gamma and beta
distributions), so four numbers describe the whole base: r, alpha (purchase rate) and a, b
(dropout). A customer's history is summarised by:

  x    repeat purchases (orders after the first)
  t_x  days from the first order to the last
  T    days from the first order to the snapshot

and the model gives the probability that the customer is still alive. 1 - P(alive) is a churn
score that uses no machine learning features, so it shows how much the feature-based model adds.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import expit, gammaln

_LOG_BOUND = 12.0  # parameters are searched on a log scale within e^-12 .. e^12


def _history(
    frequency: npt.ArrayLike, recency_days: npt.ArrayLike, tenure_days: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x = np.asarray(frequency, dtype=float) - 1
    T = np.asarray(tenure_days, dtype=float)
    t_x = T - np.asarray(recency_days, dtype=float)
    if (x < 0).any() or (t_x < 0).any() or (t_x > T).any():
        raise ValueError("Each customer needs frequency >= 1 and 0 <= recency_days <= tenure_days")
    return x, t_x, T


def _log_likelihood(
    params: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    t_x: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> float:
    r, alpha, a, b = np.exp(params)
    repeat = x > 0
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    ln_a3 = -(r + x) * np.log(alpha + T)
    ln_a4 = (
        np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + t_x)
    )  # only used where x > 0
    inner = np.where(repeat, np.logaddexp(ln_a3, ln_a4), ln_a3)
    return float(np.sum(ln_a1 + ln_a2 + inner))


@dataclass(frozen=True)
class BgNbd:
    r: float
    alpha: float
    a: float
    b: float

    @classmethod
    def fit(
        cls,
        frequency: npt.ArrayLike,
        recency_days: npt.ArrayLike,
        tenure_days: npt.ArrayLike,
    ) -> "BgNbd":
        """Fit by maximum likelihood to customers' purchase histories as of one date."""
        x, t_x, T = _history(frequency, recency_days, tenure_days)
        if len(x) < 50:
            raise ValueError("Need at least 50 customers to fit BG/NBD")
        result = minimize(
            lambda p: -_log_likelihood(p, x, t_x, T),
            x0=np.log([1.0, float(np.median(T)) or 1.0, 1.0, 1.0]),
            method="L-BFGS-B",
            bounds=[(-_LOG_BOUND, _LOG_BOUND)] * 4,
        )
        if not result.success:
            raise ValueError(f"BG/NBD did not converge: {result.message}")
        r, alpha, a, b = (float(v) for v in np.exp(result.x))
        return cls(r, alpha, a, b)

    def probability_alive(
        self,
        frequency: npt.ArrayLike,
        recency_days: npt.ArrayLike,
        tenure_days: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """P(the customer has not dropped out yet), given their history at the snapshot."""
        x, t_x, T = _history(frequency, recency_days, tenure_days)
        repeat = x > 0
        log_odds_dead = (
            np.log(self.a)
            - np.log(np.maximum(self.b + x - 1, 1e-12))
            + (self.r + x) * (np.log(self.alpha + T) - np.log(self.alpha + t_x))
        )
        # a customer with no repeat purchase is alive as far as the model can tell
        return np.where(repeat, expit(-log_odds_dead), 1.0)
