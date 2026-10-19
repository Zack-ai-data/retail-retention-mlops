"""The two-part ("hurdle") model of a customer's spend over the next 90 days.

Spend is zero for the many customers who do not buy and large and skewed for those who do, so
it is modelled in two parts that a business can read separately:

    expected spend = P(the customer buys) x E[spend | the customer buys]

The first part is the churn model (P(buy) = 1 - P(churn)), so churn and value cannot disagree
about who is likely to buy. The second part is a regressor trained only on customers who did
buy, with a gamma loss that suits positive, right-skewed money.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class HurdleValueModel(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Expected spend = P(buy) x E[spend | buy], from a churn model and a spend model."""

    def __init__(self, churn_model: Any, spend_model: Any) -> None:
        self.churn_model = churn_model
        self.spend_model = spend_model

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike | None = None) -> "HurdleValueModel":
        # scikit-learn tooling (permutation importance) expects every estimator to have fit()
        raise NotImplementedError(
            "HurdleValueModel is assembled from two fitted models; fit the churn model and the "
            "spend model separately"
        )

    def predict_buy_probability(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(1.0 - self.churn_model.predict_proba(X)[:, 1])

    def predict_spend_if_buying(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.clip(np.asarray(self.spend_model.predict(X), dtype=float), 0.0, None)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return self.predict_buy_probability(X) * self.predict_spend_if_buying(X)
