"""The two-part value model: expected spend = P(buy) x E[spend | buy]."""

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from src import hurdle


class _Churn:
    def __init__(self, churn_probability):
        self.p = np.asarray(churn_probability, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


class _Spend:
    def __init__(self, amounts):
        self.amounts = np.asarray(amounts, dtype=float)

    def predict(self, X):
        return self.amounts


X = pd.DataFrame({"x": [0, 1, 2]})


def test_expected_spend_is_the_chance_of_buying_times_the_spend_of_a_buyer():
    model = hurdle.HurdleValueModel(_Churn([0.2, 0.5, 1.0]), _Spend([100, 200, 300]))

    assert model.predict(X) == pytest.approx([80.0, 100.0, 0.0])


def test_the_chance_of_buying_is_one_minus_the_churn_probability():
    model = hurdle.HurdleValueModel(_Churn([0.1, 0.6, 0.9]), _Spend([1, 1, 1]))

    assert model.predict_buy_probability(X) == pytest.approx([0.9, 0.4, 0.1])


def test_a_negative_spend_prediction_counts_as_zero():
    model = hurdle.HurdleValueModel(_Churn([0.0, 0.0, 0.0]), _Spend([-50, 0, 40]))

    assert model.predict_spend_if_buying(X).tolist() == [0.0, 0.0, 40.0]
    assert model.predict(X).tolist() == [0.0, 0.0, 40.0]


def test_the_model_cannot_be_fitted_as_a_whole():
    model = hurdle.HurdleValueModel(_Churn([0.5] * 3), _Spend([1] * 3))

    with pytest.raises(NotImplementedError, match="fit the churn model and the spend model"):
        model.fit(X, [1, 2, 3])


def test_the_model_survives_pickling():
    model = hurdle.HurdleValueModel(_Churn([0.2, 0.5, 0.7]), _Spend([10, 20, 30]))

    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert restored.predict(X) == pytest.approx(model.predict(X))
