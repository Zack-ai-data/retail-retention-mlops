"""BG/NBD: fitted to simulated customers whose true parameters are known."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import baselines

TRUE = {"r": 0.8, "alpha": 40.0, "a": 0.7, "b": 2.5}


def simulate(n=3000, seed=0):
    """Customers who buy at their own rate until they drop out, as the BG/NBD model assumes.

    Returns frequency (orders), days since the last order, days since the first order, and
    whether each customer is still alive at the end of the observation period.
    """
    rng = np.random.default_rng(seed)
    rate = rng.gamma(TRUE["r"], 1 / TRUE["alpha"], n)  # purchases per day
    dropout = rng.beta(TRUE["a"], TRUE["b"], n)
    horizon = rng.uniform(150, 450, n)  # days from first order to the snapshot
    frequency, recency, alive = [], [], []
    for lam, p, T in zip(rate, dropout, horizon, strict=True):
        t, last, repeats, is_alive = 0.0, 0.0, 0, True
        while True:
            t += rng.exponential(1 / lam)
            if t > T:
                break
            repeats, last = repeats + 1, t
            if rng.random() < p:
                is_alive = False
                break
        frequency.append(repeats + 1)
        recency.append(T - last)
        alive.append(is_alive)
    return np.array(frequency), np.array(recency), horizon, np.array(alive)


@pytest.fixture(scope="module")
def customers():
    return simulate()


@pytest.fixture(scope="module")
def fitted(customers):
    frequency, recency, tenure, _ = customers
    return baselines.BgNbd.fit(frequency, recency, tenure)


def test_the_fit_recovers_the_purchase_rate_distribution(fitted):
    # the mean purchase rate r/alpha is what the data pins down most firmly
    assert fitted.r / fitted.alpha == pytest.approx(TRUE["r"] / TRUE["alpha"], rel=0.15)


def test_the_fit_recovers_the_dropout_distribution(fitted):
    assert fitted.a / (fitted.a + fitted.b) == pytest.approx(
        TRUE["a"] / (TRUE["a"] + TRUE["b"]), rel=0.25
    )


def test_probability_alive_is_a_probability_and_ranks_the_truly_dropped_out_lower(
    customers, fitted
):
    frequency, recency, tenure, alive = customers
    p_alive = fitted.probability_alive(frequency, recency, tenure)
    repeat = frequency > 1

    assert ((p_alive >= 0) & (p_alive <= 1)).all()
    # among customers with repeat orders, those who really dropped out score lower
    assert roc_auc_score(alive[repeat], p_alive[repeat]) > 0.7


def test_a_customer_with_no_repeat_order_is_assumed_alive(fitted):
    assert fitted.probability_alive([1], [100.0], [100.0]) == pytest.approx([1.0])


def test_a_longer_silence_lowers_the_probability_alive(fitted):
    recent = fitted.probability_alive([5], [10.0], [300.0])
    lapsed = fitted.probability_alive([5], [200.0], [300.0])

    assert lapsed < recent


def test_more_repeat_orders_before_a_silence_make_the_silence_more_telling(fitted):
    few = fitted.probability_alive([2], [150.0], [300.0])
    many = fitted.probability_alive([12], [150.0], [300.0])

    assert many < few


@pytest.mark.parametrize(
    "frequency, recency, tenure",
    [
        ([0] * 60, [1.0] * 60, [10.0] * 60),  # no orders at all
        ([2] * 60, [20.0] * 60, [10.0] * 60),  # silent for longer than the customer has existed
        ([2] * 60, [-1.0] * 60, [10.0] * 60),  # a last order in the future
    ],
)
def test_impossible_histories_are_rejected(frequency, recency, tenure):
    with pytest.raises(ValueError, match="frequency >= 1 and 0 <= recency_days <= tenure_days"):
        baselines.BgNbd.fit(frequency, recency, tenure)


def test_too_few_customers_are_rejected():
    with pytest.raises(ValueError, match="at least 50 customers"):
        baselines.BgNbd.fit([2] * 10, [5.0] * 10, [50.0] * 10)


def test_a_fit_that_does_not_converge_is_an_error(monkeypatch, customers):
    class _Failed:
        success = False
        message = "stuck"
        x = np.zeros(4)

    monkeypatch.setattr(baselines, "minimize", lambda *args, **kwargs: _Failed())
    frequency, recency, tenure, _ = customers

    with pytest.raises(ValueError, match="did not converge: stuck"):
        baselines.BgNbd.fit(frequency, recency, tenure)
