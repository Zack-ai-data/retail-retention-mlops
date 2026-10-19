"""The paired bootstrap that tells the promotion gate whether a gain is real or noise."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import evaluation


def _scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    informative = y * 1.0 + rng.normal(0, 1.0, n)  # separates the classes
    noise = rng.normal(0, 1.0, n)  # does not
    return y, informative, noise


def test_a_clearly_better_score_has_an_interval_above_zero():
    y, informative, noise = _scores()

    gain = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score)

    assert gain.diff == pytest.approx(roc_auc_score(y, informative) - roc_auc_score(y, noise))
    assert gain.candidate == pytest.approx(roc_auc_score(y, informative))
    assert gain.reference == pytest.approx(roc_auc_score(y, noise))
    assert gain.low < gain.diff < gain.high
    assert gain.excludes_zero_above() is True


def test_two_equally_useless_scores_have_an_interval_around_zero():
    y, _, noise = _scores()
    other_noise = np.random.default_rng(99).normal(size=len(y))

    gain = evaluation.paired_bootstrap(y, noise, other_noise, roc_auc_score)

    assert gain.low < 0 < gain.high
    assert gain.excludes_zero_above() is False


def test_the_same_score_twice_gives_a_zero_difference_with_no_spread():
    y, informative, _ = _scores()

    gain = evaluation.paired_bootstrap(y, informative, informative, roc_auc_score)

    assert gain.diff == gain.low == gain.high == 0
    assert gain.excludes_zero_above() is False  # no gain is not a gain


def test_a_worse_candidate_has_a_negative_interval():
    y, informative, noise = _scores()

    gain = evaluation.paired_bootstrap(y, noise, informative, roc_auc_score)

    assert gain.high < 0
    assert gain.excludes_zero_above() is False


def test_the_result_is_reproducible_and_depends_on_the_seed():
    y, informative, noise = _scores()

    first = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, seed=1)
    again = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, seed=1)
    other = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, seed=2)

    assert first == again
    assert (first.low, first.high) != (other.low, other.high)


def test_a_higher_confidence_level_gives_a_wider_interval():
    y, informative, noise = _scores()

    narrow = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, level=0.8)
    wide = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, level=0.99)

    assert wide.high - wide.low > narrow.high - narrow.low


def test_resamples_where_the_metric_is_undefined_are_skipped():
    # 3 positives in 200 customers: some resamples contain no positive and roc_auc raises
    y = np.zeros(200, dtype=int)
    y[:3] = 1
    rng = np.random.default_rng(0)

    gain = evaluation.paired_bootstrap(y, rng.normal(size=200), rng.normal(size=200), roc_auc_score)

    assert 500 <= gain.resamples < evaluation.N_BOOTSTRAP


def test_an_evaluation_set_where_the_metric_is_mostly_undefined_is_an_error():
    y = np.zeros(50, dtype=int)  # one class only: roc_auc is never defined

    with pytest.raises(ValueError, match="too small or has too little variety"):
        evaluation.paired_bootstrap(y, np.arange(50.0), np.arange(50.0)[::-1], roc_auc_score)


def test_inputs_of_different_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        evaluation.paired_bootstrap([0, 1, 0], [0.1, 0.2], [0.3, 0.4, 0.5], roc_auc_score)


def test_to_metrics_names_the_difference_and_its_interval():
    y, informative, noise = _scores()
    gain = evaluation.paired_bootstrap(y, informative, noise, roc_auc_score, n_resamples=50)

    out = gain.to_metrics("test_x_")

    assert out == {
        "test_x_gain": gain.diff,
        "test_x_gain_ci_low": gain.low,
        "test_x_gain_ci_high": gain.high,
    }
