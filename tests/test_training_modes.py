"""Which training settings the tests run with, and proof that the quick ones are really quick.

By default (see the quick_training fixture in conftest.py) tests train with one hyperparameter
combination and 100 bootstrap resamples. With FULL_TRAINING_TESTS=1 they use the real ones. These
tests check whichever mode is active, so neither can silently turn into the other.
"""

import pytest

from src import evaluation, model_search
from tests.conftest import FULL_TRAINING, QUICK_BOOTSTRAP

ALL = ["CHURN_CANDIDATES", "VALUE_CANDIDATES", "SPEND_CANDIDATES"]


def combinations(candidates):
    return [len(list(model_search.grid_combinations(c.grid))) for c in candidates]


@pytest.mark.skipif(FULL_TRAINING, reason="quick mode is off")
def test_quick_mode_uses_one_combination_per_algorithm_and_few_resamples():
    assert evaluation.N_BOOTSTRAP == QUICK_BOOTSTRAP
    for name in ALL:
        assert set(combinations(getattr(model_search, name))) == {1}, name


@pytest.mark.skipif(not FULL_TRAINING, reason="full mode is off")
def test_full_mode_uses_the_real_grids_and_resamples():
    assert evaluation.N_BOOTSTRAP == 1000
    assert max(combinations(model_search.CHURN_CANDIDATES)) == 16  # gradient boosting's real grid
    assert max(combinations(model_search.SPEND_CANDIDATES)) == 16
    assert max(combinations(model_search.VALUE_CANDIDATES)) == 16


@pytest.mark.warehouse
def test_the_real_data_regression_always_uses_the_real_settings():
    assert evaluation.N_BOOTSTRAP == 1000
    assert max(combinations(model_search.CHURN_CANDIDATES)) == 16


def test_quick_mode_keeps_every_algorithm_not_just_the_first():
    names = {c.name for c in model_search.CHURN_CANDIDATES}

    assert names == {"hist_gradient_boosting", "logistic_regression"}


def test_the_quick_grid_values_are_real_values_from_the_full_grid():
    quick = {c.name: c.grid for c in model_search.CHURN_CANDIDATES}
    if FULL_TRAINING:
        pytest.skip("full mode")

    assert quick["hist_gradient_boosting"] == {
        "learning_rate": [0.05],
        "max_leaf_nodes": [8],
        "max_iter": [100],
        "l2_regularization": [0.0],
    }
