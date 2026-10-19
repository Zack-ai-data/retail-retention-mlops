"""The promotion rules. Each check is flipped on its own to show it can block a model."""

import pytest

from src import gate
from src.evaluation import Gain


def gain(candidate=0.70, reference=0.60, low=0.05, high=0.15):
    return Gain(candidate, reference, candidate - reference, low, high, 1000)


# ----------------------------------------------------------------- decide


def test_a_decision_needs_at_least_one_check():
    with pytest.raises(ValueError, match="at least one check"):
        gate.decide([])


def test_all_checks_passing_promotes_and_lists_the_evidence():
    decision = gate.decide([gate.Check("a", True, "detail a"), gate.Check("b", True, "detail b")])

    assert decision.promoted is True
    assert decision.reason == "passed: a (detail a); b (detail b)"


def test_one_failing_check_rejects_and_names_it_and_what_passed():
    decision = gate.decide([gate.Check("a", True, "ok"), gate.Check("b", False, "too low")])

    assert decision.promoted is False
    assert decision.reason == "failed: b (too low) | passed: a"


def test_when_everything_fails_there_is_no_passed_section():
    decision = gate.decide([gate.Check("a", False, "x")])

    assert decision.reason == "failed: a (x)"


def test_notes_are_appended_to_the_reason():
    decision = gate.decide([gate.Check("a", True, "ok")], notes=["no champion yet"])

    assert decision.reason.endswith("| no champion yet")


# ------------------------------------------------------------ churn checks


def churn(vs_recency=None, share=0.75, champion=None):
    return gate.churn_checks(vs_recency or gain(), share, champion)


def names(checks, passed):
    return [c.name for c in checks if c.passed is passed]


def test_a_real_gain_over_recency_on_most_origins_passes():
    assert all(c.passed for c in churn())


def test_a_gain_whose_interval_includes_zero_does_not_beat_the_baseline():
    checks = churn(vs_recency=gain(low=-0.01, high=0.05))

    assert names(checks, False) == ["beats the recency baseline"]


def test_a_gain_that_holds_on_half_the_origins_or_fewer_is_not_enough():
    assert names(churn(share=0.5), False) == ["gain holds across backtest origins"]
    assert names(churn(share=0.51), False) == []  # strictly more than half


def test_the_champion_check_exists_only_when_a_champion_was_re_scored():
    assert len(churn(champion=None)) == 2
    assert len(churn(champion=0.5)) == 3


def test_a_candidate_worse_than_the_re_scored_champion_is_blocked():
    checks = churn(vs_recency=gain(candidate=0.70), champion=0.72)

    assert names(checks, False) == ["not worse than the champion"]


def test_a_candidate_equal_to_the_champion_is_allowed():
    assert all(c.passed for c in churn(vs_recency=gain(candidate=0.70), champion=0.70))


# ------------------------------------------------------------ value checks


def value(**overrides):
    args = {
        "ranking": gain(0.59, 0.44, 0.08, 0.20),
        "capture": gain(0.56, 0.52, -0.02, 0.10),
        "bias_pct": -20.0,
        "naive_bias_pct": -35.0,
        "error_guard": gain(-2000.0, -2100.0, -150.0, 300.0),  # negated RMSE: -2000 is better
        "champion_spearman": None,
    }
    args.update(overrides)
    return gate.value_checks(**args)


def test_a_model_that_ranks_better_and_is_within_the_bias_bound_passes():
    assert all(c.passed for c in value())


def test_ranking_must_be_better_with_confidence():
    checks = value(ranking=gain(0.59, 0.55, -0.01, 0.08))

    assert names(checks, False) == ["ranks customers better than the naive forecast"]


def test_top_decile_capture_must_not_be_lower_than_the_naive_forecast():
    checks = value(capture=gain(0.50, 0.52, -0.08, 0.04))

    assert names(checks, False) == ["top-decile spend capture is not lower"]


def test_total_bias_beyond_the_bound_is_blocked_in_either_direction():
    assert gate.MAX_ABS_BIAS_PCT == 40.0
    for too_far in (-40.1, -78.0, +40.1, +65.0):
        checks = value(bias_pct=too_far)
        assert names(checks, False) == ["total bias is within the allowed bound"], too_far


def test_total_bias_exactly_at_the_bound_is_allowed():
    assert all(c.passed for c in value(bias_pct=-40.0))
    assert all(c.passed for c in value(bias_pct=+40.0))


def test_the_naive_forecasts_bias_does_not_decide_but_is_shown_for_context():
    # a model more biased than the naive forecast is fine as long as it is inside the bound
    checks = value(bias_pct=-32.0, naive_bias_pct=-10.0)
    bias = next(c for c in checks if c.name == "total bias is within the allowed bound")

    assert bias.passed is True
    assert "-32.0%" in bias.detail and "-10.0%" in bias.detail


def test_error_is_blocked_only_when_significantly_worse_than_naive():
    # negated RMSE: the model is worse (-2300 vs -2100) and the whole interval is below zero
    significantly_worse = value(error_guard=gain(-2300.0, -2100.0, -400.0, -50.0))
    # worse on average, but the interval includes zero: not shown to be worse
    unclear = value(error_guard=gain(-2300.0, -2100.0, -400.0, 50.0))

    assert names(significantly_worse, False) == [
        "error is not significantly worse than the naive forecast"
    ]
    assert all(c.passed for c in unclear)


def test_the_champion_check_compares_spearman_on_the_same_data():
    assert len(value(champion_spearman=None)) == 4
    worse = value(champion_spearman=0.62)  # candidate's Spearman is 0.59
    assert names(worse, False) == ["does not rank worse than the champion"]
    assert all(c.passed for c in value(champion_spearman=0.55))


def test_the_error_detail_reports_rmse_not_its_negative():
    checks = value(error_guard=gain(-2000.0, -2100.0, -150.0, 300.0))
    error = next(c for c in checks if c.name.startswith("error"))

    assert "2,000 vs 2,100" in error.detail
