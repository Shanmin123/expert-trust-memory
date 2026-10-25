"""Tests for aggregation.soft_weighting (design-D three-expert committee).

Covers:
- Equal weights (explicit and fallback from empty dict)
- Weights below the floor are raised and renormalised
- Weights needing renormalisation (don't sum to 1)
- Missing experts in views contribute 0 to score
- Missing experts in weights get the floor
- Score is always in [-1, 1]
- Normalised weights always sum to 1
- Custom threshold changes decision
- AggregationResult is frozen
- HOLD semantics (exit_threshold hysteresis)
"""

from __future__ import annotations

import dataclasses

import pytest

from trading_agent.aggregation.soft_weighting import (
    EXPERT_IDS,
    WEIGHT_FLOOR,
    AggregationResult,
    aggregate,
    aggregate_by_ticker,
    normalise_weights,
)
from trading_agent.contracts import ExpertView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _view(expert_id: str, direction: str = "long", confidence: float = 1.0) -> ExpertView:
    return ExpertView(
        expert_id=expert_id,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="test",
    )


def _all_views(direction: str = "long", confidence: float = 1.0) -> dict[str, ExpertView]:
    return {eid: _view(eid, direction, confidence) for eid in EXPERT_IDS}


EQ = 1.0 / len(EXPERT_IDS)          # 1/3 for the three-expert committee
_EQUAL = {eid: EQ for eid in EXPERT_IDS}
# Max weight after flooring the other two experts: 1 - 2*floor = 0.90.
_DOMINANT_MAX = 1.0 - (len(EXPERT_IDS) - 1) * WEIGHT_FLOOR


# ---------------------------------------------------------------------------
# normalise_weights
# ---------------------------------------------------------------------------

def test_empty_weights_produce_equal_weights():
    norm = normalise_weights({})
    assert set(norm.keys()) == set(EXPERT_IDS)
    for v in norm.values():
        assert v == pytest.approx(EQ)


def test_equal_weights_unchanged_after_normalisation():
    norm = normalise_weights(_EQUAL)
    for v in norm.values():
        assert v == pytest.approx(EQ)


def test_weights_sum_to_one():
    norm = normalise_weights({"technical": 0.5, "news": 0.3})
    assert sum(norm.values()) == pytest.approx(1.0)


def test_missing_expert_gets_floor():
    # Only "technical" provided; others are projected to >= WEIGHT_FLOOR and the
    # vector sums to 1 (genuine floored-simplex projection, not a single pass).
    norm = normalise_weights({"technical": 0.9})
    assert set(norm.keys()) == set(EXPERT_IDS)
    assert all(v >= WEIGHT_FLOOR - 1e-9 for v in norm.values())
    assert sum(norm.values()) == pytest.approx(1.0)


def test_weight_below_floor_is_raised():
    tiny = WEIGHT_FLOOR / 2
    norm = normalise_weights({eid: tiny for eid in EXPERT_IDS})
    # All weights equal after flooring and renorm -> each == 1/3.
    for v in norm.values():
        assert v == pytest.approx(EQ)


def test_zero_weights_produce_equal_weights():
    norm = normalise_weights({eid: 0.0 for eid in EXPERT_IDS})
    for v in norm.values():
        assert v == pytest.approx(EQ)


def test_large_weights_are_renormalised():
    norm = normalise_weights({eid: 100.0 for eid in EXPERT_IDS})
    for v in norm.values():
        assert v == pytest.approx(EQ)


def test_single_dominant_weight_still_floors_others():
    # "technical" gets 0.9; others projected to exactly WEIGHT_FLOOR (0.05) while
    # technical keeps the remaining budget (0.90). Sum stays 1, floor holds.
    norm = normalise_weights({"technical": 0.9})
    assert all(v >= WEIGHT_FLOOR - 1e-9 for v in norm.values())
    assert norm["technical"] > norm["news"]     # dominant expert remains largest
    assert norm["technical"] == pytest.approx(_DOMINANT_MAX)
    for eid in ("news", "fundamental"):
        assert norm[eid] == pytest.approx(WEIGHT_FLOOR)
    assert sum(norm.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate: score and direction
# ---------------------------------------------------------------------------

def test_all_long_full_confidence_equal_weights_scores_one():
    result = aggregate(_all_views("long", 1.0), _EQUAL)
    assert result.score == pytest.approx(1.0)
    assert result.direction == "long"


def test_all_flat_equal_weights_scores_minus_one():
    result = aggregate(_all_views("flat"), _EQUAL)
    assert result.score == pytest.approx(-1.0)   # three -1 votes
    assert result.direction == "flat"


def test_empty_weights_fallback_same_as_equal_weights():
    result_empty = aggregate(_all_views("long", 1.0), {})
    result_equal = aggregate(_all_views("long", 1.0), _EQUAL)
    assert result_empty.score == pytest.approx(result_equal.score)
    assert result_empty.direction == result_equal.direction


def test_two_long_one_flat_scores_a_third_and_stays_below_entry():
    # 2 long votes (+1) and 1 flat vote (-1) at equal weight -> score 1/3;
    # below an EXPLICIT 0.5 entry it stays flat (the frozen default theta=0
    # enters on any strictly positive score -- pinned in the hardening tests).
    views = {
        "technical":   _view("technical",   "long", 1.0),
        "news":        _view("news",        "long", 1.0),
        "fundamental": _view("fundamental", "flat", 1.0),
    }
    result = aggregate(views, _EQUAL, threshold=0.5)
    assert result.score == pytest.approx(1.0 / 3.0)
    assert result.direction == "flat"


def test_one_long_two_flat_scores_minus_a_third():
    views = {
        "technical":   _view("technical",   "long", 1.0),
        "news":        _view("news",        "flat", 1.0),
        "fundamental": _view("fundamental", "flat", 1.0),
    }
    result = aggregate(views, _EQUAL)
    assert result.score == pytest.approx(-1.0 / 3.0)   # (1 - 1 - 1) / 3
    assert result.direction == "flat"


def test_confidence_weighting_lets_a_confident_minority_win():
    # 2 low-confidence longs + 1 high-confidence flat. The plain +/-1 vote is long (2-1 at
    # theta=0); confidence weighting lets the confident flat outweigh the two hesitant longs.
    vbe = {
        "technical":   {"X": _view("technical",   "long", 0.3)},
        "news":        {"X": _view("news",        "long", 0.3)},
        "fundamental": {"X": _view("fundamental", "flat", 0.9)},
    }
    plain = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0)
    assert plain.by_ticker["X"].direction == "long"
    conf = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0, confidence_weighted=True)
    assert conf.by_ticker["X"].direction == "flat"
    assert conf.by_ticker["X"].score == pytest.approx((0.3 + 0.3 - 0.9) / 3)


def _neutral_forced(expert_id: str, forced: str) -> ExpertView:
    return ExpertView(
        expert_id=expert_id, direction="neutral", confidence=0.5,
        reasoning="no edge", forced_direction=forced,  # type: ignore[arg-type]
    )


def test_soft_abstention_tilts_a_split_committee():
    # 1 long + 1 flat is a tie (score 0, boundary maintains). A neutral whose forced call
    # is long joins at half a vote and breaks the tie toward long.
    vbe = {
        "technical":   {"X": _view("technical", "long", 1.0)},
        "news":        {"X": _view("news", "flat", 1.0)},
        "fundamental": {"X": _neutral_forced("fundamental", "long")},
    }
    plain = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0)
    assert plain.by_ticker["X"].score == pytest.approx(0.0)          # tie without it
    soft = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0, soft_abstain_weight=0.5)
    # weights: full 1/3+1/3, soft 0.5*(1/3); budget=2.5/3 -> score=(1/3-1/3+0.5/3)/(2.5/3)=0.2
    assert soft.by_ticker["X"].score == pytest.approx(0.2)
    assert soft.by_ticker["X"].direction == "long"


def test_soft_abstention_never_conjures_a_vote_from_silence():
    # All three neutral (each with a forced call): still a no-speaker date -> silence
    # maintains, exactly as sealed. Forced calls may tilt a spoken committee, not create one.
    vbe = {eid: {"X": _neutral_forced(eid, "long")} for eid in ("technical", "news", "fundamental")}
    soft = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0,
                               soft_abstain_weight=0.5, prev_directions={"X": "long"})
    assert soft.by_ticker["X"].direction == "long"     # carried, not voted
    assert soft.by_ticker["X"].score == pytest.approx(0.0)


def test_soft_abstention_ignores_safe_defaults():
    # A safe default (flat, confidence 0, data_unavailable) is not a judgment: it never
    # enters, forced or not.
    safe = ExpertView(expert_id="fundamental", direction="flat", confidence=0.0,
                      reasoning="data_unavailable", risk_flags=("data_unavailable",),
                      data_quality="low", forced_direction="long")
    vbe = {
        "technical":   {"X": _view("technical", "long", 1.0)},
        "news":        {"X": _view("news", "flat", 1.0)},
        "fundamental": {"X": safe},
    }
    soft = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0, soft_abstain_weight=0.5)
    assert soft.by_ticker["X"].score == pytest.approx(0.0)   # tie unchanged: safe default excluded


def test_confidence_weighting_leaves_learned_memory_untouched():
    # Confidence weighting scales the committee vote only; attribution reads each expert's
    # own +/-1 direction, so a unanimous full-confidence vote is unchanged either way.
    vbe = {eid: {"X": _view(eid, "long", 1.0)} for eid in ("technical", "news", "fundamental")}
    plain = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0)
    conf = aggregate_by_ticker(vbe, _EQUAL, tickers=["X"], threshold=0.0, confidence_weighted=True)
    assert plain.by_ticker["X"].score == conf.by_ticker["X"].score == pytest.approx(1.0)


def test_score_in_minus_one_to_one_range():
    for conf in (0.0, 0.3, 0.7, 1.0):
        for direction in ("long", "flat"):
            result = aggregate(_all_views(direction, conf), _EQUAL)
            assert -1.0 - 1e-9 <= result.score <= 1.0 + 1e-9


def test_custom_threshold_changes_decision():
    # 2 long votes, 1 flat -> score = (1 + 1 - 1) / 3 = 1/3.
    views = {
        "technical":   _view("technical",   "long", 1.0),
        "news":        _view("news",        "long"),
        "fundamental": _view("fundamental", "flat"),
    }
    third = 1.0 / 3.0
    # STRICT entry boundary (frozen 2026-07-17): a score exactly AT the
    # threshold is not net evidence and does not enter.
    assert aggregate(views, _EQUAL, threshold=third).direction == "flat"   # 1/3 == 1/3
    assert aggregate(views, _EQUAL, threshold=0.3).direction == "long"     # 1/3 > 0.3
    assert aggregate(views, _EQUAL, threshold=0.4).direction == "flat"     # 1/3 < 0.4


def test_weights_below_floor_are_raised_in_aggregate():
    tiny = {eid: 0.001 for eid in EXPERT_IDS}
    result = aggregate(_all_views("long", 1.0), tiny)
    # After flooring all to 0.05 and renorm: equal weights -> score ~ 1.0
    assert result.score == pytest.approx(1.0)


def test_weights_needing_renormalisation():
    # Weights sum to 1.5 -> renormalised; after floor same effect.
    unnorm = {eid: 0.5 for eid in EXPERT_IDS}
    result = aggregate(_all_views("long", 1.0), unnorm)
    assert result.score == pytest.approx(1.0)
    assert sum(result.weights_used.values()) == pytest.approx(1.0)


def test_missing_view_drops_out_and_weights_renormalise():
    # Two experts have views, "fundamental" is absent: the remaining weights
    # renormalise, so a missing expert neither supports nor opposes the trade.
    views = {
        "technical": _view("technical", "long", 1.0),
        "news":      _view("news",      "long", 1.0),
        # "fundamental" missing
    }
    result_missing = aggregate(views, _EQUAL)
    # Score = (1/3 + 1/3) / (2/3) = 1.0 over the available experts.
    assert result_missing.score == pytest.approx(1.0)
    assert result_missing.direction == "long"


def test_renormalisation_matches_proportional_vote_counting():
    # Equal weights, threshold 0.5: the full committee's 2-vs-1 (score 1/3) is
    # NOT enough; with only two available experts both long, the renormalised
    # score is 1.0 and clears the same threshold. Proportional counting.
    two_vs_one = {
        "technical":   _view("technical",   "long", 1.0),
        "news":        _view("news",        "long", 1.0),
        "fundamental": _view("fundamental", "flat", 1.0),
    }
    result = aggregate(two_vs_one, _EQUAL, threshold=0.5)
    assert result.score == pytest.approx(1.0 / 3.0)
    assert result.direction == "flat"
    two_available_both_long = {
        eid: _view(eid, "long", 1.0) for eid in ("technical", "news")
    }
    assert aggregate(two_available_both_long, _EQUAL, threshold=0.5).direction == "long"


def test_safe_default_view_drops_out_like_a_missing_view():
    # A data-unavailable fallback is not a judgment: it must not vote flat.
    from trading_agent.experts.base_expert import _safe_default

    with_default = {
        "technical":   _view("technical", "long", 1.0),
        "news":        _safe_default("news", "data_unavailable"),
        "fundamental": _view("fundamental", "flat", 1.0),
    }
    result = aggregate(with_default, _EQUAL)
    # Available: technical long, fundamental flat -> (1/3 - 1/3) / (2/3) = 0.
    assert result.score == pytest.approx(0.0)
    without_news = {k: v for k, v in with_default.items() if k != "news"}
    assert aggregate(without_news, _EQUAL).score == pytest.approx(result.score)


def test_all_views_missing_scores_zero_and_flat_even_at_zero_threshold():
    result = aggregate({}, _EQUAL, threshold=0.0)
    assert result.score == pytest.approx(0.0)
    assert result.direction == "flat"   # no evidence -> no position, regardless of threshold


# ---------------------------------------------------------------------------
# HOLD semantics (exit_threshold hysteresis)
# ---------------------------------------------------------------------------

def _views_by_ticker(directions: dict[str, str]) -> dict[str, dict[str, ExpertView]]:
    return {
        eid: {"AAA": _view(eid, direction, 1.0)}
        for eid, direction in directions.items()
    }


_ALL_LONG = {"technical": "long", "news": "long", "fundamental": "long"}         # score +1
_TWO_VS_ONE = {"technical": "long", "news": "long", "fundamental": "flat"}       # score +1/3
_ONE_VS_TWO = {"technical": "long", "news": "flat", "fundamental": "flat"}       # score -1/3


def test_hold_semantics_keeps_a_long_between_entry_and_exit_bands():
    # Enter on a unanimous +1, then a 2-vs-1 day (score +1/3) sits between the
    # 0.5 entry and the 0 exit: HOLD keeps the long. A 1-vs-2 day (-1/3) exits.
    entry = aggregate_by_ticker(
        _views_by_ticker(_ALL_LONG), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=0.0, prev_directions={},
    )
    assert entry.by_ticker["AAA"].direction == "long"

    held = aggregate_by_ticker(
        _views_by_ticker(_TWO_VS_ONE), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=0.0, prev_directions={"AAA": "long"},
    )
    assert held.by_ticker["AAA"].direction == "long"      # 1/3 > 0 -> HOLD
    assert held.by_ticker["AAA"].score == pytest.approx(1.0 / 3.0)

    exited = aggregate_by_ticker(
        _views_by_ticker(_ONE_VS_TWO), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=0.0, prev_directions={"AAA": "long"},
    )
    assert exited.by_ticker["AAA"].direction == "flat"    # -1/3 <= 0 -> exit


def test_hold_semantics_needs_a_prior_long():
    # Below entry with no position: hysteresis never CREATES a long.
    result = aggregate_by_ticker(
        _views_by_ticker(_TWO_VS_ONE), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=0.0, prev_directions={"AAA": "flat"},
    )
    assert result.by_ticker["AAA"].direction == "flat"


def test_daily_revote_ablation_requires_explicit_exit_none():
    # Strict HOLD (exit_threshold=0.0) is the frozen DEFAULT; the daily
    # re-vote ablation is opted into with an explicit exit_threshold=None.
    result = aggregate_by_ticker(
        _views_by_ticker(_TWO_VS_ONE), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=None, prev_directions={"AAA": "long"},
    )
    assert result.by_ticker["AAA"].direction == "flat"    # re-voted below entry


def test_hold_semantics_deep_flip_exits():
    result = aggregate_by_ticker(
        _views_by_ticker(_ONE_VS_TWO), _EQUAL, tickers=("AAA",),
        threshold=0.5, exit_threshold=0.0, prev_directions={"AAA": "long"},
    )
    assert result.by_ticker["AAA"].score == pytest.approx(-1.0 / 3.0)
    assert result.by_ticker["AAA"].direction == "flat"


def test_weights_used_in_result_sum_to_one():
    result = aggregate(_all_views(), {"technical": 0.6, "news": 0.1})
    assert sum(result.weights_used.values()) == pytest.approx(1.0)


def test_weights_used_all_above_floor():
    result = aggregate(_all_views(), {"technical": 0.001})
    for v in result.weights_used.values():
        assert v >= WEIGHT_FLOOR - 1e-9


# ---------------------------------------------------------------------------
# AggregationResult contract
# ---------------------------------------------------------------------------

def test_aggregation_result_is_frozen():
    result = aggregate(_all_views(), _EQUAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0  # type: ignore[misc]


def test_aggregation_result_direction_is_valid():
    for direction in ("long", "flat"):
        result = aggregate(_all_views(direction, 0.8), _EQUAL)
        assert result.direction in ("long", "flat")


# ---------------------------------------------------------------------------
# Confidence = 0 for "long" does not contribute
# ---------------------------------------------------------------------------

def test_confidence_is_ignored_only_direction_votes():
    # Aggregation reads direction as a +/-1 vote; the stated confidence is not
    # used, so three long views at confidence 0.0 still score +1 (three +1 votes).
    views = {eid: _view(eid, "long", 0.0) for eid in EXPERT_IDS}
    result = aggregate(views, _EQUAL)
    assert result.score == pytest.approx(1.0)
    assert result.direction == "long"


# ---------------------------------------------------------------------------
# Partial weight dict: experts absent from weights get the floor
# ---------------------------------------------------------------------------

def test_partial_weight_dict_floors_missing_experts():
    # Only "technical" is specified; others should be floored.
    norm = normalise_weights({"technical": 0.7})
    assert all(norm[eid] >= WEIGHT_FLOOR - 1e-9 for eid in EXPERT_IDS)
    assert sum(norm.values()) == pytest.approx(1.0)


def test_per_ticker_one_long_expert_equal_weight_stays_flat():
    views = {
        "technical": {"AAA": _view("technical", "long", 1.0)},
        "news": {"AAA": _view("news", "flat", 1.0)},
        "fundamental": {"AAA": _view("fundamental", "flat", 1.0)},
    }
    result = aggregate_by_ticker(views, _EQUAL, tickers=["AAA"])
    assert result.by_ticker["AAA"].score == pytest.approx(-1.0 / 3.0)   # (1 - 1 - 1)/3
    assert result.by_ticker["AAA"].direction == "flat"


def test_per_ticker_dominant_weight_crosses_threshold():
    views = {
        "technical": {"AAA": _view("technical", "long", 0.6)},
        "news": {"AAA": _view("news", "flat", 1.0)},
        "fundamental": {"AAA": _view("fundamental", "flat", 1.0)},
    }
    result = aggregate_by_ticker(views, {"technical": 0.9}, tickers=["AAA"])
    assert result.weights_used["technical"] == pytest.approx(_DOMINANT_MAX)
    # 0.90*(+1) + 0.05*(-1)*2 = 0.90 - 0.10 = 0.80 (confidence is not used).
    assert result.by_ticker["AAA"].score == pytest.approx(0.80)
    assert result.by_ticker["AAA"].direction == "long"


def test_per_ticker_decisions_are_independent():
    views = {
        "technical": {
            "AAA": _view("technical", "long", 1.0),
            "BBB": _view("technical", "flat", 1.0),
        },
        "news": {
            "AAA": _view("news", "long", 1.0),   # AAA all long -> score 1.0 -> long
            "BBB": _view("news", "flat", 1.0),
        },
        "fundamental": {
            "AAA": _view("fundamental", "long", 1.0),
            "BBB": _view("fundamental", "flat", 1.0),
        },
    }
    before = aggregate_by_ticker(views, _EQUAL, tickers=["AAA", "BBB"], threshold=0.5)
    views["technical"]["AAA"] = _view("technical", "flat", 1.0)  # now 2L1F -> 1/3 -> flat
    after = aggregate_by_ticker(views, _EQUAL, tickers=["AAA", "BBB"], threshold=0.5)
    assert before.by_ticker["BBB"] == after.by_ticker["BBB"]
    assert before.by_ticker["AAA"].direction == "long"
    assert after.by_ticker["AAA"].direction == "flat"
