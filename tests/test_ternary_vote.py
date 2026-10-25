"""Design D3: ternary vote semantics — neutral abstains everywhere, silence holds state.

Spec: "neutral" is a deliberate no-edge judgment. It (1) leaves
the day's vote (weights renormalise over the speakers, sleeping-experts semantics), (2) never
enters learning evidence or error records (trust = P(correct | speaks)), and (3) a
no-speaker date maintains the previous position — silence is not a sell signal.
"""
from __future__ import annotations

from datetime import date

import pytest

from trading_agent.aggregation.soft_weighting import EXPERT_IDS, aggregate, aggregate_by_ticker
from trading_agent.contracts import ExpertView, is_abstention_view, is_safe_default_view
from trading_agent.memory.expert_error import build_expert_error_records

EQ = {eid: 1.0 / len(EXPERT_IDS) for eid in EXPERT_IDS}


def _view(eid, direction, confidence=0.8):
    return ExpertView(expert_id=eid, direction=direction, confidence=confidence,
                      reasoning="test", evidence=("e",), risk_flags=(), data_quality="medium")


def test_neutral_is_abstention_but_not_safe_default():
    v = _view("news", "neutral")
    assert is_abstention_view(v) is True
    assert is_safe_default_view(v) is False   # a judgment, not a fallback


def test_neutral_abstains_and_weights_renormalise_over_speakers():
    # (+1, 0, -1): score = (w_t - w_f) / (w_t + w_f); equal weights -> 0.
    views = {
        "technical": _view("technical", "long"),
        "news": _view("news", "neutral"),
        "fundamental": _view("fundamental", "flat"),
    }
    result = aggregate(views, EQ)
    assert result.score == pytest.approx(0.0)
    # Tilted weights decide the sign — the weight layer is decision-expressive again.
    tilted = aggregate(views, {"technical": 0.5, "news": 0.25, "fundamental": 0.25})
    assert tilted.score > 0.0


def test_all_neutral_maintains_previous_position_per_ticker():
    views = {eid: {"AAA": _view(eid, "neutral")} for eid in EXPERT_IDS}
    held = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=0.0,
                               exit_threshold=0.0, prev_directions={"AAA": "long"})
    assert held.by_ticker["AAA"].direction == "long"      # silence != sell
    idle = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=0.0,
                               exit_threshold=0.0, prev_directions={"AAA": "flat"})
    assert idle.by_ticker["AAA"].direction == "flat"      # silence never enters
    # Ablation replay switch: silence -> cash isolates the carry rule's effect.
    cashed = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=0.0,
                                 exit_threshold=0.0, prev_directions={"AAA": "long"},
                                 silence_policy="flat")
    assert cashed.by_ticker["AAA"].direction == "flat"


def test_exact_tie_maintains_position_both_ways():
    # D3 atom: one neutral + a 1-1 split renormalises to score EXACTLY 0.
    # Strict boundaries (frozen a priori 2026-07-17): zero net spoken edge is
    # evidence for nothing -- it neither enters nor exits, at ANY theta.
    views = {
        "technical": {"AAA": _view("technical", "long")},
        "news": {"AAA": _view("news", "neutral")},
        "fundamental": {"AAA": _view("fundamental", "flat")},
    }
    for theta in (0.0, 0.25):
        stay_long = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=theta,
                                        exit_threshold=0.0, prev_directions={"AAA": "long"})
        assert stay_long.by_ticker["AAA"].direction == "long", theta   # tie != exit
        stay_flat = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=theta,
                                        exit_threshold=0.0, prev_directions={"AAA": "flat"})
        assert stay_flat.by_ticker["AAA"].direction == "flat", theta   # tie != entry
        assert stay_flat.by_ticker["AAA"].score == pytest.approx(0.0)


def test_strict_exit_leaves_only_on_net_opposition():
    # Long position, speakers flip to net-negative -> exit (strictly below 0).
    views = {
        "technical": {"AAA": _view("technical", "flat")},
        "news": {"AAA": _view("news", "neutral")},
        "fundamental": {"AAA": _view("fundamental", "flat")},
    }
    out = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=0.0,
                              exit_threshold=0.0, prev_directions={"AAA": "long"})
    assert out.by_ticker["AAA"].direction == "flat"
    assert out.by_ticker["AAA"].score == pytest.approx(-1.0)


def test_single_speaker_acts_normally():
    views = {
        "technical": {"AAA": _view("technical", "long")},
        "news": {"AAA": _view("news", "neutral")},
        "fundamental": {"AAA": _view("fundamental", "neutral")},
    }
    out = aggregate_by_ticker(views, EQ, tickers=("AAA",), threshold=0.0,
                              prev_directions={})
    assert out.by_ticker["AAA"].direction == "long"
    assert out.by_ticker["AAA"].score == pytest.approx(1.0)


def test_forced_binary_expert_maps_neutral_onto_same_response_field():
    import dataclasses

    from trading_agent.experts.backends import ForcedBinaryExpert

    class _Inner:
        expert_id = "news"

        def views(self, observation, regime_label, context=None):
            return {
                "AAA": dataclasses.replace(_view("news", "neutral"), forced_direction="long"),
                "BBB": _view("news", "neutral"),          # neutral WITHOUT a forced field
                "CCC": _view("news", "flat"),             # spoken judgment
            }

    wrapped = ForcedBinaryExpert(_Inner())
    assert wrapped.expert_id == "news"
    views = wrapped.views(object(), "up_calm")
    assert views["AAA"].direction == "long"               # forced field applied
    assert not is_safe_default_view(views["AAA"])
    assert is_safe_default_view(views["BBB"])             # unforceable neutral degrades
    assert views["CCC"].direction == "flat"               # spoken passes through untouched
    assert views["CCC"].forced_direction is None


def test_neutral_excluded_from_error_records():
    views = {
        "technical": _view("technical", "long"),
        "news": _view("news", "neutral"),
        "fundamental": _view("fundamental", "flat"),
    }
    records = build_expert_error_records(
        market_state="up_calm", views=views, weights=EQ, realized_next_return=0.01,
    )
    assert {r.expert_id for r in records} == {"technical", "fundamental"}


def test_neutral_excluded_from_learning_attribution():
    from trading_agent.runner import _attribute

    class _Panel:
        def _price_on(self, tk, d):
            return 100.0

        def price_after(self, tk, d, n):
            return 102.0

    class _Obs:
        tickers = ("AAA",)

    views_by_expert = {
        "technical": {"AAA": _view("technical", "long")},
        "news": {"AAA": _view("news", "neutral")},
        "fundamental": {"AAA": _view("fundamental", "flat")},
    }
    episodes = _attribute(
        date(2024, 7, 1), "up_calm", views_by_expert, EQ, _Obs(), _Panel(),
        {"AAA": "long"}, market_return=0.0,
    )
    sa = episodes[0].signal_attribution
    assert set(sa) == {"technical", "fundamental"}   # neutral never trains the scorecard
    assert sa["technical"] == pytest.approx(0.02)
    assert sa["fundamental"] == pytest.approx(-0.02)
