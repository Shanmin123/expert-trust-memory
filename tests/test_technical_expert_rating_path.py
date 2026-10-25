"""Design D: TechnicalExpert votes deterministically from the context rating block."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from trading_agent.contracts import Bar, Observation
from trading_agent.experts.technical_expert import TechnicalExpert


class _ExplodingBackend:
    """Fails if called — proves the deterministic path skips the LLM."""

    def call(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("backend must not be called on the deterministic rating path")


class _StaticLongBackend:
    def call(self, system_prompt: str, user_prompt: str) -> str:
        return (
            '[{"ticker": "TSLA", "direction": "long", "confidence": 0.5,'
            ' "reasoning": "llm fallback", "evidence": ["e"], "risk_flags": [],'
            ' "data_quality": "medium"}]'
        )


def _obs(tickers=("TSLA",)):
    bar = Bar(open=100.0, high=101.0, low=99.0, close=100.0, adjusted_close=100.0, volume=1_000_000)
    dt = datetime(2024, 7, 1, 9, 30)
    return Observation(
        decision_date=date(2024, 7, 1),
        decision_time=dt,
        availability_time=dt,
        bars={t: bar for t in tickers},
        tickers=list(tickers),
    )


def _ctx(summary, ma=0.2, osc=-0.1, ticker="TSLA"):
    return {"technical": {"by_ticker": {ticker: {
        "available": True,
        "rating": {"summary": summary, "ma": ma, "oscillator": osc},
    }}}}


def test_positive_summary_votes_long_without_backend_call():
    expert = TechnicalExpert(_ExplodingBackend())
    views = expert.views(_obs(), "up_calm", _ctx(0.42))
    v = views["TSLA"]
    assert v.direction == "long"
    assert v.confidence == pytest.approx(0.42)
    assert v.data_quality == "high"
    assert "summary=0.4200" in v.evidence[0]


def test_tradingview_bands_map_to_ternary_directions():
    # Official TradingView verdict bands: Buy > +0.1, Neutral in [-0.1, +0.1], Sell < -0.1.
    expert = TechnicalExpert(_ExplodingBackend())
    assert expert.views(_obs(), "up_calm", _ctx(0.42))["TSLA"].direction == "long"
    assert expert.views(_obs(), "up_calm", _ctx(0.05))["TSLA"].direction == "neutral"
    assert expert.views(_obs(), "up_calm", _ctx(0.0))["TSLA"].direction == "neutral"
    assert expert.views(_obs(), "up_calm", _ctx(-0.05))["TSLA"].direction == "neutral"
    assert expert.views(_obs(), "up_calm", _ctx(-0.3))["TSLA"].direction == "flat"
    assert "rating_neutral_band" in expert.views(_obs(), "up_calm", _ctx(0.0))["TSLA"].risk_flags


def test_confidence_is_clipped_to_expert_view_bounds():
    expert = TechnicalExpert(_ExplodingBackend())
    assert expert.views(_obs(), "up_calm", _ctx(0.98))["TSLA"].confidence == pytest.approx(0.9)
    assert expert.views(_obs(), "up_calm", _ctx(0.01))["TSLA"].confidence == pytest.approx(0.05)


def test_deterministic_rating_carries_forced_binary_direction():
    # forced_direction = the design-D sign rule (long when summary > 0): the
    # forced-binary replay reproduces the pre-D3 technical vote exactly.
    expert = TechnicalExpert(_ExplodingBackend())
    assert expert.views(_obs(), "up_calm", _ctx(0.42))["TSLA"].forced_direction == "long"
    assert expert.views(_obs(), "up_calm", _ctx(0.05))["TSLA"].forced_direction == "long"
    assert expert.views(_obs(), "up_calm", _ctx(0.0))["TSLA"].forced_direction == "flat"
    assert expert.views(_obs(), "up_calm", _ctx(-0.3))["TSLA"].forced_direction == "flat"


def test_missing_rating_falls_back_to_llm_prompt_path():
    expert = TechnicalExpert(_StaticLongBackend())
    views = expert.views(_obs(), "up_calm", {"technical": {"by_ticker": {"TSLA": {"available": True}}}})
    assert views["TSLA"].direction == "long"
    assert views["TSLA"].reasoning == "llm fallback"
