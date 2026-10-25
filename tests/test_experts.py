"""Tests for the expert committee (experts/ package).

Covers:
- Each concrete expert returns a valid ExpertView on canned responses
- Safe default (direction="flat", confidence=0.0, reasoning="parse_error") on any
  failure: malformed JSON, wrong direction value, confidence out of range, backend
  exception
- Report-style audit fields adapted from analyst-agent systems:
  evidence, risk_flags, and data_quality
- ExpertView is frozen (decision-side invariant)
- ExpertView has no STATE_AUTHORITY_FIELDS (decision-side invariant)
- Expert Protocol is structurally satisfied by each concrete class
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from typing import Any, Optional

import pytest

from trading_agent.contracts import (
    STATE_AUTHORITY_FIELDS,
    Bar,
    Expert,
    ExpertView,
    Observation,
)
from trading_agent.experts import (
    BaseExpert,
    ExpertBackend,
    FactorExpert,
    FundamentalExpert,
    NewsExpert,
    TechnicalExpert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeBackend:
    """Backend that returns a fixed string; raises on demand."""

    def __init__(self, response: str, raises: bool = False) -> None:
        self._response = response
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def call(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._raises:
            raise RuntimeError("backend failure")
        return self._response


def _canned(direction: str = "long", confidence: float = 0.75, reasoning: str = "ok") -> str:
    return json.dumps({
        "direction": direction,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": ["signal is constructive"],
        "risk_flags": ["sample is small"],
        "data_quality": "medium",
    })


def _canned_batch(*items: dict[str, Any]) -> str:
    return json.dumps(list(items))


def _item(
    ticker: str,
    direction: str = "long",
    confidence: float = 0.75,
    reasoning: str = "ok",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": ["signal is constructive"],
        "risk_flags": ["sample is small"],
        "data_quality": "medium",
    }


def _make_observation(tickers: tuple[str, ...] = ("AAA", "BBB")) -> Observation:
    bar = Bar(open=100.0, high=101.0, low=99.0, close=100.0, adjusted_close=100.0, volume=1_000_000)
    dt = datetime(2024, 3, 15, 9, 30)
    return Observation(
        decision_date=date(2024, 3, 15),
        decision_time=dt,
        availability_time=dt,
        bars={t: bar for t in tickers},
        tickers=list(tickers),
    )


_OBS = _make_observation()
_REGIME = "up_stress"

EXPERT_CLASSES = [TechnicalExpert, FactorExpert, NewsExpert, FundamentalExpert]


# ---------------------------------------------------------------------------
# ExpertView contract tests
# ---------------------------------------------------------------------------

def test_expert_view_is_frozen():
    ev = ExpertView(expert_id="technical", direction="long", confidence=0.8, reasoning="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.confidence = 0.0  # type: ignore[misc]


def test_expert_view_carries_report_style_audit_fields():
    ev = ExpertView(
        expert_id="technical",
        direction="long",
        confidence=0.8,
        reasoning="ok",
        evidence=("momentum positive",),
        risk_flags=("breadth weak",),
        data_quality="high",
    )
    assert ev.evidence == ("momentum positive",)
    assert ev.risk_flags == ("breadth weak",)
    assert ev.data_quality == "high"


def test_expert_view_has_no_state_authority_fields():
    field_names = {f.name for f in dataclasses.fields(ExpertView)}
    assert field_names.isdisjoint(STATE_AUTHORITY_FIELDS)


# ---------------------------------------------------------------------------
# Per-expert: happy path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_returns_long_on_canned_response(cls):
    backend = FakeBackend(_canned("long", 0.8))
    expert = cls(backend)
    ev = expert.view(_OBS, _REGIME)
    assert ev.direction == "long"
    assert ev.confidence == pytest.approx(0.8)
    assert ev.expert_id == expert.expert_id
    assert ev.evidence == ("signal is constructive",)
    assert ev.risk_flags == ("sample is small",)
    assert ev.data_quality == "medium"


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_returns_flat_on_canned_response(cls):
    backend = FakeBackend(_canned("flat", 0.3))
    expert = cls(backend)
    ev = expert.view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.expert_id == expert.expert_id


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_passes_context_to_backend(cls):
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    expert = cls(backend)
    ctx = {expert.expert_id: {"some": "signal"}}
    expert.views(_OBS, _REGIME, context=ctx)
    assert len(backend.calls) == 1
    _, user_prompt = backend.calls[0]
    payload = json.loads(user_prompt)
    assert payload["date"] == "2024-03-15"
    assert "regime" not in payload
    assert "market_state" not in json.dumps(payload)
    assert "prior_lessons" not in json.dumps(payload)


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_prompt_requests_report_style_fields(cls):
    backend = FakeBackend(_canned())
    cls(backend).view(_OBS, _REGIME)
    system_prompt, _ = backend.calls[0]
    assert "evidence" in system_prompt
    assert "risk_flags" in system_prompt
    assert "data_quality" in system_prompt
    assert "source-grounded" in system_prompt


def test_expert_user_prompt_excludes_retrieved_lessons_when_available():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    TechnicalExpert(backend).views(
        _OBS,
        _REGIME,
        context={
            "technical": {"momentum": 0.5},
            "prior_lessons": ["normal: trend evidence worked when volatility was contained"],
        },
    )
    _, user_prompt = backend.calls[0]
    payload = json.loads(user_prompt)
    assert "prior_lessons" not in payload


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_returns_one_view_per_ticker_from_batch_response(cls):
    backend = FakeBackend(_canned_batch(
        _item("AAA", "long", 0.8),
        _item("BBB", "flat", 0.2),
    ))
    views = cls(backend).views(_OBS, _REGIME, context={})
    assert set(views) == {"AAA", "BBB"}
    assert views["AAA"].direction == "long"
    assert views["AAA"].confidence == pytest.approx(0.8)
    assert views["BBB"].direction == "flat"
    assert views["BBB"].confidence == pytest.approx(0.2)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_batch_malformed_element_falls_back_for_only_that_ticker(cls):
    backend = FakeBackend(_canned_batch(
        _item("AAA", "long", 0.8),
        {"ticker": "BBB", "direction": "buy", "confidence": 2.0},
    ))
    views = cls(backend).views(_OBS, _REGIME, context={})
    assert views["AAA"].direction == "long"
    assert views["AAA"].confidence == pytest.approx(0.8)
    assert views["BBB"].direction == "flat"
    assert views["BBB"].confidence == 0.0
    assert views["BBB"].reasoning == "parse_error"


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_unparseable_batch_falls_back_for_all_tickers_and_records_raw_response(cls):
    backend = FakeBackend("not json")
    expert = cls(backend)
    views = expert.views(_OBS, _REGIME, context={})
    assert set(views) == {"AAA", "BBB"}
    assert all(view.direction == "flat" for view in views.values())
    assert all(view.confidence == 0.0 for view in views.values())
    assert expert.last_raw_response == "not json"


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_all_tickers_unavailable_skips_the_backend_call(cls):
    # When every ticker's input is marked unavailable, the outcome is the
    # data_unavailable safe default no matter what the model says, so the
    # (paid) backend call is skipped entirely.
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    expert = cls(backend)
    ctx = {cls.expert_id: {"by_ticker": {
        "AAA": {"available": False},
        "BBB": {"available": False},
    }}}
    views = expert.views(_OBS, _REGIME, context=ctx)
    assert backend.calls == []                       # no LLM round-trip
    assert set(views) == {"AAA", "BBB"}
    assert all(view.direction == "flat" for view in views.values())
    assert all(view.reasoning == "data_unavailable" for view in views.values())


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_unavailable_ticker_returns_safe_default_for_that_ticker_only(cls):
    backend = FakeBackend(_canned_batch(_item("AAA", "long", 0.8), _item("BBB", "long", 0.8)))
    ctx = {cls.expert_id: {"by_ticker": {
        "AAA": {"available": True, "value": 1.0},
        "BBB": {"available": False},
    }}}
    views = cls(backend).views(_OBS, _REGIME, context=ctx)
    assert views["AAA"].direction == "long"
    assert views["BBB"].direction == "flat"
    assert views["BBB"].confidence == 0.0
    assert views["BBB"].reasoning == "data_unavailable"
    assert views["BBB"].risk_flags == ("data_unavailable",)


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_handles_none_context(cls):
    backend = FakeBackend(_canned())
    ev = cls(backend).view(_OBS, _REGIME, context=None)
    assert ev.direction in ("long", "flat")


# ---------------------------------------------------------------------------
# Expert-specific context field names
# ---------------------------------------------------------------------------

def test_technical_expert_embeds_technical_key():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    TechnicalExpert(backend).views(_OBS, _REGIME, context={"technical": {"by_ticker": {}}})
    _, user = backend.calls[0]
    assert "technical_signals" in json.loads(user)


def test_factor_expert_embeds_factor_key():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    FactorExpert(backend).views(_OBS, _REGIME, context={"factor": {"by_ticker": {}}})
    _, user = backend.calls[0]
    assert "factor_signals" in json.loads(user)


def test_news_expert_embeds_news_key():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    NewsExpert(backend).views(_OBS, _REGIME, context={"news": {"by_ticker": {}}})
    _, user = backend.calls[0]
    assert "news_signals" in json.loads(user)


def test_news_expert_prompt_frames_text_plus_macro_lens():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    NewsExpert(backend).views(_OBS, _REGIME, context={"news": {"coverage": 0.5, "by_ticker": {}}})
    system_prompt, _ = backend.calls[0]
    # TradingAgents-style news analyst: reads recent news TEXT and grounds the macro
    # backdrop in FRED indicators (rates / curve / inflation / labour) in the SAME lens.
    assert "recent_news" in system_prompt
    assert "macro" in system_prompt
    assert "fed_funds_rate" in system_prompt
    assert "sentiment" in system_prompt
    # keeps ONE concise caution (sparse coverage / disagreement) and steers away from a
    # reflexive flat default.
    assert "sparse" in system_prompt
    assert "rather than" in system_prompt


def test_fundamental_expert_embeds_fundamental_key():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    FundamentalExpert(backend).views(_OBS, _REGIME, context={"fundamental": {"by_ticker": {}}})
    _, user = backend.calls[0]
    assert "fundamental_inputs" in json.loads(user)


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_prompt_hygiene_removes_regime_market_state_and_prior_lessons(cls):
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    cls(backend).views(
        _OBS,
        _REGIME,
        context={
            cls.expert_id: {"by_ticker": {"AAA": {"available": True}}},
            "prior_lessons": ["do not leak this"],
            "fundamental": {"market_state": "up_stress"},
        },
    )
    payload = json.loads(backend.calls[0][1])
    text = json.dumps(payload)
    assert "regime" not in payload
    assert "market_state" not in text
    assert "prior_lessons" not in text


def test_news_prompt_does_not_include_raw_news_text():
    backend = FakeBackend(_canned_batch(_item("AAA"), _item("BBB")))
    NewsExpert(backend).views(
        _OBS,
        _REGIME,
        context={"news": {"by_ticker": {"AAA": {
            "available": True,
            "sentiment": 0.7,
            "raw_text": "this must not enter the prompt",
        }}}},
    )
    assert "this must not enter the prompt" not in backend.calls[0][1]


# ---------------------------------------------------------------------------
# Safe-default tests (malformed responses)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_malformed_json_returns_safe_default(cls):
    backend = FakeBackend("this is not json")
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0
    assert ev.reasoning == "parse_error"


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_invalid_direction_returns_safe_default(cls):
    backend = FakeBackend(json.dumps({"direction": "buy", "confidence": 0.9, "reasoning": "x"}))
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_confidence_above_one_returns_safe_default(cls):
    backend = FakeBackend(json.dumps({"direction": "long", "confidence": 1.5, "reasoning": "x"}))
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_confidence_below_zero_returns_safe_default(cls):
    backend = FakeBackend(json.dumps({"direction": "long", "confidence": -0.1, "reasoning": "x"}))
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_missing_direction_field_returns_safe_default(cls):
    backend = FakeBackend(json.dumps({"confidence": 0.9, "reasoning": "x"}))
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_empty_response_returns_safe_default(cls):
    backend = FakeBackend("")
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_backend_exception_returns_safe_default(cls):
    backend = FakeBackend("", raises=True)
    ev = cls(backend).view(_OBS, _REGIME)
    assert ev.direction == "flat"
    assert ev.confidence == 0.0
    assert ev.reasoning == "parse_error"
    assert ev.evidence == ()
    assert ev.risk_flags == ("parse_error",)
    assert ev.data_quality == "low"


def test_missing_audit_fields_default_to_conservative_values():
    backend = FakeBackend(json.dumps({"direction": "long", "confidence": 0.6, "reasoning": "ok"}))
    ev = TechnicalExpert(backend).view(_OBS, _REGIME)
    assert ev.direction == "long"
    assert ev.evidence == ()
    assert ev.risk_flags == ()
    assert ev.data_quality == "unknown"


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_expert_satisfies_protocol(cls):
    backend = FakeBackend(_canned())
    expert = cls(backend)
    assert isinstance(expert, Expert)
    assert hasattr(expert, "expert_id")
    assert isinstance(expert.expert_id, str)


# ---------------------------------------------------------------------------
# Reasoning is capped at 200 chars
# ---------------------------------------------------------------------------

def test_reasoning_is_truncated_to_200_chars():
    long_reasoning = "x" * 300
    backend = FakeBackend(_canned("long", 0.6, long_reasoning))
    ev = TechnicalExpert(backend).view(_OBS, _REGIME)
    assert len(ev.reasoning) == 200


# ---------------------------------------------------------------------------
# ExpertBackend protocol satisfaction (FakeBackend structural check)
# ---------------------------------------------------------------------------

def test_fake_backend_satisfies_expert_backend_protocol():
    backend = FakeBackend(_canned())
    assert isinstance(backend, ExpertBackend)


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_raw_newline_inside_json_string_still_parses(cls):
    # LLMs sometimes emit a raw newline inside a JSON string; strict parsing
    # would downgrade a real judgment to a parse_error safe default.
    backend = FakeBackend(
        '[{"ticker": "AAA", "direction": "long", "confidence": 0.7,'
        ' "reasoning": "line one\nline two"},'
        ' {"ticker": "BBB", "direction": "flat", "confidence": 0.4,'
        ' "reasoning": "ok"}]'
    )
    views = cls(backend).views(_OBS, _REGIME, context={})
    assert views["AAA"].direction == "long"
    assert views["AAA"].confidence == pytest.approx(0.7)
    assert views["BBB"].direction == "flat"


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_forced_direction_is_parsed_and_lenient(cls):
    # D3 comparison instrumentation: the forced binary judgment rides in the
    # SAME response. Optional and LENIENT — a missing or invalid value must
    # never degrade an otherwise valid judgment into a safe default.
    backend = FakeBackend(
        '[{"ticker": "AAA", "direction": "neutral", "confidence": 0.5,'
        ' "reasoning": "mixed", "forced_direction": "long"},'
        ' {"ticker": "BBB", "direction": "flat", "confidence": 0.4,'
        ' "reasoning": "ok", "forced_direction": "sideways"}]'
    )
    views = cls(backend).views(_OBS, _REGIME, context={})
    assert views["AAA"].direction == "neutral"
    assert views["AAA"].forced_direction == "long"
    assert views["BBB"].direction == "flat"
    assert views["BBB"].forced_direction is None   # invalid value -> None, view intact


def test_prompt_contract_requests_the_forced_direction_field():
    from trading_agent.experts.base_expert import _BASE_SYSTEM_SUFFIX

    assert "forced_direction" in _BASE_SYSTEM_SUFFIX


@pytest.mark.parametrize("cls", EXPERT_CLASSES)
def test_participation_prior_is_opt_in_and_changes_system_prompt(cls):
    from trading_agent.experts.base_expert import PARTICIPATION_PRIOR

    backend = FakeBackend(_canned())
    expert = cls(backend)
    assert PARTICIPATION_PRIOR not in expert._system_prompt()   # off by default
    expert.participation_prior = PARTICIPATION_PRIOR
    sp = expert._system_prompt()
    assert PARTICIPATION_PRIOR in sp
    assert "equity risk premium" in sp
    # changing the system prompt changes the corpus cache key, forcing a fresh
    # recording (a different experiment), which is the intended behaviour.


def test_build_experts_participation_prior_flag():
    from trading_agent.experts.backends import build_experts
    from trading_agent.experts.base_expert import PARTICIPATION_PRIOR

    off = build_experts("static")
    on = build_experts("static", participation_prior=True)
    assert all(e.participation_prior == "" for e in off)
    assert all(e.participation_prior == PARTICIPATION_PRIOR for e in on)
    assert PARTICIPATION_PRIOR in on[0]._system_prompt()
