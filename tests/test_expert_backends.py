"""Expert backend factory tests."""

from __future__ import annotations

import sys
import json
import types
from datetime import date, datetime

import pytest


def test_static_backend_factory_builds_three_experts():
    from trading_agent.experts.backends import build_experts

    experts = build_experts("static")

    assert [expert.expert_id for expert in experts] == [
        "technical",
        "news",
        "fundamental",
    ]


def test_unknown_backend_name_is_rejected():
    from trading_agent.experts.backends import build_experts

    with pytest.raises(ValueError, match="unknown expert backend"):
        build_experts("missing")


def test_heuristic_backend_factory_builds_three_experts():
    from trading_agent.experts.backends import build_experts

    experts = build_experts("heuristic")

    assert [expert.expert_id for expert in experts] == [
        "technical",
        "news",
        "fundamental",
    ]


def test_heuristic_backend_uses_structured_signals_for_per_ticker_views():
    from trading_agent.experts.backends import HeuristicExpertBackend

    backend = HeuristicExpertBackend()
    payload = {
        "tickers": ["AAA", "BBB"],
        "technical_signals": {
            "by_ticker": {
                "AAA": {"available": True, "momentum": 1.0, "reversal": 0.5, "lowvol": 0.2},
                "BBB": {"available": True, "momentum": -1.0, "reversal": -0.3, "lowvol": -0.2},
            }
        },
    }

    views = json.loads(backend.call("system", json.dumps(payload)))

    assert [view["ticker"] for view in views] == ["AAA", "BBB"]
    assert views[0]["direction"] == "long"
    assert views[1]["direction"] == "flat"
    assert views[0]["confidence"] > views[1]["confidence"]


def test_openai_backend_reports_missing_optional_dependency(monkeypatch):
    from trading_agent.experts.backends import OpenAIChatBackend

    monkeypatch.setitem(sys.modules, "openai", None)

    backend = OpenAIChatBackend(model="test-model")
    with pytest.raises(RuntimeError, match="openai package is required"):
        backend.call("system", "user")


def test_openai_backend_defaults_to_zero_temperature_and_records_revision():
    from trading_agent.experts.backends import OpenAIChatBackend

    backend = OpenAIChatBackend(model="test-model", model_revision="2026-07-03")

    assert backend.temperature == 0.0
    assert backend.model_name == "test-model"
    assert backend.model_revision == "2026-07-03"


def test_codex_sdk_backend_calls_openai_codex_thread(monkeypatch):
    from trading_agent.experts.backends import CodexSDKExpertBackend

    calls = {}

    class FakeResult:
        final_response = "codex raw json"

    class FakeThread:
        def run(self, payload, *, model=None, **kwargs):
            calls["payload"] = payload
            calls["model"] = model
            calls["kwargs"] = kwargs
            return FakeResult()

    class FakeCodex:
        def thread_start(self, **kwargs):
            calls["thread_start"] = kwargs
            return FakeThread()

        def close(self):
            calls["closed"] = True

    fake_module = types.SimpleNamespace(Codex=lambda: FakeCodex())
    monkeypatch.setitem(sys.modules, "openai_codex", fake_module)

    backend = CodexSDKExpertBackend(model="gpt-5.4-mini", model_revision="rev")
    out = backend.call("system prompt", "user prompt")

    assert out == "codex raw json"
    assert "system prompt" in calls["payload"]
    assert "user prompt" in calls["payload"]
    assert calls["model"] == "gpt-5.4-mini"
    assert calls["thread_start"]["model"] == "gpt-5.4-mini"
    assert backend.temperature == 0.0
    assert backend.model_name == "gpt-5.4-mini"
    assert backend.model_revision == "rev"


def test_codex_sdk_backend_reuses_client_across_calls(monkeypatch):
    from trading_agent.experts.backends import CodexSDKExpertBackend

    calls = {"codex_inits": 0, "runs": 0}

    class FakeResult:
        final_response = "codex raw json"

    class FakeThread:
        def run(self, payload, *, model=None, **kwargs):
            calls["runs"] += 1
            return FakeResult()

    class FakeCodex:
        def __init__(self):
            calls["codex_inits"] += 1

        def thread_start(self, **kwargs):
            return FakeThread()

        def close(self):
            calls["closed"] = calls.get("closed", 0) + 1

    monkeypatch.setitem(sys.modules, "openai_codex", types.SimpleNamespace(Codex=FakeCodex))

    backend = CodexSDKExpertBackend(model="gpt-5.4-mini")
    assert backend.call("system", "user 1") == "codex raw json"
    assert backend.call("system", "user 2") == "codex raw json"
    backend.close()

    assert calls["runs"] == 2
    assert calls["codex_inits"] == 1
    assert calls["closed"] == 1


def test_codex_sdk_backend_reports_missing_optional_dependency(monkeypatch):
    from trading_agent.experts.backends import CodexSDKExpertBackend

    monkeypatch.setitem(sys.modules, "openai_codex", None)

    backend = CodexSDKExpertBackend(model="gpt-5.4-mini")
    with pytest.raises(RuntimeError, match="openai_codex package is required"):
        backend.call("system", "user")


def test_codex_sdk_backend_factory_builds_three_experts():
    from trading_agent.experts.backends import build_experts

    experts = build_experts("codex_sdk", model="gpt-5.4-mini")

    assert [expert.expert_id for expert in experts] == [
        "technical",
        "news",
        "fundamental",
    ]


def test_codex_sdk_backend_mock_response_feeds_expert_parser(monkeypatch):
    from trading_agent.contracts import Bar, Observation
    from trading_agent.experts import TechnicalExpert
    from trading_agent.experts.backends import CodexSDKExpertBackend

    class FakeResult:
        final_response = json.dumps([
            {
                "ticker": "AAA",
                "direction": "long",
                "confidence": 0.7,
                "reasoning": "codex",
                "evidence": ["mock"],
                "risk_flags": [],
                "data_quality": "medium",
            },
            {
                "ticker": "BBB",
                "direction": "flat",
                "confidence": 0.2,
                "reasoning": "codex",
                "evidence": ["mock"],
                "risk_flags": [],
                "data_quality": "medium",
            },
        ])

    class FakeThread:
        def run(self, *args, **kwargs):
            return FakeResult()

    class FakeCodex:
        def thread_start(self, **kwargs):
            return FakeThread()

    monkeypatch.setitem(sys.modules, "openai_codex", types.SimpleNamespace(Codex=lambda: FakeCodex()))

    bar = Bar(open=100.0, high=101.0, low=99.0, close=100.0, adjusted_close=100.0, volume=1000)
    now = datetime(2024, 3, 15, 9, 30)
    observation = Observation(
        decision_date=date(2024, 3, 15),
        decision_time=now,
        availability_time=now,
        bars={"AAA": bar, "BBB": bar},
        tickers=["AAA", "BBB"],
    )
    views = TechnicalExpert(CodexSDKExpertBackend(model="gpt-5.4-mini")).views(
        observation,
        "up_stress",
        context={"technical": {"by_ticker": {}}},
    )

    assert set(views) == {"AAA", "BBB"}
    assert views["AAA"].direction == "long"
    assert views["AAA"].confidence == pytest.approx(0.7)


def test_expert_backend_metadata_records_codex_transport(monkeypatch):
    from trading_agent.experts.backends import expert_backend_metadata

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "openai_codex" else None)

    metadata = expert_backend_metadata(
        "codex_sdk",
        model="gpt-5.4-mini",
        model_revision="gpt-5.4-mini",
    )

    assert metadata == {
        "expert_transport": "openai_codex",
        "expert_temperature": 0.0,
        "expert_temperature_policy": "deterministic_requested",
        "codex_sdk_available": True,
    }
