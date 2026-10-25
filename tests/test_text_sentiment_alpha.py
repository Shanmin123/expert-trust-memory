"""Tests for LLM-derived text sentiment alpha signals."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from trading_agent._bootstrap import FinsaberDataset
from trading_agent.alpha.llm_news_sentiment import (
    DEFAULT_LLM_NEWS_MODEL,
    LLMNewsSentimentAlpha,
    NewsSentimentCache,
    SIGNAL_ID,
)
from trading_agent.contracts import Observation
from trading_agent.data.finsaber_text import LoaderTextPanel


def _bar(c=100.0):
    return {"open": c, "high": c, "low": c, "close": c, "adjusted_close": c, "volume": 1_000_000}


def _obs(t, tickers):
    as_of = datetime.combine(t, time.min)
    return Observation(decision_date=t, decision_time=as_of, availability_time=as_of,
                       bars={}, tickers=tuple(tickers))


class StubNewsSentimentBackend:
    model_name = DEFAULT_LLM_NEWS_MODEL
    model_revision = "test-revision"

    def call(self, system_prompt, user_prompt):
        import json

        payload = json.loads(user_prompt)
        items = []
        for row in payload["items"]:
            text = row["text"].lower()
            if "strong" in text or "profit" in text:
                scores = {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
            elif "weak" in text or "loss" in text:
                scores = {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
            else:
                scores = {"positive": 0.20, "neutral": 0.70, "negative": 0.10}
            items.append({"index": row["index"], "confidence": max(scores.values()), **scores})
        return json.dumps({"items": items})


class CountingNewsSentimentBackend(StubNewsSentimentBackend):
    def __init__(self):
        self.calls = 0

    def call(self, system_prompt, user_prompt):
        self.calls += 1
        return super().call(system_prompt, user_prompt)


def test_llm_news_sentiment_alpha_raw_polarity_and_source_id():
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar(), "BBB": _bar(), "CCC": _bar()},
            "news": {
                "AAA": ["strong profit growth beat upgrade"],
                "BBB": ["weak loss warning downgrade"],
            },
        },
    })
    alpha = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=StubNewsSentimentBackend(),
        lookback=1,
    )

    sigs = alpha.score(_obs(date(2024, 1, 2), ["AAA", "BBB", "CCC"]))
    by_ticker = {s.ticker: s for s in sigs}

    assert set(by_ticker) == {"AAA", "BBB"}   # CCC has no news -> no signal at all
    assert all(s.source_id == SIGNAL_ID for s in sigs)
    # Raw polarity (positive minus negative probability), no cross-sectional
    # normalisation: the sign itself is the information.
    assert by_ticker["AAA"].value == pytest.approx(0.80)   # 0.85 - 0.05
    assert by_ticker["BBB"].value == pytest.approx(-0.75)  # 0.05 - 0.80
    assert alpha.metadata()["nlp_model_name"] == DEFAULT_LLM_NEWS_MODEL
    assert alpha.metadata()["nlp_model_revision"] == "test-revision"
    assert alpha.metadata()["nlp_item_count"] == 2


def test_llm_news_sentiment_single_name_keeps_raw_polarity():
    # A cross-sectional z-score would send a single name to 0 regardless of tone.
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar()},
            "news": {"AAA": ["strong profit growth beat upgrade"]},
        },
    })
    alpha = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=StubNewsSentimentBackend(),
        lookback=1,
    )
    sigs = alpha.score(_obs(date(2024, 1, 2), ["AAA"]))
    assert len(sigs) == 1
    assert sigs[0].value == pytest.approx(0.80)


def test_llm_news_sentiment_alpha_emits_no_signal_without_text():
    # No text in the lookback window -> NO signal, not a fabricated neutral 0.0.
    # A fabricated neutral would mark the news context available and turn a
    # data gap into a votable judgment.
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {"price": {"AAA": _bar(), "BBB": _bar()}},
    })
    sigs = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=StubNewsSentimentBackend(),
    ).score(_obs(date(2024, 1, 2), ["AAA", "BBB"]))

    assert sigs == []


def test_llm_news_sentiment_alpha_does_not_use_future_news():
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar(), "BBB": _bar()},
            "news": {"AAA": ["strong profit growth"]},
        },
        date(2024, 1, 3): {
            "price": {"AAA": _bar(), "BBB": _bar()},
            "news": {"BBB": ["strong profit growth beat"]},
        },
    })
    alpha = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=StubNewsSentimentBackend(),
        lookback=2,
    )

    s_today = {s.ticker: s.value for s in alpha.score(_obs(date(2024, 1, 2), ["AAA", "BBB"]))}
    s_tomorrow = {s.ticker: s.value for s in alpha.score(_obs(date(2024, 1, 3), ["AAA", "BBB"]))}

    # BBB's news is dated 2024-01-03: invisible (no signal) on 01-02, positive on 01-03.
    assert s_today["AAA"] > 0
    assert "BBB" not in s_today
    assert s_tomorrow["BBB"] > 0


def test_llm_news_sentiment_alpha_rejects_invalid_probabilities():
    class BrokenBackend:
        model_name = "broken"
        model_revision = None

        def call(self, system_prompt, user_prompt):
            return '{"items":[{"index":0,"positive":0.9,"neutral":0.1}]}'

    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar()},
            "news": {"AAA": ["strong profit growth"]},
        },
    })
    alpha = LLMNewsSentimentAlpha(LoaderTextPanel(ds), backend=BrokenBackend())

    try:
        alpha.score(_obs(date(2024, 1, 2), ["AAA"]))
    except ValueError as exc:
        assert "positive/neutral/negative" in str(exc)
    else:
        raise AssertionError("expected invalid classifier output to fail fast")


def test_llm_news_sentiment_cache_prevents_second_run_calls(tmp_path):
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar(), "BBB": _bar()},
            "news": {
                "AAA": ["strong profit growth"],
                "BBB": ["weak loss warning"],
            },
        },
    })
    cache_path = tmp_path / "news_sentiment_cache.jsonl"
    cache = NewsSentimentCache(cache_path)
    backend = CountingNewsSentimentBackend()
    alpha = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=backend,
        cache=cache,
        lookback=1,
    )

    alpha.score(_obs(date(2024, 1, 2), ["AAA", "BBB"]))
    first_metadata = alpha.metadata()
    assert backend.calls == 2
    assert first_metadata["nlp_cache_misses"] == 2
    assert first_metadata["nlp_cache_hits"] == 0
    assert first_metadata["nlp_config_hash"]

    second_backend = CountingNewsSentimentBackend()
    second_alpha = LLMNewsSentimentAlpha(
        LoaderTextPanel(ds),
        backend=second_backend,
        cache=NewsSentimentCache(cache_path),
        lookback=1,
    )
    second_alpha.score(_obs(date(2024, 1, 2), ["AAA", "BBB"]))
    second_metadata = second_alpha.metadata()

    assert second_backend.calls == 0
    assert second_metadata["nlp_cache_hits"] == 2
    assert second_metadata["nlp_cache_misses"] == 0
    assert second_metadata["nlp_model_name"] == DEFAULT_LLM_NEWS_MODEL
    assert second_metadata["nlp_model_revision"] == "test-revision"
