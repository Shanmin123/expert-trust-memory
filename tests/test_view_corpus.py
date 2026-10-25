"""Tests for the frozen expert-view corpus (record/replay reproducibility)."""

from __future__ import annotations

import json

import pytest

from trading_agent.alpha.market_state_classifier import MarketStateConfig
from trading_agent.experts import FactorExpert, FundamentalExpert, NewsExpert, TechnicalExpert
from trading_agent.experts.view_corpus import (
    READTHROUGH,
    RECORD,
    REPLAY,
    REPLAY_BY_DATE,
    CachedExpertBackend,
    ViewCorpus,
    attach_corpus,
    _cache_key,
)
from trading_agent.runner import run_expert_loop


class CountingBackend:
    """Records how many times the inner backend is actually called."""

    def __init__(self, confidence: float):
        self._confidence = confidence
        self.calls = 0

    def call(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        tickers = json.loads(user_prompt).get("tickers", [])
        return json.dumps([
            {
                "ticker": ticker,
                "direction": "long",
                "confidence": self._confidence,
                "reasoning": f"call#{self.calls}",
                "evidence": ["e"],
                "risk_flags": [],
                "data_quality": "medium",
            }
            for ticker in tickers
        ])


def _experts(corpus, mode=None):
    experts = [
        TechnicalExpert(CountingBackend(0.9)),
        FactorExpert(CountingBackend(0.6)),
        NewsExpert(CountingBackend(0.5)),
        FundamentalExpert(CountingBackend(0.4)),
    ]
    return attach_corpus(experts, corpus, mode=mode)


def test_cache_key_separates_expert_and_prompt():
    a = _cache_key("technical", "user")
    b = _cache_key("factor", "user")
    c = _cache_key("technical", "user2")
    assert a != b
    assert a != c
    assert a == _cache_key("technical", "user")
    # readable, so a reader can see what a record is about
    assert a == ("technical", "user")


def test_record_then_replay_returns_identical_response_without_inner_calls(tmp_path):
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    rec = CachedExpertBackend(inner, ViewCorpus(path, mode=RECORD), "technical", RECORD)
    user = json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    first = rec.call("sys", user)
    assert inner.calls == 1

    replay_backend = CachedExpertBackend(inner, ViewCorpus(path, mode=REPLAY), "technical", REPLAY)
    replayed = replay_backend.call("sys", user)
    assert replayed == first
    assert inner.calls == 1  # replay must not touch the inner backend


def test_replay_miss_raises(tmp_path):
    path = tmp_path / "corpus.jsonl"
    CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=RECORD), "technical", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    )
    replay = CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=REPLAY), "technical", REPLAY)
    with pytest.raises(KeyError, match="replay corpus miss"):
        replay.call("sys", json.dumps({"date": "2099-01-01", "tickers": ["ZZZ"]}))


def test_replay_mode_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="replay corpus not found"):
        ViewCorpus(tmp_path / "absent.jsonl", mode=REPLAY)


def test_replay_by_date_serves_frozen_view_when_prompt_drifts(tmp_path):
    # A frozen view is the expert's decision for a name/date; downstream experiments
    # that keep the views fixed must replay it even when prompt serialisation drifts
    # (the sealed-corpus-on-current-tree situation). replay_by_date serves it by the
    # exact (expert, date, tickers) identity when the prompt hash misses.
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    recorded = CachedExpertBackend(inner, ViewCorpus(path, mode=RECORD), "news", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "signals": 1})
    )
    assert inner.calls == 1

    drifted = json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "signals": 2})  # same id, new string
    replay = CachedExpertBackend(inner, ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    served = replay.call("sys", drifted)
    assert served == recorded          # identity fallback served the frozen view
    assert inner.calls == 1            # and did not touch the inner backend


def test_strict_replay_still_raises_when_prompt_drifts(tmp_path):
    # The strict prompt-hash guarantee (freeze check) must be UNCHANGED: a drifted
    # prompt is still a hard miss under plain `replay`.
    path = tmp_path / "corpus.jsonl"
    CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=RECORD), "news", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "signals": 1})
    )
    strict = CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=REPLAY), "news", REPLAY)
    with pytest.raises(KeyError, match="replay corpus miss"):
        strict.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "signals": 2}))


def test_replay_by_date_raises_on_unknown_request(tmp_path):
    # replay_by_date is still strict about coverage: a name/date the corpus never
    # recorded is a hard miss, not a silent backend call.
    path = tmp_path / "corpus.jsonl"
    CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=RECORD), "news", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    )
    replay = CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    with pytest.raises(KeyError, match="replay corpus miss"):
        replay.call("sys", json.dumps({"date": "2099-01-01", "tickers": ["ZZZ"]}))


def test_replay_by_date_assembles_a_pooled_multi_ticker_request(tmp_path):
    # Several single-name corpora replay as ONE pooled universe: a multi-ticker request
    # with no exact record is assembled from the per-ticker frozen responses for the same
    # (expert, date), so cross-name pooling costs no backend calls.
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    rec = CachedExpertBackend(inner, ViewCorpus(path, mode=RECORD), "news", RECORD)
    rec.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"]}))
    rec.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["BBB"]}))
    assert inner.calls == 2

    pooled = CachedExpertBackend(inner, ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    served = json.loads(
        pooled.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA", "BBB"]}))
    )
    assert [v["ticker"] for v in served] == ["AAA", "BBB"]   # merged into one committee
    assert inner.calls == 2                                  # and no new backend call


def test_pooled_assembly_omits_a_name_with_no_view_that_date(tmp_path):
    # Ragged coverage is normal in a multi-name universe (a name starts late or has a
    # gap). The assembly serves the names it has; the missing one is omitted and takes
    # the expert's data-unavailable safe default, exactly as it would have live.
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    CachedExpertBackend(inner, ViewCorpus(path, mode=RECORD), "news", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    )
    pooled = CachedExpertBackend(inner, ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    served = json.loads(
        pooled.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA", "BBB"]}))
    )
    assert [v["ticker"] for v in served] == ["AAA"]   # BBB omitted -> abstains
    assert inner.calls == 1                           # still no new backend call


def test_pooled_assembly_is_a_hard_miss_when_no_name_has_a_view(tmp_path):
    # Coverage is still enforced: a date no name was recorded on is a hard miss, never a
    # silent backend call.
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    CachedExpertBackend(inner, ViewCorpus(path, mode=RECORD), "news", RECORD).call(
        "sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    )
    pooled = CachedExpertBackend(inner, ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    with pytest.raises(KeyError, match="replay corpus miss"):
        pooled.call("sys", json.dumps({"date": "2099-01-01", "tickers": ["AAA", "BBB"]}))


def test_replay_by_date_uses_last_recorded_on_conflict(tmp_path):
    # When a name/date/expert view was recorded more than once (e.g. across
    # recording passes whose prompts drifted), replay_by_date serves the most
    # recent frozen response, mirroring the store's last-wins for a repeated hash key.
    path = tmp_path / "corpus.jsonl"
    rec = CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=RECORD), "news", RECORD)
    rec.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "v": 1}))   # call#1, distinct prompt
    rec.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "v": 2}))   # call#2, distinct prompt
    replay = CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=REPLAY_BY_DATE), "news", REPLAY_BY_DATE)
    served = replay.call("sys", json.dumps({"date": "2022-01-03", "tickers": ["AAA"], "v": 3}))  # hash miss -> identity
    assert json.loads(served)[0]["reasoning"] == "call#2"   # the LAST recorded response for this date-key wins


def test_readthrough_records_on_miss_and_reuses_after(tmp_path):
    path = tmp_path / "corpus.jsonl"
    inner = CountingBackend(0.7)
    backend = CachedExpertBackend(inner, ViewCorpus(path, mode=READTHROUGH), "news", READTHROUGH)
    user = json.dumps({"date": "2022-01-03", "tickers": ["AAA"]})
    backend.call("sys", user)
    backend.call("sys", user)
    assert inner.calls == 1  # second call served from corpus


def test_corpus_persists_metadata_across_reopen(tmp_path):
    path = tmp_path / "corpus.jsonl"
    CachedExpertBackend(CountingBackend(0.7), ViewCorpus(path, mode=RECORD), "factor", RECORD).call(
        "sys", json.dumps({"date": "2022-01-04", "tickers": ["AAA", "BBB"]})
    )
    reopened = ViewCorpus(path, mode=REPLAY)
    assert len(reopened) == 1
    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["expert_id"] == "factor"
    assert line["date"] == "2022-01-04"
    assert line["tickers"] == ["AAA", "BBB"]
    assert line["user_prompt"]
    # the record carries no digest of itself
    assert "prompt_sha" not in line and "key" not in line


def test_full_loop_record_then_replay_reproduces_backtest(regime_dataset, mem_backtest_config, tmp_path):
    cfg = MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2)
    path = tmp_path / "loop_corpus.jsonl"

    record_corpus = ViewCorpus(path, mode=RECORD)
    rec_result = run_expert_loop(
        regime_dataset, mem_backtest_config, _experts(record_corpus, RECORD),
        market_state_config=cfg, allow_reflection=False,
    )
    assert record_corpus.records_written > 0

    replay_corpus = ViewCorpus(path, mode=REPLAY)
    replay_experts = _experts(replay_corpus, REPLAY)
    rep_result = run_expert_loop(
        regime_dataset, mem_backtest_config, replay_experts,
        market_state_config=cfg, allow_reflection=False,
    )

    assert rep_result.backtest.equity_curve == rec_result.backtest.equity_curve
    assert replay_corpus.hits > 0
    # Every inner backend call during replay must be avoided.
    assert all(e._backend._inner.calls == 0 for e in replay_experts)


def test_corrupt_lines_are_skipped_not_fatal(tmp_path):
    # A torn line from an interrupted/concurrent writer must not brick the corpus.
    import json as _json
    from trading_agent.experts.view_corpus import CorpusRecord, ViewCorpus

    path = tmp_path / "corpus.jsonl"
    good = CorpusRecord(
        expert_id="technical", date="2024-01-02", tickers=("AAA",),
        response="{}", user_prompt="{}",
    )
    path.write_text(
        _json.dumps(good.to_json()) + "\n" + '{"expert_id": "technical", "trunc' + "\n",
        encoding="utf-8",
    )
    corpus = ViewCorpus(path, mode="readthrough")
    assert len(corpus) == 1
    assert corpus.lookup("technical", "{}") == "{}"
    assert corpus.corrupt_lines == 1
