"""Tests for the B1 abnormal attention/novelty news alpha."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from statistics import mean, pstdev

from trading_agent.alpha.text_attention import (
    NewsAttentionConfig,
    NewsAttentionNoveltyAlpha,
    SIGNAL_ID,
)
from trading_agent.contracts import Observation
from trading_agent.data.finsaber_news import FinsaberNewsPanel, NewsItem


def _dates(n: int, start=date(2024, 1, 2)):
    return [start + timedelta(days=i) for i in range(n)]


def _obs(t, tickers):
    as_of = datetime.combine(t, time.min)
    return Observation(decision_date=t, decision_time=as_of, availability_time=as_of,
                       bars={}, tickers=tuple(tickers))


def _alpha(items, trading_dates, min_history=3, baseline_lookback=3, novelty_lookback=3):
    return NewsAttentionNoveltyAlpha(
        FinsaberNewsPanel.from_items(items),
        NewsAttentionConfig(
            novelty_lookback=novelty_lookback,
            baseline_lookback=baseline_lookback,
            min_history=min_history,
        ),
        trading_dates=trading_dates,
    )


def test_attention_features_dedup_and_length_fallbacks():
    d = date(2024, 1, 2)
    alpha = _alpha([
        NewsItem(d, "AAA", "same", text_len=100, text_crc32=7),
        NewsItem(d, "AAA", "same syndicated copy", text_len=200, text_crc32=7),
        NewsItem(d, "AAA", "fallback", text_len=None, text_crc32=None),
    ], [d], min_history=1)

    f = alpha.daily_features("AAA", d)

    assert f.raw_count == 3
    assert f.dedup_count == 2
    assert abs(f.dup_share - 1 / 3) < 1e-12
    assert f.total_len == 100 + len("fallback")


def test_attention_novelty_uses_prior_same_ticker_only():
    d0, d1 = _dates(2)
    only_other_ticker_prior = _alpha([
        NewsItem(d0, "BBB", "merger approval unique phrase", text_crc32=1),
        NewsItem(d1, "AAA", "merger approval unique phrase", text_crc32=2),
    ], [d0, d1], min_history=1)
    same_ticker_prior = _alpha([
        NewsItem(d0, "AAA", "merger approval unique phrase", text_crc32=1),
        NewsItem(d1, "AAA", "merger approval unique phrase", text_crc32=2),
    ], [d0, d1], min_history=1)

    assert only_other_ticker_prior.daily_features("AAA", d1).novelty_sum == 1.0
    assert same_ticker_prior.daily_features("AAA", d1).novelty_sum == 0.0


def test_attention_baseline_excludes_current_date():
    ds = _dates(4)
    alpha = _alpha([
        NewsItem(ds[1], "AAA", "old item", text_len=10, text_crc32=1),
        NewsItem(ds[3], "AAA", "fresh one", text_len=10, text_crc32=2),
        NewsItem(ds[3], "AAA", "fresh two", text_len=10, text_crc32=3),
        NewsItem(ds[3], "AAA", "fresh three", text_len=10, text_crc32=4),
    ], ds, min_history=3, baseline_lookback=3)

    f = alpha.features("AAA", ds[3])
    count_hist = [math.log1p(v) for v in [0, 1, 0]]
    expected = (math.log1p(3) - mean(count_hist)) / pstdev(count_hist)

    assert abs(f.abn_count - expected) < 1e-12
    assert f.abn_count > 0


def test_attention_future_news_does_not_affect_date_t_score():
    ds = _dates(5)
    base_items = [
        NewsItem(ds[0], "AAA", "prior", text_crc32=1),
        NewsItem(ds[2], "AAA", "today", text_crc32=2),
    ]
    with_future = base_items + [NewsItem(ds[4], "AAA", "future future future", text_crc32=3)]
    alpha_base = _alpha(base_items, ds, min_history=2, baseline_lookback=2)
    alpha_future = _alpha(with_future, ds, min_history=2, baseline_lookback=2)

    assert alpha_base.features("AAA", ds[2]) == alpha_future.features("AAA", ds[2])


def test_attention_score_is_the_time_series_raw_score_not_recentred():
    # The per-ticker raw_score is already the prior-only abnormal score; the
    # signal must pass it through unchanged. Re-centring across the universe
    # would degenerate to 0.0 for a single name (silencing the channel) and
    # throw away the level information for small universes.
    ds = _dates(4)
    items = [
        NewsItem(ds[0], "AAA", "old", text_crc32=1),
        NewsItem(ds[1], "BBB", "old", text_crc32=2),
        NewsItem(ds[3], "AAA", "fresh one", text_crc32=3),
        NewsItem(ds[3], "AAA", "fresh two", text_crc32=4),
        NewsItem(ds[3], "BBB", "fresh", text_crc32=5),
    ]
    alpha = _alpha(items, ds, min_history=3, baseline_lookback=3)
    sigs = {s.ticker: s for s in alpha.score(_obs(ds[3], ["AAA", "BBB", "CCC"]))}

    assert set(sigs) == {"AAA", "BBB", "CCC"}
    assert all(s.source_id == SIGNAL_ID for s in sigs.values())
    assert all(math.isfinite(s.value) for s in sigs.values())
    for ticker in ("AAA", "BBB", "CCC"):
        assert sigs[ticker].value == alpha.raw_score(ticker, ds[3])
    assert sigs["AAA"].value > 0          # news burst vs own quiet history


def test_attention_single_name_burst_is_nonzero():
    # THE regression: one-ticker universe must still see its own news burst.
    ds = _dates(4)
    items = [
        NewsItem(ds[0], "AAA", "old", text_crc32=1),
        NewsItem(ds[3], "AAA", "fresh one", text_crc32=2),
        NewsItem(ds[3], "AAA", "fresh two", text_crc32=3),
    ]
    alpha = _alpha(items, ds, min_history=3, baseline_lookback=3)
    sigs = alpha.score(_obs(ds[3], ["AAA"]))
    assert len(sigs) == 1
    assert sigs[0].value > 0


def test_attention_all_no_news_outputs_neutral_zero():
    d = date(2024, 1, 2)
    alpha = _alpha([], [d], min_history=1)

    assert [s.value for s in alpha.score(_obs(d, ["AAA", "BBB"]))] == [0.0, 0.0]
