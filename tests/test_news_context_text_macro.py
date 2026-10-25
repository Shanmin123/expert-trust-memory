"""Design D news context: recent news TEXT snippets + FRED macro block (TA news-analyst)."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from trading_agent.contracts import Bar, Observation
from trading_agent.data.finsaber_news import NewsItem
from trading_agent.expert_context import _news_context


def _obs(t=date(2024, 7, 3)):
    bar = Bar(open=100.0, high=101.0, low=99.0, close=100.0, adjusted_close=100.0, volume=1000)
    dt = datetime(t.year, t.month, t.day, 9, 30)
    return Observation(decision_date=t, decision_time=dt, availability_time=dt,
                       bars={"TSLA": bar}, tickers=["TSLA"])


class _FakeNewsPanel:
    def __init__(self, items):
        self._items = items

    def coverage(self, tickers, t):
        todays = [i for i in self._items if i.date == t]
        return (1.0 if todays else 0.0), len(todays), len(todays)

    def items_on(self, ticker, t):
        return [i for i in self._items if i.ticker == ticker and i.date == t]

    def dedup_items(self, items):
        return list(items)

    def trailing_items(self, ticker, t, lookback, include_current=True):
        lo = t - timedelta(days=lookback)
        return [i for i in self._items
                if i.ticker == ticker and lo <= i.date <= (t if include_current else t - timedelta(days=1))]


class _FakeMacro:
    def features_for(self, t):
        return {"available": True, "fed_funds_rate": 5.33, "yield_curve_10y_minus_2y": -0.35}


def test_recent_news_snippets_are_latest_first_capped_and_pit():
    t = date(2024, 7, 3)
    items = [
        NewsItem(date=t - timedelta(days=d), ticker="TSLA", text=f"headline {d} " + "x" * 300)
        for d in range(0, 9)   # 9 days back; only <=7 within lookback
    ] + [NewsItem(date=t + timedelta(days=1), ticker="TSLA", text="FUTURE leak")]
    panel = _FakeNewsPanel(items)
    out = _news_context(_obs(t), news_panel=panel)
    snippets = out["by_ticker"]["TSLA"]["recent_news"]
    assert len(snippets) == 5                                # capped
    assert snippets[0].startswith("headline 0")              # latest first
    assert all(len(s) <= 160 for s in snippets)              # prompt-size cap
    assert not any("FUTURE" in s for s in snippets)          # point-in-time


def test_macro_block_attached_when_available():
    t = date(2024, 7, 3)
    panel = _FakeNewsPanel([NewsItem(date=t, ticker="TSLA", text="hi")])
    out = _news_context(_obs(t), news_panel=panel, macro_alpha=_FakeMacro())
    assert out["macro"]["fed_funds_rate"] == 5.33
    assert out["macro"]["yield_curve_10y_minus_2y"] == -0.35


def test_no_macro_key_when_macro_absent():
    t = date(2024, 7, 3)
    panel = _FakeNewsPanel([NewsItem(date=t, ticker="TSLA", text="hi")])
    out = _news_context(_obs(t), news_panel=panel, macro_alpha=None)
    assert "macro" not in out
