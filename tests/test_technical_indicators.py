"""Time-series technical signals and the TradingAgents-style indicator block.

The load-bearing property: a SINGLE-NAME universe must still produce
informative technical inputs. Cross-sectional z-scores degenerate to 0 with
one ticker; the time-series formulation compares the stock with its own
prior history instead.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

from trading_agent.alpha.lowvol import LowVolAlpha
from trading_agent.alpha.momentum import MomentumAlpha, ts_zscore
from trading_agent.alpha.reversal import ReversalAlpha
from trading_agent.alpha.technical_indicators import TechnicalIndicatorBlock
from trading_agent.contracts import Bar, Observation
from trading_agent.market import LoaderPricePanel


class _Loader:
    def __init__(self, prices_by_date):
        self._prices_by_date = prices_by_date

    def get_date_range(self):
        return sorted(self._prices_by_date)

    def get_data_by_date(self, d):
        return {
            "price": {
                ticker: {"adjusted_close": price, "close": price}
                for ticker, price in self._prices_by_date[d].items()
            }
        }


def _panel(closes: list[float], ticker: str = "AAA") -> tuple[LoaderPricePanel, list]:
    dates = list(pd.bdate_range("2023-01-02", periods=len(closes)).date)
    prices = {d: {ticker: closes[i]} for i, d in enumerate(dates)}
    return LoaderPricePanel(_Loader(prices)), dates


def _obs(day: date, tickers=("AAA",)) -> Observation:
    bar = Bar(open=1, high=1, low=1, close=1, adjusted_close=1, volume=1)
    return Observation(
        decision_date=day,
        decision_time=datetime(day.year, day.month, day.day, 9, 30),
        availability_time=datetime(day.year, day.month, day.day, 9, 30),
        bars={ticker: bar for ticker in tickers},
        tickers=tuple(tickers),
    )


def _wavy_closes(n: int, spike_last: float = 1.06) -> list[float]:
    # Oscillating series with a final up-spike: rolling stats have spread, and
    # the current value sits away from its own historical mean.
    closes = [100.0 * (1.0 + 0.002 * math.sin(i / 3.0)) for i in range(n)]
    closes[-1] = closes[-2] * spike_last
    return closes


def test_single_name_momentum_is_nonzero():
    panel, dates = _panel(_wavy_closes(320))
    signals = list(MomentumAlpha(panel).score(_obs(dates[-1])))
    assert len(signals) == 1
    assert signals[0].source_id == "ts_momentum"
    assert signals[0].value > 0  # final spike beats the stock's own history


def test_single_name_reversal_is_nonzero_and_negative_after_a_spike():
    # Reversal skips the most recent day: the spike sits at t-1, so the 5d
    # skip-1 NEGATIVE return is deeply negative vs its own history.
    closes = _wavy_closes(320)
    closes.append(closes[-1])  # one more flat day so the spike is at t-1
    panel, dates = _panel(closes)
    signals = list(ReversalAlpha(panel).score(_obs(dates[-1])))
    assert len(signals) == 1
    assert signals[0].source_id == "ts_reversal"
    assert signals[0].value < 0


def test_single_name_lowvol_flags_a_calm_stock_positive():
    # Volatile first half, calm second half: current 60d vol is far below the
    # stock's own history, so the negative-vol z-score is positive.
    n = 400
    closes = [
        100.0 * (1.0 + (0.02 if i < n // 2 else 0.001) * math.sin(i / 2.0))
        for i in range(n)
    ]
    panel, dates = _panel(closes)
    signals = list(LowVolAlpha(panel).score(_obs(dates[-1])))
    assert len(signals) == 1
    assert signals[0].source_id == "ts_lowvol"
    assert signals[0].value > 0


def test_insufficient_history_yields_no_signal_not_zero():
    panel, dates = _panel(_wavy_closes(40))  # < lookback + min_history + 1
    assert list(MomentumAlpha(panel).score(_obs(dates[-1]))) == []
    assert list(ReversalAlpha(panel).score(_obs(dates[-1]))) == []
    assert list(LowVolAlpha(panel).score(_obs(dates[-1]))) == []


def test_ts_zscore_prior_only_history():
    assert ts_zscore(1.0, []) == 0.0
    assert ts_zscore(1.0, [1.0, 1.0]) == 0.0  # zero spread guarded
    z = ts_zscore(2.0, [0.0, 1.0, -1.0, 0.5, -0.5])
    assert z > 1.0


def test_indicator_block_readings_on_a_rising_series():
    closes = [100.0 * (1.005 ** i) for i in range(260)]
    panel, dates = _panel(closes)
    block = TechnicalIndicatorBlock(panel)
    indicators = block.indicators_for("AAA", dates[-1])
    assert indicators is not None
    # Steady uptrend: price above its own averages, momentum positive, RSI high.
    assert indicators["close_10_ema_gap"] > 0
    assert indicators["close_50_sma_gap"] > 0
    assert indicators["close_200_sma_gap"] > 0
    assert indicators["macd_pct"] > 0
    assert indicators["rsi"] > 50
    assert indicators["boll_position"] > 0


def test_indicator_block_short_history_returns_none():
    panel, dates = _panel([100.0 + i for i in range(20)])
    assert TechnicalIndicatorBlock(panel).indicators_for("AAA", dates[-1]) is None


def test_indicator_block_degrades_gracefully_without_stockstats(monkeypatch):
    import trading_agent.alpha.technical_indicators as mod

    panel, dates = _panel(_wavy_closes(260))
    monkeypatch.setattr(mod, "_wrap_closes", lambda closes: None)
    assert TechnicalIndicatorBlock(panel).indicators_for("AAA", dates[-1]) is None


def test_indicator_block_is_point_in_time():
    closes = _wavy_closes(320)
    panel_short, dates_short = _panel(closes[:300])
    panel_full, dates_full = _panel(closes)
    t = dates_short[-1]
    assert dates_full[299] == t
    block_short = TechnicalIndicatorBlock(panel_short).indicators_for("AAA", t)
    block_full = TechnicalIndicatorBlock(panel_full).indicators_for("AAA", t)
    assert block_short == block_full  # future rows change nothing at t
