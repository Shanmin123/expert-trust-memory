"""Point-in-time + determinism guards for the deterministic TradingView-style rating."""

from __future__ import annotations

from datetime import date

import pytest


class _ClosePanel:
    """Minimal close-only panel exposing the trailing() API the rating needs."""

    def __init__(self, series):
        self._series = series  # list[(date, close)]

    def trailing(self, ticker, t, n):
        vals = [c for d, c in self._series if d <= t]
        return vals[-n:]


def _ramp_panel(n=260):
    base = date(2020, 1, 1)
    from datetime import timedelta
    # gently rising series so long MAs are defined and price sits above them
    return _ClosePanel([(base + timedelta(days=i), 100.0 + i * 0.5) for i in range(n)])


def test_rating_is_in_range_and_long_on_uptrend():
    from trading_agent.alpha.technical_rating import TechnicalRating
    panel = _ramp_panel()
    r = TechnicalRating(panel)
    out = r.rating_for("X", date(2020, 9, 1))
    assert out is not None
    for k in ("ma", "oscillator", "summary"):
        if out[k] is not None:
            assert -1.0 <= out[k] <= 1.0
    # price strictly above all its trailing averages -> MA rating should be bullish
    assert out["ma"] is not None and out["ma"] > 0
    assert r.direction_for("X", date(2020, 9, 1)) == "long"


def test_rating_is_point_in_time():
    # Truncating the series at t must not change the rating computed at t: the
    # reading uses only closes <= t (no look-ahead into later bars).
    from trading_agent.alpha.technical_rating import TechnicalRating
    full = _ramp_panel(260)
    t = full._series[180][0]
    truncated = _ClosePanel(full._series[:181])   # everything up to and including t
    a = TechnicalRating(full).rating_for("X", t)
    b = TechnicalRating(truncated).rating_for("X", t)
    assert a == b


def test_negative_threshold_lifts_long_rate_participation():
    from trading_agent.alpha.technical_rating import TechnicalRating
    panel = _ramp_panel()
    r = TechnicalRating(panel)
    t = date(2020, 9, 1)
    # a lower long-threshold can only turn flats into longs, never the reverse
    strict = r.direction_for("X", t, long_threshold=0.5)
    loose = r.direction_for("X", t, long_threshold=-0.5)
    assert not (strict == "long" and loose == "flat")


class _OHLCVPanel:
    """Minimal OHLCV panel exposing trailing_frame() for the full rating."""

    def __init__(self, rows):
        self._rows = rows  # list[(date, o,h,l,c,v)]

    def trailing_frame(self, ticker, t, n):
        import pandas as pd
        vals = [(o, h, l, c, v) for d, o, h, l, c, v in self._rows if d <= t]
        vals = vals[-n:]
        if not vals:
            return None
        return pd.DataFrame(vals, columns=("open", "high", "low", "close", "volume"))


def _ohlcv_ramp(n=300):
    from datetime import date, timedelta
    base = date(2020, 1, 1)
    rows = []
    for i in range(n):
        c = 100.0 + i * 0.5
        rows.append((base + timedelta(days=i), c - 0.3, c + 0.4, c - 0.5, c, 1_000_000.0))
    return _OHLCVPanel(rows)


def test_full_rating_uses_many_indicators_and_is_bounded():
    from trading_agent.alpha.technical_rating import TradingViewRating
    from datetime import date
    r = TradingViewRating(_ohlcv_ramp())
    out = r.rating_for("X", date(2020, 9, 1))
    assert out is not None
    assert out["n_ma"] >= 12 and out["n_osc"] >= 8   # near-full menu voted
    for k in ("ma", "oscillator", "summary"):
        assert -1.0 <= out[k] <= 1.0
    assert out["ma"] > 0                              # price above its averages on an uptrend


def test_full_rating_is_point_in_time():
    from trading_agent.alpha.technical_rating import TradingViewRating
    full = _ohlcv_ramp(300)
    t = full._rows[200][0]
    truncated = _OHLCVPanel(full._rows[:201])
    a = TradingViewRating(full).rating_for("X", t)
    b = TradingViewRating(truncated).rating_for("X", t)
    assert a == b
