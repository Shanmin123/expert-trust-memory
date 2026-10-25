"""Deterministic TradingView-style technical rating (close-computable subset).

TradingView's "Technical Ratings" widget votes ~26 indicators Buy/Sell/Neutral and
averages them into a Moving-Averages rating, an Oscillators rating, and a Summary.
This reimplements that rule set LOCALLY and STRICTLY POINT-IN-TIME (each reading uses
only closes <= t) so it is reproducible and leak-free -- never scrape the live widget,
whose values are computed on the latest bar and would inject look-ahead into a backtest.

Close-only subset: the FINSABER panel exposes adjusted close, so this keeps the
close-computable indicators (MA votes at 10/20/30/50/100/200 for both EMA and SMA;
RSI, MACD, Momentum, Stochastic-RSI oscillators) and OMITS the high/low/volume ones
from the full widget (ADX/DI, CCI, Awesome, Williams %R, UO, BBP, VWMA), matching the
same limitation already documented in ``technical_indicators.py``. Reuses ``stockstats``
rather than reimplementing formulas, per the project's open-source-reuse rule.

Rating in [-1, 1] = mean(MA-rating, Oscillator-rating); each sub-rating is
(#buy - #sell) / #available. Direction for a binary long/flat committee: long when
summary > ``long_threshold`` (default 0), else flat.
"""

from __future__ import annotations

from typing import Optional

# EMA/SMA lookbacks voted by the moving-average block (TradingView menu, close subset).
_MA_LOOKBACKS = (10, 20, 30, 50, 100, 200)


def _wrap_closes(closes: list[float]):
    try:
        import pandas as pd
        from stockstats import wrap
    except ImportError:
        return None
    return wrap(pd.DataFrame({"close": [float(c) for c in closes]}))


def _last(frame, column) -> Optional[float]:
    try:
        import math
        v = float(frame[column].iloc[-1])
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


class TechnicalRating:
    """Per-ticker point-in-time TradingView-style rating over a close panel."""

    def __init__(self, panel, history_days: int = 260, min_history: int = 30):
        if min_history < 15 or history_days < min_history:
            raise ValueError("need history_days >= min_history >= 15")
        self._panel = panel
        self._history_days = history_days
        self._min_history = min_history

    def _ma_votes(self, frame, close: float, n: int) -> list[int]:
        votes = []
        for lb in _MA_LOOKBACKS:
            if n < lb:
                continue  # not enough history for this average -> not voted
            for kind in ("ema", "sma"):
                ma = _last(frame, f"close_{lb}_{kind}")
                if ma is None or ma <= 0:
                    continue
                votes.append(1 if close > ma else (-1 if close < ma else 0))
        return votes

    def _oscillator_votes(self, frame, closes: list[float]) -> list[int]:
        votes: list[int] = []
        # RSI(14): oversold buy / overbought sell.
        rsi = _last(frame, "rsi")
        if rsi is not None:
            votes.append(1 if rsi < 30 else (-1 if rsi > 70 else 0))
        # MACD(12,26,9): line vs signal.
        macd, macds = _last(frame, "macd"), _last(frame, "macds")
        if macd is not None and macds is not None:
            votes.append(1 if macd > macds else (-1 if macd < macds else 0))
        # Momentum(10): sign of the 10-bar price change.
        if len(closes) >= 11:
            mom = closes[-1] - closes[-11]
            votes.append(1 if mom > 0 else (-1 if mom < 0 else 0))
        # Stochastic-RSI(14): oversold buy / overbought sell, from the RSI series.
        try:
            rsi_series = [float(x) for x in frame["rsi"].tolist()[-14:]]
            rsi_series = [x for x in rsi_series if x == x]  # drop NaN
            if len(rsi_series) >= 14:
                lo, hi = min(rsi_series), max(rsi_series)
                if hi - lo > 0:
                    stochrsi = (rsi_series[-1] - lo) / (hi - lo)
                    votes.append(1 if stochrsi < 0.2 else (-1 if stochrsi > 0.8 else 0))
        except Exception:
            pass
        return votes

    def _sub_rating(self, votes: list[int]) -> Optional[float]:
        return (sum(votes) / len(votes)) if votes else None

    def rating_for(self, ticker: str, t) -> Optional[dict]:
        """{'ma','oscillator','summary'} ratings in [-1,1], or None when unavailable."""
        closes = self._panel.trailing(ticker, t, self._history_days)
        if len(closes) < self._min_history or any(c <= 0 for c in closes):
            return None
        frame = _wrap_closes(closes)
        if frame is None:
            return None
        close = float(closes[-1])
        ma = self._sub_rating(self._ma_votes(frame, close, len(closes)))
        osc = self._sub_rating(self._oscillator_votes(frame, closes))
        parts = [x for x in (ma, osc) if x is not None]
        if not parts:
            return None
        return {"ma": ma, "oscillator": osc, "summary": sum(parts) / len(parts)}

    def direction_for(self, ticker: str, t, long_threshold: float = 0.0) -> Optional[str]:
        r = self.rating_for(ticker, t)
        if r is None:
            return None
        return "long" if r["summary"] > long_threshold else "flat"


# --- full OHLCV rating -----------------------------------------------------

_FULL_MA_LOOKBACKS = (10, 20, 30, 50, 100, 200)


def _vote_level(value, buy_below, sell_above):
    """Oversold/overbought oscillator: buy below a floor, sell above a ceiling."""
    if value is None:
        return None
    return 1 if value < buy_below else (-1 if value > sell_above else 0)


def _vote_sign(value, prev=None):
    """Zero-line oscillator: buy above 0, sell below 0 (optionally require rising/falling)."""
    if value is None:
        return None
    if value > 0 and (prev is None or value >= prev):
        return 1
    if value < 0 and (prev is None or value <= prev):
        return -1
    return 0


class TradingViewRating:
    """Full TradingView-style Technical Rating over split-adjusted OHLCV, point-in-time.

    Reimplements the widget's 11 oscillators + 15 moving-average votes locally on this
    project's data (never scraping the live widget). stockstats supplies RSI, Stochastic
    (KDJ), CCI, ADX/DI, MACD, Williams %R; Awesome Oscillator, Ultimate Oscillator, Bull
    Bear Power, Ichimoku baseline, Hull MA and VWMA are computed by hand. Each indicator
    votes buy/sell/neutral by the standard rule; MA-rating and Oscillator-rating are the
    mean vote, and the summary is their mean, all in [-1, 1].
    """

    def __init__(self, ohlcv_panel, history_days: int = 320, min_history: int = 40):
        if min_history < 35 or history_days < min_history:
            raise ValueError("need history_days >= min_history >= 35 (200-SMA + buffer)")
        self._panel = ohlcv_panel
        self._history_days = history_days
        self._min_history = min_history

    # -- manual indicators (not in stockstats) --
    @staticmethod
    def _wma(series, n):
        import numpy as np
        if len(series) < n:
            return None
        w = np.arange(1, n + 1, dtype=float)
        return float((series[-n:] * w).sum() / w.sum())

    def _hull(self, close, n=9):
        import numpy as np
        half = self._wma(close, n // 2)
        full = self._wma(close, n)
        if half is None or full is None:
            return None
        raw = 2 * half - full
        # WMA(sqrt(n)) of the raw series: rebuild a short raw tail
        m = int(n ** 0.5)
        tail = []
        for k in range(m):
            h = self._wma(close[: len(close) - k], n // 2)
            f = self._wma(close[: len(close) - k], n)
            if h is None or f is None:
                return None
            tail.append(2 * h - f)
        tail = np.array(tail[::-1])
        return self._wma(tail, m)

    def rating_for(self, ticker: str, t) -> Optional[dict]:
        df = self._panel.trailing_frame(ticker, t, self._history_days)
        if df is None or len(df) < self._min_history:
            return None
        import numpy as np
        try:
            from stockstats import wrap
        except ImportError:
            return None
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)
        c = float(close[-1])
        s = wrap(df.copy())

        def col(name):
            try:
                return s[name]
            except Exception:
                return None

        def last(name, i=-1):
            ser = col(name)
            try:
                v = float(ser.iloc[i])
                return None if (v != v) else v
            except Exception:
                return None

        # ---- oscillators (11) ----
        osc: list[int] = []

        def add(v):
            if v is not None:
                osc.append(v)

        add(_vote_level(last("rsi"), 30, 70))                                  # RSI(14)
        # Stochastic %K/%D (KDJ): buy oversold & K>D, sell overbought & K<D
        k, d = last("kdjk"), last("kdjd")
        if k is not None and d is not None:
            add(1 if (k < 20 and k > d) else (-1 if (k > 80 and k < d) else 0))
        add(_vote_level(last("cci"), -100, 100))                                 # CCI(20)
        # ADX(14) with DI direction
        adx, pdi, mdi = last("adx"), last("pdi"), last("mdi")
        if adx is not None and pdi is not None and mdi is not None:
            add(1 if (adx > 20 and pdi > mdi) else (-1 if (adx > 20 and mdi > pdi) else 0))
        # Awesome Oscillator: SMA(median,5) - SMA(median,34)
        med = (high + low) / 2.0
        if len(med) >= 34:
            ao = med[-5:].mean() - med[-34:].mean()
            ao_prev = med[-6:-1].mean() - med[-35:-1].mean() if len(med) >= 35 else None
            add(_vote_sign(ao, ao_prev))
        # Momentum(10): TradingView votes by momentum DIRECTION (rising -> buy, falling ->
        # sell), not by sign — confirmed against the reference widget (mom=+0.39 shown as
        # Sell because it was falling).
        if len(close) >= 12:
            mom = float(close[-1] - close[-11])
            mom_prev = float(close[-2] - close[-12])
            add(1 if mom > mom_prev else (-1 if mom < mom_prev else 0))
        elif len(close) >= 11:
            add(_vote_sign(float(close[-1] - close[-11])))
        # MACD line vs signal
        macd, macds = last("macd"), last("macds")
        if macd is not None and macds is not None:
            add(1 if macd > macds else (-1 if macd < macds else 0))
        # Stochastic RSI(14) from the RSI series
        rsi_ser = col("rsi")
        if rsi_ser is not None:
            rv = [float(x) for x in rsi_ser.tolist()[-14:] if x == x]
            if len(rv) >= 14 and max(rv) > min(rv):
                srsi = (rv[-1] - min(rv)) / (max(rv) - min(rv))
                add(_vote_level(srsi, 0.20, 0.80))
        # Williams %R(14): buy oversold (<-80), sell overbought (>-20)
        add(_vote_level(last("wr"), -80, -20))
        # Bull Bear Power = (high - EMA13) + (low - EMA13)
        ema13 = last("close_13_ema")
        if ema13 is not None and len(high) >= 2:
            bbp = (high[-1] - ema13) + (low[-1] - ema13)
            ema13_prev = last("close_13_ema", -2)
            bbp_prev = ((high[-2] - ema13_prev) + (low[-2] - ema13_prev)
                        if ema13_prev is not None else None)
            add(_vote_sign(float(bbp), None if bbp_prev is None else float(bbp_prev)))
        # Ultimate Oscillator(7,14,28)
        if len(close) >= 29:
            prev_close = close[:-1]
            cur = close[1:]
            tl = np.minimum(low[1:], prev_close)
            bp = cur - tl
            tr = np.maximum(high[1:], prev_close) - tl
            def avg(nn):
                a, b = bp[-nn:].sum(), tr[-nn:].sum()
                return (a / b) if b > 0 else None
            a7, a14, a28 = avg(7), avg(14), avg(28)
            if None not in (a7, a14, a28):
                uo = 100 * (4 * a7 + 2 * a14 + a28) / 7.0
                add(1 if uo > 70 else (-1 if uo < 30 else 0))                     # UO buy>70 sell<30

        # ---- moving averages (15) ----
        ma: list[int] = []

        def ma_vote(value):
            if value is None or value <= 0:
                return
            ma.append(1 if c > value else (-1 if c < value else 0))

        n = len(close)
        for lb in _FULL_MA_LOOKBACKS:
            if n >= lb:
                ma_vote(last(f"close_{lb}_ema"))
                ma_vote(last(f"close_{lb}_sma"))
        # Ichimoku baseline (Kijun, 26)
        if n >= 26:
            ma_vote((high[-26:].max() + low[-26:].min()) / 2.0)
        # VWMA(20)
        if n >= 20 and vol[-20:].sum() > 0:
            ma_vote(float((close[-20:] * vol[-20:]).sum() / vol[-20:].sum()))
        # Hull MA(9)
        ma_vote(self._hull(close, 9))

        ma_r = (sum(ma) / len(ma)) if ma else None
        osc_r = (sum(osc) / len(osc)) if osc else None
        parts = [x for x in (ma_r, osc_r) if x is not None]
        if not parts:
            return None
        return {"ma": ma_r, "oscillator": osc_r, "summary": sum(parts) / len(parts),
                "n_osc": len(osc), "n_ma": len(ma)}

    def direction_for(self, ticker: str, t, long_threshold: float = 0.0) -> Optional[str]:
        r = self.rating_for(ticker, t)
        if r is None:
            return None
        return "long" if r["summary"] > long_threshold else "flat"
