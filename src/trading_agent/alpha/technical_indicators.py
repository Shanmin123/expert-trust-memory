"""TradingAgents-style technical indicator block for the technical expert.

Reuses the indicator menu of the TradingAgents market analyst (see
external/research_refs/TradingAgents, agents/analysts/market_analyst.py) and the
same maintained library it uses (``stockstats``) instead of reimplementing the
formulas. Computed strictly point-in-time on this project's price panel
(``trailing`` returns data <= t only).

Subset note: the FINSABER panel exposes adjusted close only, so the close-based
indicators are kept (10-EMA, 50/200-SMA, MACD family, RSI, Bollinger) and the
high/low/volume-based ones from the reference menu (ATR, VWMA) are omitted.

Scale conventions for the prompt payload:
- moving averages are reported as scale-free gaps: close / average - 1
  (positive = price above the average);
- the MACD family is normalised by the close so names remain comparable;
- rsi is raw (0-100, 70/30 overbought/oversold convention);
- boll_position is (close - boll) / (boll_ub - boll): +1 at the upper band,
  -1 at the lower band, 0 at the middle.

``stockstats`` is an optional dependency: when it is missing the block is
simply omitted from the context (the time-series momentum/reversal/lowvol
signals still carry the technical view), per the project fallback rule.
"""

from __future__ import annotations

import math
from typing import Optional

INDICATOR_IDS: tuple[str, ...] = (
    "close_10_ema_gap",
    "close_50_sma_gap",
    "close_200_sma_gap",
    "macd_pct",
    "macds_pct",
    "macdh_pct",
    "rsi",
    "boll_position",
)

#: stockstats columns behind the block (TradingAgents menu names).
_STOCKSTATS_COLUMNS = (
    "close_10_ema",
    "close_50_sma",
    "close_200_sma",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
)


def _wrap_closes(closes: list[float]):
    """Build a stockstats frame from a close series; None when unavailable."""
    try:
        import pandas as pd
        from stockstats import wrap
    except ImportError:
        return None
    return wrap(pd.DataFrame({"close": [float(c) for c in closes]}))


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 6)


class TechnicalIndicatorBlock:
    """Per-ticker point-in-time indicator readings for the technical prompt."""

    def __init__(self, panel, history_days: int = 260, min_history: int = 30):
        if min_history < 2 or history_days < min_history:
            raise ValueError("need history_days >= min_history >= 2")
        self._panel = panel
        self._history_days = history_days
        self._min_history = min_history

    def indicators_for(self, ticker: str, t) -> Optional[dict[str, Optional[float]]]:
        closes = self._panel.trailing(ticker, t, self._history_days)
        if len(closes) < self._min_history or any(c <= 0 for c in closes):
            return None
        frame = _wrap_closes(closes)
        if frame is None:
            return None
        last = {}
        for column in _STOCKSTATS_COLUMNS:
            try:
                last[column] = _finite(frame[column].iloc[-1])
            except Exception:
                last[column] = None
        close = float(closes[-1])
        n = len(closes)

        def gap(column: str, needed: int) -> Optional[float]:
            base = last.get(column)
            if base is None or base <= 0 or n < needed:
                return None
            return close / base - 1.0

        def pct(column: str) -> Optional[float]:
            value = last.get(column)
            return None if value is None else value / close

        boll, boll_ub = last.get("boll"), last.get("boll_ub")
        boll_position = None
        if boll is not None and boll_ub is not None and boll_ub - boll > 0:
            boll_position = (close - boll) / (boll_ub - boll)

        return {
            "close_10_ema_gap": _round(gap("close_10_ema", 10)),
            "close_50_sma_gap": _round(gap("close_50_sma", 50)),
            "close_200_sma_gap": _round(gap("close_200_sma", 200)),
            "macd_pct": _round(pct("macd")),
            "macds_pct": _round(pct("macds")),
            "macdh_pct": _round(pct("macdh")),
            "rsi": _round(last.get("rsi")),
            "boll_position": _round(boll_position),
        }
