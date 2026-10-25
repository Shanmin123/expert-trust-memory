"""Point-in-time price access shared by the alpha signal and the regime inferer.

A thin read-only accessor over a FINSABER ``TradingData`` window. All lookups are strictly
causal: ``trailing(ticker, t, n)`` returns only bars with date <= t, never future bars.

Performance: per-ticker series are cached on first use (one pass over the window), so trailing
lookups are O(log n) via bisect and point lookups are O(1) instead of rescanning all dates per
call. Behavior is identical to the naive scan.
"""

from __future__ import annotations

import bisect
from datetime import date as Date, datetime
from typing import Optional


def _to_date(value) -> Date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        import pandas as pd

        return pd.to_datetime(value).date()
    return value


class LoaderPricePanel:
    """Causal trailing-price accessor over a FINSABER data-loader window (cached)."""

    def __init__(self, window_loader, price_field: str = "adjusted_close"):
        self._loader = window_loader
        self._price_field = price_field
        self._dates = [_to_date(d) for d in window_loader.get_date_range()]
        self._built = False
        self._tdates: dict[str, list] = {}     # ticker -> sorted dates present
        self._tprices: dict[str, list] = {}    # ticker -> prices aligned to _tdates
        self._tbydate: dict[str, dict] = {}     # ticker -> {date: price}

    @property
    def dates(self) -> list:
        return list(self._dates)

    def dates_upto(self, t: Date) -> list:
        t = _to_date(t)
        idx = bisect.bisect_right(self._dates, t)
        return self._dates[:idx]

    def _ensure_cache(self) -> None:
        if self._built:
            return
        for d in self._dates:
            price_map = self._loader.get_data_by_date(d).get("price", {}) or {}
            for tk, bar in price_map.items():
                if isinstance(bar, dict):
                    p = bar.get(self._price_field, bar.get("close", bar.get("adjusted_close")))
                else:
                    p = float(bar)
                if p is None:
                    continue
                self._tdates.setdefault(tk, []).append(d)
                self._tprices.setdefault(tk, []).append(p)
                self._tbydate.setdefault(tk, {})[d] = p
        self._built = True

    def _price_on(self, ticker: str, d: Date) -> Optional[float]:
        self._ensure_cache()
        return self._tbydate.get(ticker, {}).get(_to_date(d))

    def trailing(self, ticker: str, t: Date, n: int) -> list:
        """Up to ``n`` most-recent prices with date <= t, ascending."""
        self._ensure_cache()
        t = _to_date(t)
        dates = self._tdates.get(ticker)
        if not dates:
            return []
        idx = bisect.bisect_right(dates, t)
        lo = max(0, idx - n)
        return self._tprices[ticker][lo:idx]

    def date_after(self, t: Date, horizon: int) -> Optional[Date]:
        """The trading date ``horizon`` bars after t, or None if beyond the window."""
        t = _to_date(t)
        idx = bisect.bisect_right(self._dates, t)
        fut = self._dates[idx:]
        if len(fut) < horizon:
            return None
        return fut[horizon - 1]

    def price_after(self, ticker: str, t: Date, horizon: int) -> Optional[float]:
        """Price ``horizon`` trading days AFTER t (for forward-outcome computation only).

        Used post-hoc by reflection/learning-curve/audit over realized history, never inside a
        decision. Returns None if t+horizon is beyond the loaded window.
        """
        d = self.date_after(t, horizon)
        return None if d is None else self._price_on(ticker, d)


class LoaderOHLCVPanel:
    """Point-in-time split-adjusted OHLCV frames over the same FINSABER window loader.

    ``LoaderPricePanel`` exposes a single close series; the raw FINSABER bars actually
    carry open/high/low/close/adjusted_close/volume, which the full technical-indicator
    menu (ADX, CCI, Williams %R, Awesome, UO, BBP, VWMA) needs. This taps the identical
    causal loader and returns a trailing OHLCV frame with bars <= t only. Open/high/low
    are rescaled by the split/dividend factor ``adjusted_close / close`` so the whole bar
    sits on the adjusted-close basis (indicators stay split-consistent); volume is left
    raw. No high/low means the row is dropped.
    """

    def __init__(self, window_loader):
        self._loader = window_loader
        self._dates = [_to_date(d) for d in window_loader.get_date_range()]
        self._built = False
        self._tdates: dict[str, list] = {}
        self._tbars: dict[str, dict] = {}    # ticker -> {date: adjusted ohlcv dict}

    def _ensure_cache(self) -> None:
        if self._built:
            return
        for d in self._dates:
            price_map = self._loader.get_data_by_date(d).get("price", {}) or {}
            for tk, bar in price_map.items():
                if bar is None:
                    continue
                close = bar.get("close")
                adj = bar.get("adjusted_close", close)
                o, h, l, v = (bar.get("open"), bar.get("high"),
                              bar.get("low"), bar.get("volume"))
                if not close or close <= 0 or adj is None or h is None or l is None:
                    continue
                f = float(adj) / float(close)   # split/dividend adjustment factor
                self._tbars.setdefault(tk, {})[d] = {
                    "open": (float(o) * f) if o is not None else float(adj),
                    "high": float(h) * f,
                    "low": float(l) * f,
                    "close": float(adj),
                    "volume": float(v) if v is not None else 0.0,
                }
                self._tdates.setdefault(tk, []).append(d)
        self._built = True

    def trailing_frame(self, ticker: str, t, n: int):
        """A pandas OHLCV DataFrame of the last ``n`` adjusted bars with date <= t."""
        self._ensure_cache()
        t = _to_date(t)
        dates = self._tdates.get(ticker)
        if not dates:
            return None
        idx = bisect.bisect_right(dates, t)
        window = dates[max(0, idx - n):idx]
        if not window:
            return None
        import pandas as pd
        rows = [self._tbars[ticker][d] for d in window]
        return pd.DataFrame(rows, columns=("open", "high", "low", "close", "volume"))
