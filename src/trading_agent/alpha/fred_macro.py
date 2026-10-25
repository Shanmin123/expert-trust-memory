"""Point-in-time-guarded FRED macro features for the design-D news expert.

Reads the long-format parquet written by ``scripts/collect_fred_macro.py`` and exposes a
small, interpretable macro block (TradingAgents' news analyst grounds macro commentary in
exactly these FRED indicators). Readings at decision date t use only observations whose
date PLUS the series' publication lag is <= t:

- DFF / T10Y2Y: market data, published same day (lag 0);
- ICSA: released the Thursday after the reference week (lag 7 days);
- CPIAUCSL / UNRATE: released ~mid next month (lag 45 days, conservative).

Latest-revision series, not ALFRED vintages — a documented limitation (rates/curve are never
revised; the lag guard removes the availability side of the bias for the slow series).
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date as Date, timedelta
from pathlib import Path
from typing import Optional

#: publication lag (calendar days) added to each observation date before it becomes readable.
PUBLICATION_LAG_DAYS = {"DFF": 0, "T10Y2Y": 0, "ICSA": 7, "CPIAUCSL": 45, "UNRATE": 45}


class FredMacroAlpha:
    """Per-date macro feature block from the collected FRED parquet."""

    def __init__(self, path: str | Path) -> None:
        import pandas as pd

        self._available = False
        self._by_series: dict[str, tuple[list[Date], list[float]]] = {}
        p = Path(path)
        if not p.exists():
            return
        frame = pd.read_parquet(p)
        for sid, grp in frame.groupby("series"):
            grp = grp.sort_values("date")
            lag = timedelta(days=PUBLICATION_LAG_DAYS.get(str(sid), 45))
            dates = [d + lag for d in grp["date"].tolist()]
            self._by_series[str(sid)] = (dates, [float(v) for v in grp["value"].tolist()])
        self._available = bool(self._by_series)

    @property
    def available(self) -> bool:
        return self._available

    def _last(self, sid: str, t: Date, back: int = 0) -> Optional[float]:
        entry = self._by_series.get(sid)
        if not entry:
            return None
        dates, values = entry
        i = bisect_right(dates, t) - 1 - back
        return values[i] if i >= 0 else None

    def features_for(self, t: Date) -> dict:
        """Interpretable macro readings as of t (readable-by-t only)."""
        if not self._available:
            return {"available": False}
        ff = self._last("DFF", t)
        ff_63 = self._last("DFF", t - timedelta(days=91))
        curve = self._last("T10Y2Y", t)
        cpi = self._last("CPIAUCSL", t)
        cpi_12m = self._last("CPIAUCSL", t - timedelta(days=365))
        claims = self._last("ICSA", t)
        claims_prev = self._last("ICSA", t, back=4)
        unrate = self._last("UNRATE", t)

        def _round(v, nd=4):
            return None if v is None else round(float(v), nd)

        return {
            "available": True,
            "fed_funds_rate": _round(ff, 2),
            "fed_funds_change_3m": _round(ff - ff_63, 2) if (ff is not None and ff_63 is not None) else None,
            "yield_curve_10y_minus_2y": _round(curve, 2),
            "cpi_yoy_pct": _round((cpi / cpi_12m - 1.0) * 100.0, 2)
            if (cpi is not None and cpi_12m not in (None, 0.0)) else None,
            "initial_claims": _round(claims, 0),
            "initial_claims_change_4wk": _round(claims - claims_prev, 0)
            if (claims is not None and claims_prev is not None) else None,
            "unemployment_rate": _round(unrate, 1),
        }
