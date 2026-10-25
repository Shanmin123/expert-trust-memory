"""External market proxy (SPY) for single-name runs.

For a single traded stock the equal-weight universe proxy degenerates (the proxy
is the stock itself, so the market-adjusted excess is identically zero). This
accessor serves an external market series -- SPY dividend-adjusted daily close --
so the market-state classifier and the attribution baseline have a genuine market
reference decoupled from the traded name. Prices are point-in-time: only dates on
or before the query date are used.
"""

from __future__ import annotations

import bisect
from datetime import date as Date
from pathlib import Path
from typing import Mapping, Optional, Sequence

from trading_agent.market import _to_date


class MarketProxy:
    """Causal accessor over an external market price series (date, adjusted_close)."""

    def __init__(self, dates: Sequence[Date], closes: Sequence[float]) -> None:
        pairs = sorted(zip((_to_date(d) for d in dates), (float(c) for c in closes)))
        self._dates = [d for d, _ in pairs]
        self._closes = [c for _, c in pairs]

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MarketProxy":
        import pandas as pd

        df = pd.read_parquet(path)
        return cls(df["date"].tolist(), df["adjusted_close"].tolist())

    def __len__(self) -> int:
        return len(self._dates)

    def _index_on_or_before(self, t: Date) -> int:
        return bisect.bisect_right(self._dates, _to_date(t)) - 1

    def daily_return(self, t) -> Optional[float]:
        """Return of the proxy on date ``t`` (close_t / close_{t-1} - 1), causal."""
        idx = self._index_on_or_before(t)
        if idx < 1:
            return None
        prev, cur = self._closes[idx - 1], self._closes[idx]
        return (cur / prev - 1.0) if prev else None

    def forward_return(self, t, horizon: int = 1) -> Optional[float]:
        """Return from the close on/before ``t`` to ``horizon`` proxy dates later."""
        idx = self._index_on_or_before(t)
        if idx < 0 or idx + horizon >= len(self._closes):
            return None
        p0, p1 = self._closes[idx], self._closes[idx + horizon]
        return (p1 / p0 - 1.0) if p0 else None

    def returns_by_date(self, dates: Sequence[Date]) -> dict[Date, float]:
        """Proxy daily returns aligned to the given trading dates (causal per date)."""
        out: dict[Date, float] = {}
        for d in dates:
            r = self.daily_return(d)
            if r is not None:
                out[_to_date(d)] = r
        return out

    def level_on(self, t) -> Optional[float]:
        """The proxy level on/before ``t`` (used directly as the vol level for VIX)."""
        idx = self._index_on_or_before(t)
        return self._closes[idx] if idx >= 0 else None

    def levels_by_date(self, dates: Sequence[Date]) -> dict[Date, float]:
        """Proxy levels aligned to the given trading dates -- the VIX regime series."""
        out: dict[Date, float] = {}
        for d in dates:
            v = self.level_on(d)
            if v is not None:
                out[_to_date(d)] = v
        return out
