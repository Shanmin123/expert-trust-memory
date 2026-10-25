"""Point-in-time accessor for FINSABER ``filingk`` / ``filingq`` parquet (10-K / 10-Q).

FINSABER exposes raw filing text (median ~0.5M chars per filing), not structured
financials. This accessor reads the filing folders directly, keyed by ticker, and
exposes the most recent filing available at a decision date plus the prior filing
of the same form, which is what the fundamental expert's text signals need. Timing
is causal: a filing is usable only from its public ``date`` onward.

Two data quirks are handled here: the ``date`` column is stored as an ISO string,
and the partition ``year`` column has inconsistent integer widths across download
batches, so files are read individually (not as one unified dataset).
"""

from __future__ import annotations

import bisect
import glob
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from trading_agent.data.finsaber_panel import resolve_data_root
from trading_agent.data.finsaber_text import require_text_data
from trading_agent.market import _to_date

_FORM_FOLDER = {"10-K": "filingk", "10-Q": "filingq"}


@dataclass(frozen=True)
class FilingItem:
    date: Date
    ticker: str
    form: str  # "10-K" or "10-Q"
    text: str
    text_len: Optional[int] = None

    @property
    def length(self) -> int:
        if self.text_len is not None:
            try:
                return int(self.text_len)
            except (TypeError, ValueError):
                pass
        return len(self.text or "")


class FinsaberFilingPanel:
    """Causal accessor over FINSABER ``filingk`` / ``filingq`` rows."""

    def __init__(self, root: str | Path | None, start_date, end_date, tickers: Iterable[str] | None = None):
        require_text_data(
            str(root) if root is not None else None,
            min_modalities=("filingk", "filingq"),
        )
        self.root = resolve_data_root(str(root) if root is not None else None)
        self.start_date = _to_date(start_date)
        self.end_date = _to_date(end_date)
        self.tickers = sorted(set(tickers)) if tickers is not None else None
        self._built = False
        self._by_ticker: dict[str, list[FilingItem]] = {}
        self._dates_by_ticker: dict[str, list[Date]] = {}

    @classmethod
    def from_items(cls, items: Sequence[FilingItem]) -> "FinsaberFilingPanel":
        """Build a panel from in-memory items for tests."""
        obj = cls.__new__(cls)
        obj.root = Path("")
        dates = [_to_date(it.date) for it in items]
        obj.start_date = min(dates) if dates else None
        obj.end_date = max(dates) if dates else None
        obj.tickers = sorted({it.ticker for it in items})
        obj._built = True
        obj._by_ticker = {}
        for it in items:
            obj._by_ticker.setdefault(it.ticker, []).append(
                FilingItem(date=_to_date(it.date), ticker=it.ticker, form=it.form,
                           text=it.text, text_len=it.text_len)
            )
        obj._finalise()
        return obj

    def _finalise(self) -> None:
        self._dates_by_ticker = {}
        for ticker, items in self._by_ticker.items():
            items.sort(key=lambda x: (x.date, x.form))
            self._dates_by_ticker[ticker] = [it.date for it in items]

    def _load_from_parquet(self) -> None:
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - pyarrow present in the experiment env
            raise ImportError("pyarrow is required to read FINSABER filing parquet") from exc

        start = self.start_date.isoformat() if self.start_date else None
        end = self.end_date.isoformat() if self.end_date else None
        by_ticker: dict[str, list[FilingItem]] = {}
        for form, folder in _FORM_FOLDER.items():
            for file in sorted(glob.glob(str(self.root / folder / "year=*" / "*.parquet"))):
                dataset = ds.dataset(file, format="parquet")
                flt = None
                if self.tickers is not None:
                    flt = ds.field("symbol").isin(self.tickers)
                table = dataset.to_table(
                    columns=["date", "symbol", "filing_text", "text_len"], filter=flt
                )
                for row in table.to_pylist():
                    raw_date = str(row["date"])
                    if start is not None and raw_date < start:
                        continue
                    if end is not None and raw_date > end:
                        continue
                    text = row["filing_text"] or ""
                    if not text:
                        continue
                    item = FilingItem(
                        date=_to_date(raw_date),
                        ticker=str(row["symbol"]),
                        form=form,
                        text=text,
                        text_len=None if row["text_len"] is None else int(row["text_len"]),
                    )
                    by_ticker.setdefault(item.ticker, []).append(item)
        self._by_ticker = by_ticker
        self._finalise()
        self._built = True

    def _ensure_cache(self) -> None:
        if not self._built:
            self._load_from_parquet()

    def latest_on(self, ticker: str, t) -> Optional[FilingItem]:
        """Most recent filing (any form) available on or before ``t``."""
        self._ensure_cache()
        t = _to_date(t)
        dates = self._dates_by_ticker.get(ticker, [])
        if not dates:
            return None
        idx = bisect.bisect_right(dates, t) - 1
        if idx < 0:
            return None
        return self._by_ticker[ticker][idx]

    def latest_and_prior(self, ticker: str, t) -> tuple[Optional[FilingItem], Optional[FilingItem]]:
        """The most recent filing on/before ``t`` and the prior filing of the same form.

        The same-form pairing supports the Lazy-Prices textual-change signal:
        a 10-K is compared with the previous 10-K, a 10-Q with the previous 10-Q.
        """
        self._ensure_cache()
        t = _to_date(t)
        items = self._by_ticker.get(ticker, [])
        dates = self._dates_by_ticker.get(ticker, [])
        if not items:
            return None, None
        idx = bisect.bisect_right(dates, t) - 1
        if idx < 0:
            return None, None
        latest = items[idx]
        prior = None
        for j in range(idx - 1, -1, -1):
            if items[j].form == latest.form:
                prior = items[j]
                break
        return latest, prior

    def coverage(self, tickers: Sequence[str], t) -> float:
        """Fraction of tickers with at least one filing available on/before ``t``."""
        if not tickers:
            return 0.0
        return sum(1 for tk in tickers if self.latest_on(tk, t) is not None) / len(tickers)
