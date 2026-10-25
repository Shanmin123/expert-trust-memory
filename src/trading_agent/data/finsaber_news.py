"""Project-owned point-in-time accessor for FINSABER ``news_items`` parquet.

This module deliberately does NOT edit or depend on FINSABER's vendored parquet loader for text
metadata. The vendored loader collapses each news row to ``news_text`` for strategy compatibility;
news extractors need the original metadata (``text_len``, ``text_crc32``, ``item_index``, ``cik``), so
we read the ``news_items`` split directly here.
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from trading_agent.data.finsaber_panel import resolve_data_root
from trading_agent.data.finsaber_text import require_text_data
from trading_agent.market import _to_date


@dataclass(frozen=True)
class NewsItem:
    date: Date
    ticker: str
    text: str
    cik: Optional[str] = None
    item_index: Optional[int] = None
    text_len: Optional[int] = None
    text_crc32: Optional[int | str] = None

    @property
    def length(self) -> int:
        if self.text_len is not None:
            try:
                return int(self.text_len)
            except (TypeError, ValueError):
                pass
        return len(self.text or "")

    @property
    def dedup_key(self) -> str:
        if self.text_crc32 is not None and str(self.text_crc32) != "":
            return f"crc32:{self.text_crc32}"
        return "sha1:" + hashlib.sha1((self.text or "").encode("utf-8", errors="ignore")).hexdigest()


class FinsaberNewsPanel:
    """Causal accessor over FINSABER ``news_items`` rows.

    ``items_on(ticker, T)`` returns news attached to T. ``trailing_items(..., include_current=False)``
    returns strictly prior trading dates by default, which is the safe setting for rolling baselines and
    novelty history. T-day news may form a T-day signal, but experiments must treat that signal as
    tradable only from T+1 onward when the raw data has no intraday timestamp.
    """

    def __init__(self, root: str | Path | None, start_date, end_date, tickers: Iterable[str] | None = None):
        require_text_data(str(root) if root is not None else None, min_modalities=("news_items",))
        self.root = resolve_data_root(str(root) if root is not None else None)
        self.start_date = _to_date(start_date)
        self.end_date = _to_date(end_date)
        self.tickers = sorted(set(tickers)) if tickers is not None else None
        self._built = False
        self._items_by_ticker_date: dict[str, dict[Date, list[NewsItem]]] = {}
        self._dates_by_ticker: dict[str, list[Date]] = {}

    @classmethod
    def from_items(cls, items: Sequence[NewsItem]) -> "FinsaberNewsPanel":
        """Build a panel from in-memory items for tests."""
        obj = cls.__new__(cls)
        obj.root = Path("")
        dates = [_to_date(it.date) for it in items]
        obj.start_date = min(dates) if dates else None
        obj.end_date = max(dates) if dates else None
        obj.tickers = sorted({it.ticker for it in items})
        obj._built = True
        obj._items_by_ticker_date = {}
        obj._dates_by_ticker = {}
        for it in items:
            d = _to_date(it.date)
            normalized = NewsItem(
                date=d,
                ticker=it.ticker,
                text=it.text,
                cik=it.cik,
                item_index=it.item_index,
                text_len=it.text_len,
                text_crc32=it.text_crc32,
            )
            obj._items_by_ticker_date.setdefault(it.ticker, {}).setdefault(d, []).append(normalized)
        for ticker, by_date in obj._items_by_ticker_date.items():
            obj._dates_by_ticker[ticker] = sorted(by_date)
            for d in by_date:
                by_date[d].sort(key=lambda x: (x.item_index is None, x.item_index or 0, x.dedup_key))
        return obj

    def _load_from_parquet(self) -> None:
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - pyarrow is present in the experiment env
            raise ImportError("pyarrow is required to read FINSABER news_items parquet") from exc

        dataset = ds.dataset(str(self.root / "news_items"), format="parquet", partitioning="hive")
        columns = ["date", "symbol", "cik", "item_index", "news_text", "text_len", "text_crc32", "year"]
        filters = [ds.field("date") >= self.start_date, ds.field("date") <= self.end_date]
        if self.tickers is not None:
            filters.append(ds.field("symbol").isin(self.tickers))
        expr = filters[0]
        for f in filters[1:]:
            expr = expr & f
        table = dataset.to_table(columns=columns, filter=expr)
        df = table.to_pandas()
        if df.empty:
            self._items_by_ticker_date = {}
            self._dates_by_ticker = {}
            self._built = True
            return
        df["date"] = df["date"].apply(_to_date)
        self._items_by_ticker_date = {}
        for row in df.itertuples(index=False):
            text = row.news_text or ""
            if not text:
                continue
            item = NewsItem(
                date=row.date,
                ticker=row.symbol,
                text=text,
                cik=None if row.cik is None else str(row.cik),
                item_index=None if row.item_index is None else int(row.item_index),
                text_len=None if row.text_len is None else int(row.text_len),
                text_crc32=row.text_crc32,
            )
            self._items_by_ticker_date.setdefault(item.ticker, {}).setdefault(item.date, []).append(item)
        self._dates_by_ticker = {}
        for ticker, by_date in self._items_by_ticker_date.items():
            self._dates_by_ticker[ticker] = sorted(by_date)
            for d in by_date:
                by_date[d].sort(key=lambda x: (x.item_index is None, x.item_index or 0, x.dedup_key))
        self._built = True

    def _ensure_cache(self) -> None:
        if not self._built:
            self._load_from_parquet()

    @property
    def dates_by_ticker(self) -> dict[str, list[Date]]:
        self._ensure_cache()
        return {ticker: list(dates) for ticker, dates in self._dates_by_ticker.items()}

    def all_items(self, tickers: Sequence[str] | None = None) -> list[NewsItem]:
        """Return all cached items, optionally restricted to tickers."""
        self._ensure_cache()
        wanted = set(tickers) if tickers is not None else None
        out: list[NewsItem] = []
        for ticker, by_date in self._items_by_ticker_date.items():
            if wanted is not None and ticker not in wanted:
                continue
            for d in self._dates_by_ticker.get(ticker, sorted(by_date)):
                out.extend(by_date.get(d, []))
        return out

    def items_on(self, ticker: str, t) -> list[NewsItem]:
        self._ensure_cache()
        return list(self._items_by_ticker_date.get(ticker, {}).get(_to_date(t), []))

    def trailing_items(self, ticker: str, t, lookback: int, include_current: bool = False) -> list[NewsItem]:
        """Return ticker news from the last ``lookback`` trading dates before (or through) ``t``."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._ensure_cache()
        t = _to_date(t)
        dates = self._dates_by_ticker.get(ticker, [])
        if not dates:
            return []
        hi = bisect.bisect_right(dates, t) if include_current else bisect.bisect_left(dates, t)
        lo = max(0, hi - lookback)
        out: list[NewsItem] = []
        by_date = self._items_by_ticker_date.get(ticker, {})
        for d in dates[lo:hi]:
            out.extend(by_date.get(d, []))
        return out

    @staticmethod
    def dedup_items(items: Sequence[NewsItem]) -> list[NewsItem]:
        seen = set()
        out = []
        for item in items:
            key = item.dedup_key
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def coverage(self, tickers: Sequence[str], t) -> tuple[float, int, int]:
        """Return (fraction with >=1 raw item, raw item count, dedup item count) for date t."""
        if not tickers:
            return 0.0, 0, 0
        raw_counts = []
        dedup_counts = []
        for ticker in tickers:
            items = self.items_on(ticker, t)
            raw_counts.append(len(items))
            dedup_counts.append(len(self.dedup_items(items)))
        return sum(1 for c in raw_counts if c > 0) / len(tickers), sum(raw_counts), sum(dedup_counts)
