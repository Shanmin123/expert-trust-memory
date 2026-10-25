"""Point-in-time FINSABER text availability probes and causal text accessors.

The price experiments intentionally load only ``price_daily``. Richer-input experiments must opt in to
text and fail clearly when the configured FINSABER root has no ``news_items`` / filing folders, rather
than silently falling back to price-only data.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from trading_agent.data.finsaber_panel import resolve_data_root
from trading_agent.market import _to_date

_RAW_MODALITY_TO_FOLDER = {
    "news": "news_items",
    "news_items": "news_items",
    "filing_k": "filingk",
    "filingk": "filingk",
    "10-K": "filingk",
    "filing_q": "filingq",
    "filingq": "filingq",
    "10-Q": "filingq",
}
_FOLDER_TO_LOADER_MODALITY = {
    "news_items": "news",
    "filingk": "filing_k",
    "filingq": "filing_q",
}


def _count_parquet_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    year_parts = list(folder.glob("year=*/part-*.parquet"))
    top_level = list(folder.glob("*.parquet"))
    return len(year_parts) + len(top_level)


def _sqlite_index_candidates(root: Path) -> list[Path]:
    return [
        root / "news_text_index.sqlite",
        root.parent / "news_text_index.sqlite",
        root.parent.parent / "news_text_index.sqlite",
    ]


@dataclass(frozen=True)
class TextDataAvailability:
    root: str
    news_items_files: int
    filingk_files: int
    filingq_files: int
    sqlite_index_path: Optional[str] = None

    @property
    def has_news(self) -> bool:
        return self.news_items_files > 0

    @property
    def has_filingk(self) -> bool:
        return self.filingk_files > 0

    @property
    def has_filingq(self) -> bool:
        return self.filingq_files > 0

    @property
    def present_folders(self) -> tuple[str, ...]:
        out = []
        if self.has_news:
            out.append("news_items")
        if self.has_filingk:
            out.append("filingk")
        if self.has_filingq:
            out.append("filingq")
        return tuple(out)

    def has_folder(self, folder_or_modality: str) -> bool:
        folder = _RAW_MODALITY_TO_FOLDER.get(folder_or_modality, folder_or_modality)
        return {
            "news_items": self.has_news,
            "filingk": self.has_filingk,
            "filingq": self.has_filingq,
        }.get(folder, False)

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        return tuple(m for m in required if not self.has_folder(m))

    def loader_modalities(self, include_price: bool = True) -> tuple[str, ...]:
        mods = ["price"] if include_price else []
        for folder in self.present_folders:
            mods.append(_FOLDER_TO_LOADER_MODALITY[folder])
        return tuple(mods)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "news_items_files": self.news_items_files,
            "filingk_files": self.filingk_files,
            "filingq_files": self.filingq_files,
            "sqlite_index_path": self.sqlite_index_path,
            "present_folders": list(self.present_folders),
        }


def probe_text_data(root: Optional[str] = None) -> TextDataAvailability:
    """Return a cheap local text-data inventory for a FINSABER parquet root.

    This deliberately checks folder/file presence only. It does not materialize the large news/filing
    parquet payloads.
    """
    p = resolve_data_root(root)
    sqlite_index = next((c for c in _sqlite_index_candidates(p) if c.exists()), None)
    return TextDataAvailability(
        root=str(p),
        news_items_files=_count_parquet_files(p / "news_items"),
        filingk_files=_count_parquet_files(p / "filingk"),
        filingq_files=_count_parquet_files(p / "filingq"),
        sqlite_index_path=str(sqlite_index) if sqlite_index else None,
    )


def require_text_data(root: Optional[str] = None, min_modalities: Sequence[str] = ("news_items",)) -> TextDataAvailability:
    """Probe text data and raise a clear blocker if required text folders are absent."""
    avail = probe_text_data(root)
    missing = avail.missing(min_modalities)
    if missing:
        present = ", ".join(avail.present_folders) or "none"
        raise RuntimeError(
            "FINSABER text data is required for the richer-input experiment but is missing. "
            f"root={avail.root}; missing={list(missing)}; present={present}. "
            "Download/copy the FINSABER text parquet folders (news_items/ and optionally filingk/, "
            "filingq/) to this root, or point FINSABER_DATA_ROOT at a full FINSABER-V2-Data tree."
        )
    return avail


@dataclass(frozen=True)
class TextItem:
    date: Date
    ticker: str
    modality: str
    text: str


class LoaderTextPanel:
    """Causal trailing-text accessor over a FINSABER data-loader window.

    The accessor is keyed by the loader's trading-date range. ``trailing_items(..., lookback=N)``
    returns only text attached to the last N trading dates with date <= t. If raw weekend/holiday text
    is not pre-aligned upstream, it will not appear here; richer experiments report coverage so this
    limitation is visible.
    """

    def __init__(self, window_loader):
        self._loader = window_loader
        self._dates = [_to_date(d) for d in window_loader.get_date_range()]
        self._built = False
        self._items_by_ticker: dict[str, list[TextItem]] = {}
        self._dates_by_ticker: dict[str, list[Date]] = {}

    @property
    def dates(self) -> list[Date]:
        return list(self._dates)

    def _append(self, d: Date, ticker: str, modality: str, value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            texts = [value]
        else:
            try:
                texts = [str(x) for x in value if x is not None]
            except TypeError:
                texts = [str(value)]
        for text in texts:
            if not text:
                continue
            self._items_by_ticker.setdefault(ticker, []).append(TextItem(d, ticker, modality, text))

    def _ensure_cache(self) -> None:
        if self._built:
            return
        for d in self._dates:
            daily = self._loader.get_data_by_date(d) or {}
            for modality in ("news", "filing_k", "filing_q"):
                values = daily.get(modality, {}) or {}
                if not isinstance(values, dict):
                    continue
                for ticker, payload in values.items():
                    self._append(d, ticker, modality, payload)
        for ticker, items in self._items_by_ticker.items():
            items.sort(key=lambda it: (it.date, it.modality, it.text[:32]))
            self._dates_by_ticker[ticker] = [it.date for it in items]
        self._built = True

    def trailing_items(
        self,
        ticker: str,
        t: Date,
        lookback: int,
        modalities: Sequence[str] = ("news",),
    ) -> list[TextItem]:
        """Return text for ``ticker`` over the last ``lookback`` trading dates with date <= t."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._ensure_cache()
        t = _to_date(t)
        if not self._dates:
            return []
        date_hi = bisect.bisect_right(self._dates, t)
        if date_hi == 0:
            return []
        date_lo = max(0, date_hi - lookback)
        start = self._dates[date_lo]
        items = self._items_by_ticker.get(ticker, [])
        item_dates = self._dates_by_ticker.get(ticker, [])
        lo = bisect.bisect_left(item_dates, start)
        hi = bisect.bisect_right(item_dates, t)
        wanted = set(modalities)
        return [it for it in items[lo:hi] if it.modality in wanted]

    def count_items(
        self,
        ticker: str,
        t: Date,
        lookback: int,
        modalities: Sequence[str] = ("news",),
    ) -> int:
        return len(self.trailing_items(ticker, t, lookback, modalities))

    def coverage(
        self,
        tickers: Sequence[str],
        t: Date,
        lookback: int,
        modalities: Sequence[str] = ("news",),
    ) -> tuple[float, int]:
        """Return (fraction of tickers with at least one item, total item count)."""
        if not tickers:
            return 0.0, 0
        counts = [self.count_items(tk, t, lookback, modalities) for tk in tickers]
        return sum(1 for c in counts if c > 0) / len(tickers), sum(counts)
