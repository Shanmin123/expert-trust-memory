"""Live expert context for IB paper mode.

Adapts the IB daily-bar frames (``request_daily_bars`` output) into the project's
FINSABER window-loader protocol, then assembles the SAME price panel, technical
rating, and signal sources the backtests use — so a live paper decision is made by
the identical context pipeline, just fed from IB instead of the FINSABER archive.

Thin adapter by design: no signal logic lives here.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date as Date
from typing import Mapping, Optional, Sequence

from trading_agent.alpha.technical_rating import TradingViewRating
from trading_agent.expert_context import (
    ExpertSignalContextBuilder,
    ExpertSignalSources,
    build_expert_signal_sources,
)
from trading_agent.market import LoaderOHLCVPanel, LoaderPricePanel, _to_date
from trading_agent.memory.expert_scorecard import ExpertScorecardStore


class IBFrameWindowLoader:
    """Present ``{symbol: ib daily-bar frame}`` through the FINSABER loader protocol.

    IB TRADES daily bars are split-adjusted by IB, so ``adjusted_close`` maps to
    ``close`` (dividend adjustment is not applied; over the trailing ~1.5y context
    window this matches how the live broker quotes the instrument). Rows without a
    close are dropped.
    """

    def __init__(self, frames: Mapping[str, object]) -> None:
        self._by_date: dict[Date, dict[str, dict]] = {}
        for symbol, frame in frames.items():
            if frame is None or getattr(frame, "empty", True):
                continue
            for row in frame.to_dict("records"):
                close = row.get("close")
                if close is None or close != close:  # None or NaN
                    continue
                d = _to_date(row.get("date"))
                bar = {
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": float(close),
                    "adjusted_close": float(close),
                    "volume": row.get("volume"),
                }
                self._by_date.setdefault(d, {})[symbol] = bar
        self._dates = sorted(self._by_date)

    def get_date_range(self) -> list[Date]:
        return list(self._dates)

    def get_data_by_date(self, d) -> dict:
        return {"price": self._by_date.get(_to_date(d), {})}


def build_paper_live_context(
    frames: Mapping[str, object],
    *,
    store: Optional[ExpertScorecardStore] = None,
    sec_facts_path: Optional[str] = None,
    fred_macro_path: Optional[str] = None,
    news_panel: object | None = None,
    news_sentiment_alpha: object | None = None,
    tickers: Optional[Sequence[str]] = None,
) -> ExpertSignalContextBuilder:
    """Assemble the backtest-identical context builder over live IB frames.

    Technical: TradingView rating + momentum/reversal/lowvol engines over the IB
    panel. Fundamental: SEC XBRL point-in-time reader when ``sec_facts_path`` is
    given. News: attached only when an adapter supplies ``news_panel`` /
    ``news_sentiment_alpha`` (otherwise the news expert sees no signals and
    abstains via its safe default — the fail-soft the backtests use).
    """
    loader = IBFrameWindowLoader(frames)
    price_panel = LoaderPricePanel(loader)
    ohlcv_panel = LoaderOHLCVPanel(loader)
    rating = TradingViewRating(ohlcv_panel)
    sources: ExpertSignalSources = build_expert_signal_sources(
        tickers=tickers,
        sec_facts_path=sec_facts_path,
        fred_macro_path=fred_macro_path,
        technical_rating=rating,
        include_news=False,          # IB news arrives via the explicit adapter below
        include_filings=sec_facts_path is not None,
    )
    if news_panel is not None or news_sentiment_alpha is not None:
        sources = replace(
            sources,
            news_panel=news_panel if news_panel is not None else sources.news_panel,
            news_sentiment_alpha=(
                news_sentiment_alpha
                if news_sentiment_alpha is not None
                else sources.news_sentiment_alpha
            ),
        )
    return ExpertSignalContextBuilder(
        panel=price_panel,
        store=store if store is not None else ExpertScorecardStore(),
        sources=sources,
    )
