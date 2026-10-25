"""Live paper context assembler: IB frames -> the backtest-identical context pipeline."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from trading_agent.cli import _paper_observation
from trading_agent.paper_live_context import IBFrameWindowLoader, build_paper_live_context


def _frame(n_days: int = 60, start: date = date(2026, 4, 1), base: float = 100.0) -> pd.DataFrame:
    rows = []
    d = start
    px = base
    for i in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        px *= 1.0 + (0.002 if i % 3 else -0.001)
        rows.append({
            "date": d.isoformat(), "open": px * 0.995, "high": px * 1.01,
            "low": px * 0.99, "close": px, "volume": 1_000_000,
        })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_loader_presents_ib_frames_through_the_finsaber_protocol():
    frames = {"BIIB": _frame(), "ECL": _frame(base=50.0)}
    loader = IBFrameWindowLoader(frames)
    dates = loader.get_date_range()
    assert len(dates) == 60 and dates == sorted(dates)
    day = loader.get_data_by_date(dates[-1])
    assert set(day["price"]) == {"BIIB", "ECL"}
    bar = day["price"]["BIIB"]
    assert bar["adjusted_close"] == bar["close"] > 0    # IB TRADES bars are split-adjusted
    assert {"open", "high", "low", "volume"} <= set(bar)


def test_loader_skips_rows_without_close():
    f = _frame(5)
    f.loc[2, "close"] = float("nan")
    loader = IBFrameWindowLoader({"X": f})
    assert len(loader.get_date_range()) == 4


def test_live_context_builder_produces_real_technical_context_and_no_news():
    frames = {"BIIB": _frame()}
    builder = build_paper_live_context(frames, tickers=["BIIB"])
    obs = _paper_observation(["BIIB"], frames)
    ctx = builder.build(obs, "up_calm")
    tech = ctx["technical"]["by_ticker"]["BIIB"]
    # The LOAD-BEARING block: the deterministic D3 technical vote reads `rating`
    # (needs only ~40 bars). The momentum/reversal/lowvol z-signals need ~262
    # trading days and are advisory prompt context only, so at 60 days they are
    # listed missing without blocking the vote.
    assert tech["rating"]["summary"] is not None
    assert "momentum" in tech.get("missing_signals", [])
    # No news adapter attached -> the news channel is unavailable and the news
    # expert abstains via its safe default (the backtests' fail-soft).
    assert ctx["news"]["available"] is False


def test_live_technical_expert_votes_from_the_rating_at_short_history():
    # End-to-end: with only ~60 live bars the technical expert still votes
    # deterministically from the rating block (long/flat/neutral), never a
    # safe default — the live committee's technical seat works at paper history.
    from trading_agent.contracts import is_safe_default_view
    from trading_agent.experts import TechnicalExpert

    frames = {"BIIB": _frame()}
    builder = build_paper_live_context(frames, tickers=["BIIB"])
    obs = _paper_observation(["BIIB"], frames)
    ctx = builder.build(obs, "up_calm")
    view = TechnicalExpert(backend=None).views(obs, "up_calm", context=ctx)["BIIB"]
    assert view.direction in ("long", "flat", "neutral")
    assert not is_safe_default_view(view)
    assert view.reasoning.startswith("deterministic TradingView")
