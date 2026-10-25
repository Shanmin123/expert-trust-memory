"""Live committee read on the twenty evaluated stocks. Analyses only; it cannot trade.

Pulls daily bars from IB, labels the market state from live SPY, runs the SAME three
experts and the SAME frozen memory the backtest uses, and prints the committee's view
per stock. No order is built, previewed or sent: this module imports no execution path.

    PYTHONPATH=src python scripts/live_analysis.py
    PYTHONPATH=src python scripts/live_analysis.py --symbols BIIB,ECL --no-llm
    PYTHONPATH=src python scripts/live_analysis.py --offline reports/paper_forward/bars

The three stages are separate so the middle one can be tested without a gateway:

    load_market_data()  IB or a saved snapshot -> MarketData
    analyse()           MarketData -> [SymbolRead], no printing, no I/O
    render()            [SymbolRead] -> the terminal report

``--offline`` reads bars written by ``--save-bars`` on an earlier live run, which is how
a session is rehearsed when the gateway is down and how the tests drive the whole path.

Two facts the output states rather than hides. The technical expert is deterministic
(a TradingView-style rating over the IB bars, no model call). The news and fundamental
experts are model calls; with --no-llm they abstain, which is the same fail-soft the
backtest uses when a source is missing, and the committee then runs on whoever spoke.

IB duration note: "<n> D" beyond about a year times out and returns an empty frame
silently, which would make the classifier fall back to the traded stock's own bars.
This script requests "3 Y" (~751 rows) so the 504-day percentile window is filled.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import time as Time
from datetime import timedelta as TimeDelta
from pathlib import Path
from typing import Mapping, Sequence

from trading_agent.aggregation.soft_weighting import aggregate
from trading_agent.alpha.market_state_classifier import MarketStateClassifier, MarketStateConfig
from trading_agent.contracts import Bar, Observation, is_safe_default_view
from trading_agent.experts.backends import build_experts
from trading_agent.market import LoaderPricePanel
from trading_agent.memory.expert_scorecard import ExpertScorecardStore
from trading_agent.paper_live_context import IBFrameWindowLoader, build_paper_live_context

ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "frozen_memory"
def macro_path() -> Path | None:
    """The FRED macro block the news seat reads, wherever it actually lives.

    Same failure as the SEC facts: the repository-local path does not exist, the data
    sits under TRADING_AGENT_DATA_ROOT, and the builder was handed ``None`` in silence.
    """
    from trading_agent.paths import data_root

    for place in (Path(str(data_root())) / "macro", ROOT / "data" / "macro"):
        if place.exists() and any(place.glob("*.parquet")):
            return place
    return None


def sec_facts_path() -> Path:
    """The SEC facts the fundamental seat reads, merged across the collection passes.

    This used to point at ``data/sec_xbrl/financials.parquet`` under the repository,
    which does not exist: the facts live under TRADING_AGENT_DATA_ROOT and are split across three
    files, because the eight sealed names, the eleven added with the draw and one
    later name were collected in separate passes. The builder was handed ``None`` and
    said nothing, so on 2026-08-03 the fundamental seat abstained on all twenty stocks
    with "No financial ratios or fundamental data available" and the live committee
    silently became one seat.

    Merged here into one file the builder can read, refreshed when a source is newer.
    """
    import pandas as pd
    from trading_agent.paths import data_root

    base = Path(str(data_root())) / "sec_xbrl"
    sources = [p for p in sorted(base.glob("financials*.parquet"))
               if "pre_fix" not in p.name]
    if not sources:
        raise SystemExit(
            f"no SEC facts under {base}. The fundamental expert would abstain on every "
            "stock and the committee would run on the technical seat alone."
        )
    merged = base / "financials_merged_live.parquet"
    newest = max(p.stat().st_mtime for p in sources)
    if not merged.exists() or merged.stat().st_mtime < newest:
        frame = pd.concat([pd.read_parquet(p) for p in sources]).drop_duplicates()
        frame.to_parquet(merged, index=False)
    return merged

# The retrieval temperature selected on the validation window. The snapshot's own
# ``temperature`` field records the TRAINING store, not a retrieval choice, so it is
# overridden here (see FINAL_DESIGN, "a reading trap in the sealed artifacts").
TAU = 0.5
DURATION = "3 Y"
PROXY = "SPY"
ORDER = ("fundamental", "news", "technical")

# Every stock of the twenty-name draw, each on the arm the thesis reports for it:
# ADSK's one verified cell sits in the state memory, the rest read the global card.
# The twelve with no verified cell are here on purpose — they should reproduce their
# equal-weight committee exactly, and the fallback holding live is part of the check.
ARM_BY_SYMBOL = {"ADSK": "state", "AES": "global", "AFL": "global", "AVY": "global",
                 "AZO": "global", "BBY": "global", "BIIB": "global", "C": "global",
                 "CTAS": "global", "CVX": "global", "DDS": "global", "DHI": "global",
                 "ECL": "global", "INCY": "global", "MTCH": "global", "O": "global",
                 "PNC": "global", "RL": "global", "TEX": "global", "TGT": "global"}
DEFAULT_SYMBOLS = tuple(ARM_BY_SYMBOL)

FROZEN = MarketStateConfig(vol_window=21, percentile_window=504,
                           trend_window=200, trend_confirm_days=5)
ACCOUNT_TAGS = ("NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower",
                "GrossPositionValue", "UnrealizedPnL", "RealizedPnL")
# The news seat's attention and novelty features are defined against the stock's OWN
# news history, so the pull has to be deep enough to establish what a normal day looks
# like. At 30 days / 60 headlines it was not: BIIB came back with headlines on six
# distinct days, and the seat abstained on 19 of 20 stocks on 2026-08-03 writing "no
# recent news coverage", against 64-77% abstention on the same seat in the backtest,
# which reads a fifteen-year panel. The binding constraint is IB's 300-result cap on
# reqHistoricalNews, not the window: 365 and 1095 days return identical payloads. At
# 300 the same three stocks come back with 50, 77 and 66 distinct news days.
NEWS_DAYS = 365
NEWS_LIMIT = 300
NEWS_ATTEMPTS, NEWS_RETRY_SECONDS = 3, 2.0
W = 74


def arm_for(symbol: str) -> str:
    """Which memory a stock deploys. Asking for a stock by name must not change it.

    ``--symbols ADSK`` used to fall through to the global card while the default plan
    read ADSK's state card, so the same stock gave two different weights depending on
    how it was requested.
    """
    return ARM_BY_SYMBOL.get(symbol.upper(), "global")


# --------------------------------------------------------------------------- data


@dataclass
class MarketData:
    """Everything a read needs, however it was obtained."""

    frames: dict            # symbol -> daily-bar frame, including the SPY proxy
    account: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    news_panel: object | None = None
    news_counts: dict = field(default_factory=dict)
    source: str = "ib"


class IBNewsPanel:
    """Present IB headlines through the panel protocol the news expert already reads.

    The backtest reads FINSABER news through ``trailing_items``; this returns the same
    NewsItem shape over headlines pulled from IB, so the expert cannot tell the two
    apart. Headlines only: the article body needs a separate paid request per item.
    """

    def __init__(self, by_ticker: Mapping[str, list]) -> None:
        self._by_ticker = {k: sorted(v, key=lambda i: i.date) for k, v in by_ticker.items()}

    def items_on(self, ticker: str, t):
        """Items dated exactly on ``t``. The context reads this to decide availability."""
        return [i for i in self._by_ticker.get(ticker, []) if i.date == t]

    def coverage(self, tickers, t) -> tuple[float, int, int]:
        """(fraction of tickers with an item on t, raw item count, dedup item count).

        Same contract as the FINSABER panel's. It was missing until 2026-08-06, and its
        absence was not a partial failure: `build_expert_signal_sources` calls it on the
        FIRST line of the block that also attaches the headline snippets, so the
        AttributeError skipped that block entirely. The news seat was handed
        `available=False`, coverage zero and no headlines on every live day, and
        abstained on nineteen of twenty stocks while three hundred headlines a name sat
        unused in the panel. The reasonings said there was no recent coverage, which was
        true of what the seat was shown and false of what had been fetched.
        """
        tickers = list(tickers)
        if not tickers:
            return 0.0, 0, 0
        raw, dedup = [], []
        for ticker in tickers:
            items = self.items_on(ticker, t)
            raw.append(len(items))
            dedup.append(len(self.dedup_items(items)))
        return sum(1 for c in raw if c > 0) / len(tickers), sum(raw), sum(dedup)

    @staticmethod
    def dedup_items(items):
        """Drop repeats of the same headline text, keeping the first occurrence."""
        seen, out = set(), []
        for item in items:
            key = " ".join((getattr(item, "text", "") or "").split()).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def trailing_items(self, ticker: str, t, lookback: int, include_current: bool = False):
        items = self._by_ticker.get(ticker, [])
        cutoff = t if include_current else None
        out = []
        for item in items:
            if cutoff is not None and item.date > cutoff:
                continue
            if cutoff is None and item.date >= t:
                continue
            out.append(item)
        return out[-max(1, lookback * 4):]


def fetch_news(ib, symbols) -> tuple[IBNewsPanel | None, dict[str, int]]:
    """Pull recent headlines per symbol. Failure is not fatal: the expert abstains."""
    from data_collection.ib_news import fetch_headlines_payload, list_providers_payload
    from trading_agent.data.finsaber_news import NewsItem

    # Every provider the account actually holds, not the three module defaults. Those
    # three (BRFG, BRFUPDN, DJNL) are market columns, analyst-action feeds and
    # newsletters; the per-stock headlines come from DJ-N, so querying only the
    # defaults returns nothing and the news expert would abstain for the wrong reason.
    codes = [r["code"] for r in list_providers_payload(ib)["providers"] if r.get("code")]
    start = (DateTime.now() - TimeDelta(days=NEWS_DAYS)).strftime("%Y%m%d %H:%M:%S")
    end = DateTime.now().strftime("%Y%m%d %H:%M:%S")
    by_ticker, counts = {}, {}
    for sym in symbols:
        # The deep pull times out now and then: on 2026-08-04 AZO, BBY and PNC each
        # failed once and each returned 170 to 300 headlines on a retry seconds later.
        # Accepting the first failure would have silenced the news seat on three of
        # twenty stocks for no reason at all.
        payload, why = None, ""
        for attempt in range(NEWS_ATTEMPTS):
            try:
                payload = fetch_headlines_payload(
                    ib, symbol=sym, provider_codes=codes,
                    start=start, end=end, limit=NEWS_LIMIT,
                )
            except Exception as exc:
                payload, why = None, f"{type(exc).__name__}: {str(exc)[:44]}"
            else:
                if not payload.get("request_failed"):
                    break
                payload, why = None, "request timed out"
            if attempt + 1 < NEWS_ATTEMPTS:
                ib.sleep(NEWS_RETRY_SECONDS)
        if payload is None:
            counts[sym] = -1
            print(f"    {sym:<6} news unavailable after {NEWS_ATTEMPTS} attempts: {why}")
            continue
        items = []
        for row in payload.get("headlines", []) or payload.get("items", []):
            when, text = row.get("time"), (row.get("headline") or "").strip()
            if not when or not text:
                continue
            items.append(NewsItem(date=Date.fromisoformat(str(when)[:10]),
                                  ticker=sym, text=text))
        by_ticker[sym] = items
        counts[sym] = len(items)
        newest = max((i.date for i in items), default=None)
        print(f"    {sym:<6} {len(items):3d} headlines"
              + (f", newest {newest}" if newest else ""))
    if not any(v > 0 for v in counts.values()):
        return None, counts
    return IBNewsPanel(by_ticker), counts


def sec_coverage_gap(path: Path, symbols) -> list:
    """Symbols with no SEC filing at all, which would silently mute the third seat."""
    import pandas as pd

    have = set(pd.read_parquet(path, columns=["symbol"])["symbol"].unique())
    return sorted(s for s in symbols if s not in have)


def drop_unfinished_session(frame):
    """Remove a bar for a session that has not closed yet.

    ``request_daily_bars`` asks IB with ``endDateTime=""``, so during regular hours the
    last daily bar is TODAY'S PARTIAL bar: on 2026-08-03 at 09:50 ET, ADSK came back
    with volume 30,155 against the previous session's 782,233. The protocol decides on a
    completed close and fills at the next open, so using that bar would read an intraday
    price as a close, and the market-state windows would take it too.

    Dropping it is not a workaround, it is the protocol: the newest usable input is the
    last completed session. Returns (frame, dropped_date_or_None).
    """
    from datetime import timedelta, timezone

    if frame is None or getattr(frame, "empty", True):
        return frame, None
    eastern = timezone(timedelta(hours=-4))  # EDT; the forward window is in August
    now = DateTime.now(eastern)
    if now.weekday() >= 5:
        return frame, None
    minutes = now.hour * 60 + now.minute
    if not (9 * 60 + 30 <= minutes < 16 * 60):
        return frame, None                     # before the open or after the close
    today = now.date().isoformat()
    last = str(frame["date"].iloc[-1])[:10]
    if last != today:
        return frame, None
    return frame.iloc[:-1].copy(), last


def load_market_data(symbols, *, host, port, client_id, want_news: bool) -> MarketData:
    """One IB connection: account summary, open positions, daily bars, headlines."""
    from data_collection.config import Settings
    from data_collection.ib_connector import connect_ib, request_daily_bars

    settings = Settings(ib_host=host, ib_port=port, ib_client_id=client_id,
                        data_dir=ROOT / "data", reports_dir=ROOT / "reports",
                        rapidapi_key=None, rapidapi_host="x")
    ib = connect_ib(settings)
    frames, account, positions = {}, {}, []
    news_panel, news_counts = None, {}
    try:
        for row in ib.accountSummary():
            if row.tag in ACCOUNT_TAGS and row.currency in ("USD", ""):
                account[row.tag] = row.value
        for p in ib.positions():
            positions.append({
                "symbol": getattr(p.contract, "symbol", "?"),
                "quantity": float(p.position),
                "avg_cost": float(p.avgCost),
            })
        dropped = {}
        for sym in list(symbols) + [PROXY]:
            frame = request_daily_bars(ib, sym, duration=DURATION)
            frame, cut = drop_unfinished_session(frame)
            if cut:
                dropped[sym] = cut
            if frame is None or getattr(frame, "empty", True):
                raise SystemExit(
                    f"IB returned no bars for {sym} at duration {DURATION!r}. "
                    "Refusing to continue: an empty proxy frame would silently make the "
                    "market state self-referential."
                )
            frames[sym] = frame
        if dropped:
            day = sorted(set(dropped.values()))[0]
            print(f"  DROPPED the unfinished {day} bar on {len(dropped)} symbol(s): the "
                  "session is still open, so it is not a close. Deciding on the last "
                  "completed session, which is what the protocol specifies.")
        if want_news:
            print("  IB news")
            news_panel, news_counts = fetch_news(ib, symbols)
    finally:
        ib.disconnect()
    return MarketData(frames=frames, account=account, positions=positions,
                      news_panel=news_panel, news_counts=news_counts, source="ib")


def save_bars(data: MarketData, out_dir: Path) -> None:
    """Write the bars and headlines so a later run can rehearse without a gateway.

    Headlines were not saved until 2026-08-06, so when the news seat turned out to have
    been shown nothing on the first sessions, the inputs it should have had were no
    longer on disk and could only be pulled again from the broker. IB keeps a year, so
    nothing was lost this time; the next gap of this kind would not be so forgiving.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, frame in data.frames.items():
        frame.to_parquet(out_dir / f"{symbol}.parquet", index=False)
    (out_dir / "account.json").write_text(
        json.dumps({"account": data.account, "positions": data.positions}, indent=1),
        encoding="utf-8")
    panel = getattr(data, "news_panel", None)
    if panel is None:
        return
    by_ticker = getattr(panel, "_by_ticker", None)
    if not isinstance(by_ticker, dict):
        return
    dump = {
        ticker: [{"date": item.date.isoformat(),
                  "headline": getattr(item, "headline", ""),
                  "text": getattr(item, "text", "")}
                 for item in items]
        for ticker, items in by_ticker.items()
    }
    (out_dir / "news.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")


def _trade_date(decision_date) -> str:
    """The session the read's orders execute in: the next exchange day after it.

    The committee decides on the last completed session and its market orders fill at
    the next open, so the record is filed under the day the account actually traded.
    Filing under the decision date instead made the folder name disagree with the
    trade dates in the account statement. Weekends are skipped; the window carries no
    US market holiday, and one would have to be added here.
    """
    d = decision_date + TimeDelta(days=1)
    while d.weekday() >= 5:
        d += TimeDelta(days=1)
    return d.isoformat()


def save_analysis(reads, data: MarketData, out_dir: Path) -> Path:
    """Write the day's committee read, which the terminal report otherwise loses.

    This is the forward run's record: what each stock's memory retrieved, what each
    expert said, and what the committee decided, beside the equal-weight call over the
    same views. One file per trade date, so a re-run of the same day overwrites
    rather than accumulating.

    Saving is the default, which makes a subset run dangerous: on 2026-08-04 a
    four-symbol diagnostic silently replaced the day's twenty-stock record. A re-run
    that covers fewer stocks than the file already holds is therefore refused, because
    it can only ever destroy evidence. A same-size or larger re-run still overwrites,
    which is what a genuine correction looks like.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = {r.decision_date for r in reads}
    # The file is named for the day the orders fill; the payload keeps the day the
    # committee decided on. They are different dates and both are load-bearing:
    # cli.py refuses to trade when the payload's decision_date disagrees with the
    # observation the order path builds, which is the last completed session.
    decided = sorted(dates)[-1].isoformat() if dates else "unknown"
    stamp = _trade_date(sorted(dates)[-1]) if dates else "unknown"
    existing = out_dir / f"{stamp}.json"
    if existing.exists():
        try:
            held = len(json.loads(existing.read_text(encoding="utf-8"))["stocks"])
        except Exception:
            held = 0
        if held > len(reads):
            raise SystemExit(
                f"{existing.name} already holds {held} stocks and this run has "
                f"{len(reads)}; refusing to shrink the day's record. Re-run over the "
                f"full symbol list, or pass --save-analysis <other dir> for a subset."
            )
        # A session whose orders have already gone out is a settled record. A later run
        # can land on it again -- on 2026-08-06 one intended for the 5th fell back to
        # the 4th because that day's bar was not published yet -- and would otherwise
        # replace the read the orders were placed from with a different one.
        # Only the live record is settled by an order. Writing the same session into a
        # side directory (a counterfactual, a rehearsal) is not the same act.
        placed = out_dir.parent / stamp
        if out_dir.name == "reads" and placed.is_dir() and any(placed.glob("*.json")):
            raise SystemExit(
                f"orders were already placed against {stamp} ({placed}); refusing to "
                f"rewrite the read they were taken from. This run decided on {stamp} "
                "again, most likely because the next session's daily bar is not "
                "published yet. Wait for it, or pass --save-analysis <other dir>."
            )
    payload = {
        "decision_date": decided,
        "trade_date": stamp,
        "source": data.source,
        "account": data.account,
        "positions": data.positions,
        "stocks": [
            {
                "symbol": r.symbol,
                "arm": r.arm,
                "decision_date": r.decision_date.isoformat(),
                "close": r.close,
                "market_state": r.market_state,
                "weights": r.weights,
                "certified_cells": r.certified_cells,
                "frozen_at": r.frozen_at,
                "experts": [
                    {
                        "expert_id": e.expert_id,
                        # LONG / FLAT / ABSTAIN / NO DATA / ERROR / no view.
                        "label": e.label,
                        "confidence": e.confidence,
                        # Whether it entered the vote. An abstention is renormalised
                        # away; a flat view votes against. The distinction is the
                        # committee's, and it has to survive into the record.
                        "counted": e.counted,
                        "reasoning": e.reasoning,
                    }
                    for e in r.experts
                ],
                "score": r.score,
                "call": r.call,
                "score_equal": r.score_equal,
                "call_equal": r.call_equal,
                "agrees_with_equal": r.call == r.call_equal,
            }
            for r in reads
        ],
    }
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def load_saved_bars(in_dir: Path, symbols) -> MarketData:
    """Rebuild a MarketData from ``--save-bars`` output, headlines included.

    The news panel is restored when ``news.json`` is present, which is what makes an
    offline rehearsal a real rehearsal: without it the news seat is handed nothing and
    abstains, which is exactly the failure the saved file exists to prevent repeating.
    Runs saved before 2026-08-06 have no such file and replay without news, as they did.
    """
    import pandas as pd

    frames = {}
    for sym in list(symbols) + [PROXY]:
        path = in_dir / f"{sym}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} not found; run once with --save-bars {in_dir}")
        frames[sym] = pd.read_parquet(path)
    account, positions = {}, []
    meta = in_dir / "account.json"
    if meta.exists():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        account = payload.get("account", {})
        positions = payload.get("positions", [])
    panel = None
    news_file = in_dir / "news.json"
    if news_file.exists():
        from trading_agent.data.finsaber_news import NewsItem
        raw = json.loads(news_file.read_text(encoding="utf-8"))
        panel = IBNewsPanel({
            ticker: [NewsItem(date=Date.fromisoformat(row["date"]), ticker=ticker,
                              text=row.get("text") or row.get("headline") or "")
                     for row in rows]
            for ticker, rows in raw.items()
        })
    return MarketData(frames=frames, account=account, positions=positions,
                      news_panel=panel,
                      source=f"saved bars in {in_dir}"
                             + (" (headlines restored)" if panel is not None else ""))


# ------------------------------------------------------------------------ analysis


@dataclass
class ExpertRead:
    expert_id: str
    label: str                  # LONG / FLAT / NO DATA / ABSTAIN / ERROR / no view
    confidence: float | None
    reasoning: str
    counted: bool               # did it enter the vote


@dataclass
class SymbolRead:
    symbol: str
    arm: str
    decision_date: Date
    close: float
    market_state: str
    weights: dict
    certified_cells: list
    frozen_at: str | None
    experts: list
    score: float | None         # None when nobody spoke
    call: str                   # BULLISH / BEARISH / NEUTRAL / MAINTAIN
    # The equal-weight committee over the same views, scored alongside at no extra
    # model call. On a stock with no verified cell the two calls must agree every
    # day; that identity is the fallback holding live.
    score_equal: float | None
    call_equal: str


def observation_for(symbol: str, frame) -> Observation:
    row = frame.to_dict("records")[-1]
    d = Date.fromisoformat(str(row["date"])[:10])
    close = float(row["close"])
    # Same convention as the paper bridge: the decision is stamped at the close of the
    # last settled session, so nothing later than that bar is visible to the experts.
    stamp = DateTime.combine(d, Time(16, 0))
    return Observation(
        decision_date=d,
        decision_time=stamp,
        availability_time=stamp,
        bars={symbol: Bar(open=float(row.get("open") or close),
                          high=float(row.get("high") or close),
                          low=float(row.get("low") or close),
                          close=close, adjusted_close=close,
                          volume=float(row.get("volume") or 0.0))},
        tickers=[symbol],
    )


def call_for(score: float | None) -> str:
    """The frozen rule: long above zero, cash below, and a boundary score maintains."""
    if score is None:
        return "MAINTAIN"
    if score > 0:
        return "BULLISH"
    if score < 0:
        return "BEARISH"
    return "NEUTRAL"


def analyse(symbols: Sequence[str], data: MarketData, *, experts,
            live_experts: Sequence[str], news_alpha=None,
            memory_dir: Path = MEM) -> list[SymbolRead]:
    """Frames in, committee reads out. No connection, no printing.

    This is the whole decision path, so a test that drives it with synthetic frames
    exercises the market-state label, the context, the experts, the memory retrieval,
    the abstention rule and the boundary rule together.
    """
    loader = IBFrameWindowLoader(data.frames)
    panel = LoaderPricePanel(loader)
    proxy_levels = {
        Date.fromisoformat(str(r["date"])[:10]): float(r["close"])
        for r in data.frames[PROXY].to_dict("records") if r.get("close") == r.get("close")
    }
    classifier = MarketStateClassifier(panel, FROZEN, proxy_levels=proxy_levels)
    sec_path = sec_facts_path()
    macro = macro_path()
    missing = sec_coverage_gap(sec_path, symbols)
    if missing:
        raise SystemExit(
            f"the SEC facts at {sec_path.name} cover no filing for {missing}. The "
            "fundamental seat would abstain there on data grounds, not judgement, and "
            "the committee would be short a voter without saying so."
        )
    ctx_builder = build_paper_live_context(
        data.frames,
        sec_facts_path=str(sec_path),
        fred_macro_path=str(macro) if macro else None,
        news_panel=data.news_panel,
        news_sentiment_alpha=news_alpha,
        tickers=list(symbols),
    )

    out = []
    for symbol in symbols:
        arm = arm_for(symbol)
        obs = observation_for(symbol, data.frames[symbol])
        state = classifier.classify_at(obs.decision_date, [symbol])
        context = ctx_builder.build(obs, state)

        payload = json.loads((memory_dir / f"{symbol}__{arm}.json").read_text(encoding="utf-8"))
        store = ExpertScorecardStore.from_snapshot(payload, temperature=TAU)
        weights = store.get_weights("global" if arm == "global" else state)

        reads, voting = [], {}
        for eid in ORDER:
            if eid not in live_experts:
                reads.append(ExpertRead(eid, "ABSTAIN", None,
                                        "model call disabled by --no-llm", False))
                continue
            try:
                view = experts[eid].views(obs, state, context=context).get(symbol)
            except Exception as exc:
                reads.append(ExpertRead(eid, "ERROR", None,
                                        f"{type(exc).__name__}: {str(exc)[:44]}", False))
                continue
            if view is None:
                reads.append(ExpertRead(eid, "no view", None, "", False))
                continue
            # A safe default (data_unavailable / parse_error) is not a judgment. It
            # drops out of the vote and the remaining weights renormalise, which is
            # what the backtest does; counting it as flat would invent a bearish vote.
            safe = is_safe_default_view(view)
            counted = not safe and view.direction in ("long", "flat")
            label = ("NO DATA" if safe
                     else {"long": "LONG", "flat": "FLAT"}.get(view.direction, "ABSTAIN"))
            reads.append(ExpertRead(eid, label, None if safe else view.confidence,
                                    (view.reasoning or "").strip(), counted))
            if counted:
                voting[eid] = view

        score = aggregate(voting, weights, threshold=0.0).score if voting else None
        equal = {eid: 1.0 / len(ORDER) for eid in ORDER}
        score_equal = aggregate(voting, equal, threshold=0.0).score if voting else None
        out.append(SymbolRead(
            symbol=symbol, arm=arm, decision_date=obs.decision_date,
            close=obs.bars[symbol].close, market_state=state, weights=weights,
            certified_cells=payload.get("certified_cells", []),
            frozen_at=payload.get("frozen_at"), experts=reads,
            score=score, call=call_for(score),
            score_equal=score_equal, call_equal=call_for(score_equal),
        ))
    return out


# ------------------------------------------------------------------------- render


def bar_row(v: float, width: int = 20) -> str:
    return "#" * max(0, round(v * width))


def money(raw) -> str:
    try:
        return f"{float(raw):,.2f}"
    except (TypeError, ValueError):
        return str(raw)


def render(reads: list[SymbolRead], data: MarketData, symbols: Sequence[str]) -> None:
    print()
    print("=" * W)
    print("  LIVE COMMITTEE READ".ljust(W - 22) + "no order path".rjust(20))
    print("=" * W)
    if data.source != "ib":
        print(f"  source: {data.source}  (rehearsal, not a live pull)")

    print()
    print("  ACCOUNT (IB paper)")
    shown = False
    for tag, label in (("NetLiquidation", "net liquidation"), ("TotalCashValue", "cash"),
                       ("AvailableFunds", "available funds"),
                       ("GrossPositionValue", "position value"),
                       ("UnrealizedPnL", "unrealised P&L")):
        if tag in data.account:
            print(f"    {label:<18}{money(data.account[tag]):>16}")
            shown = True
    if not shown:
        print("    not available")

    print()
    print("  POSITIONS")
    if not data.positions:
        print("    none open")
    else:
        print(f"    {'symbol':<8}{'quantity':>10}{'avg cost':>12}{'':4}{'watchlist':<10}")
        for p in sorted(data.positions, key=lambda r: r["symbol"]):
            mark = "yes" if p["symbol"] in symbols else ""
            print(f"    {p['symbol']:<8}{p['quantity']:>10.0f}"
                  f"{p['avg_cost']:>12.2f}    {mark:<10}")

    print()
    print("  WATCHLIST")
    print(f"    {'symbol':<14}{'last close':>12}{'date':>14}{'bars':>7}")
    for sym in list(symbols) + [PROXY]:
        f = data.frames[sym]
        label = f"{PROXY} (proxy)" if sym == PROXY else sym
        print(f"    {label:<14}{float(f['close'].iloc[-1]):>12.2f}"
              f"{str(f['date'].iloc[-1])[:10]:>14}{len(f):>7}")

    for r in reads:
        print()
        print("-" * W)
        print(f"  {r.symbol}   {r.decision_date}   close {r.close:.2f}"
              f"   market state: {r.market_state}")
        print("-" * W)
        print("  expert views")
        for e in r.experts:
            conf = "" if e.confidence is None else f"conf {e.confidence:.2f}"
            print(f"    {e.expert_id:<14}{e.label:<10}{conf}")
            if e.reasoning:
                print(f"                  {e.reasoning[:56]}")
        n = len(r.certified_cells)
        print()
        print(f"  trust weights   frozen {r.frozen_at}, {n} verified "
              f"cell{'s' if n != 1 else ''}, tau={TAU}, arm={r.arm}")
        for eid in ORDER:
            w = r.weights.get(eid, 0.0)
            print(f"    {eid:<14}{w:.3f}  {bar_row(w)}")
        print()
        if r.score is None:
            print("  VIEW            no expert spoke -> maintain the current position")
        else:
            spoke = ", ".join(sorted(e.expert_id for e in r.experts if e.counted))
            print(f"  AGGREGATE       A = {r.score:+.3f}   over {spoke}"
                  f"   (weights renormalised over these)")
            detail = {"BULLISH": "target long", "BEARISH": "target cash",
                      "NEUTRAL": "score at the boundary, hold the current position"}
            print(f"  VIEW            {r.call}   {detail.get(r.call, '')}")
            same = "same call" if r.call_equal == r.call else "DIFFERS"
            print(f"  EQUAL WEIGHTS   A = {r.score_equal:+.3f}   {r.call_equal}"
                  f"   ({same})")

    print()
    print("=" * W)
    print("  Analysis only. No order was built, previewed or submitted.")
    print("=" * W)
    print()


# --------------------------------------------------------------------------- main


def load_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = ROOT / ".env"
    if not env.exists():
        return
    # utf-8-sig strips a leading BOM. Windows editors write one, and it would otherwise
    # fuse into the first key name and break the lookup silently.
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live committee read. Cannot trade.")
    ap.add_argument("--symbols", default=None,
                    help="comma list; default is all twenty deployed stocks")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=41)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the two model-backed experts; they abstain")
    # On by default: a live read that does not keep its bars cannot be re-examined,
    # and the baselines of Chapter 6 are recomputed from these files after the run.
    ap.add_argument("--save-bars", default=str(ROOT / "reports" / "paper_forward" / "bars"),
                    metavar="DIR",
                    help="write the bars so --offline can rehearse this read later")
    ap.add_argument("--save-analysis", nargs="?", metavar="DIR",
                    const=str(ROOT / "reports" / "paper_forward" / "reads"),
                    default=str(ROOT / "reports" / "paper_forward" / "reads"),
                    help="write the day's committee read as JSON. The terminal report "
                         "is ephemeral, and this is the forward run's record.")
    ap.add_argument("--offline", default=None, metavar="DIR",
                    help="read saved bars instead of connecting; implies --no-llm news")
    args = ap.parse_args(argv)

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else list(DEFAULT_SYMBOLS))

    if args.offline:
        data = load_saved_bars(Path(args.offline), symbols)
    else:
        data = load_market_data(symbols, host=args.host, port=args.port,
                                client_id=args.client_id, want_news=not args.no_llm)
        if args.save_bars:
            save_bars(data, Path(args.save_bars))

    if not args.no_llm:
        load_openai_key()
    # The technical expert votes from the TradingView rating in the context and never
    # calls its backend, so it is real under either backend. With --no-llm the other two
    # would return the static stub, so they are skipped and recorded as abstaining.
    use_llm = not args.no_llm and not args.offline
    experts = {e.expert_id: e for e in build_experts("openai" if use_llm else "static")}
    live_experts = ORDER if use_llm else ("technical",)

    # The news expert reads attention, novelty and sentiment, not raw headlines, so the
    # panel alone leaves it with no signals. The sentiment alpha is what turns the
    # headlines into those fields, and it is cached so a repeated run costs nothing.
    news_alpha = None
    if data.news_panel is not None and use_llm:
        from trading_agent.alpha.llm_news_sentiment import LLMNewsSentimentAlpha, NewsSentimentCache
        from trading_agent.experts.backends import OpenAIChatBackend

        cache_path = ROOT / "reports" / "corpus" / "news_sentiment_live.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        news_alpha = LLMNewsSentimentAlpha(
            data.news_panel,
            backend=OpenAIChatBackend(model="gpt-4.1-mini", temperature=0.0),
            cache=NewsSentimentCache(cache_path),
        )

    reads = analyse(symbols, data, experts=experts, live_experts=live_experts,
                    news_alpha=news_alpha)
    render(reads, data, symbols)
    if args.save_analysis:
        path = save_analysis(reads, data, Path(args.save_analysis))
        print(f"\n  committee read written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
