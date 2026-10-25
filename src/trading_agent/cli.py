"""TradingAgents-style launcher for research and forward-evaluation modes."""

from __future__ import annotations

import argparse
import json
from textwrap import dedent
from datetime import date, datetime, time, timedelta
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

from trading_agent.aggregation.soft_weighting import aggregate, aggregate_by_ticker
from trading_agent.alpha.market_state_classifier import (
    MARKET_STATES,
    MarketStateClassifier,
    MarketStateConfig,
)
from trading_agent.alpha.alpha158_strategy import available_tickers_from_sqlite_qlib_factors
from trading_agent.contracts import ALL_AVAILABLE, BacktestConfig, Bar, ExpertView, Observation
from trading_agent.experts.backends import (
    DEFAULT_CODEX_SDK_MODEL,
    DEFAULT_EXPERT_MODEL,
    SUPPORTED_EXPERT_BACKENDS,
    OpenAIChatBackend,
    build_experts,
    expert_backend_metadata,
)
from trading_agent.memory.expert_weights import GLOBAL_MARKET_STATE, load_snapshot_payload
from trading_agent.memory.expert_scorecard import (
    ExpertScorecardStore,
    snapshot_entries,
)
from trading_agent.paper_trading import (
    COMMON_IB_PAPER_PORTS,
    IBGatewayConfig,
    account_cash_from_summary,
    auto_decide_order_intents,
    build_order_previews,
    build_paper_manifest,
    parse_order_spec,
    submit_order_previews,
)
from trading_agent.reporting import REPORT_SCHEMA_VERSION, SYSTEM_IDENTITY

try:
    from data_collection.ib_connector import connect_ib, request_daily_bars
except ImportError:  # pragma: no cover - only relevant outside the project env
    connect_ib = None
    request_daily_bars = None


def _package_version() -> str:
    try:
        return metadata.version("trading-agent")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def build_static_experts():
    return build_experts("static")


def resolve_data_root(*args, **kwargs):
    from trading_agent.data.finsaber_panel import resolve_data_root as _resolve_data_root

    return _resolve_data_root(*args, **kwargs)


def select_fixed_universe(*args, **kwargs):
    from trading_agent.data.finsaber_panel import select_fixed_universe as _select_fixed_universe

    return _select_fixed_universe(*args, **kwargs)


def load_panel(*args, **kwargs):
    from trading_agent.data.finsaber_panel import load_panel as _load_panel

    return _load_panel(*args, **kwargs)


def build_expert_signal_sources(*args, **kwargs):
    from trading_agent.expert_context import build_expert_signal_sources as _build_expert_signal_sources

    return _build_expert_signal_sources(*args, **kwargs)


def run_expert_loop(*args, **kwargs):
    from trading_agent.runner import run_expert_loop as _run_expert_loop

    return _run_expert_loop(*args, **kwargs)


def _error_summary_from_result(*args, **kwargs):
    from trading_agent.experiments.expert_weight_holdout import (
        _error_summary_from_result as _impl,
    )

    return _impl(*args, **kwargs)


def _per_state_metrics(*args, **kwargs):
    from trading_agent.experiments.expert_weight_holdout import _per_state_metrics as _impl

    return _impl(*args, **kwargs)


def _state_counts(*args, **kwargs):
    from trading_agent.experiments.expert_weight_holdout import _state_counts as _impl

    return _impl(*args, **kwargs)


def _episode_counts_by_state_action(result) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for episode in result.episodes:
        state = episode.market_state
        action = getattr(episode, "action", "unknown")
        out.setdefault(state, {"long": 0, "flat": 0})
        out[state][action] = out[state].get(action, 0) + 1
    return out


def _episode_count_before_after(result) -> dict[str, dict[str, int]]:
    before: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for episode in result.episodes:
        key = (episode.decision_date.isoformat(), episode.market_state)
        if key in seen:
            continue
        seen.add(key)
        before[episode.market_state] = before.get(episode.market_state, 0) + 1
    after = {
        state: sum(actions.values())
        for state, actions in _episode_counts_by_state_action(result).items()
    }
    return {
        state: {
            "before_universe_level": int(before.get(state, 0)),
            "after_per_name": int(after.get(state, 0)),
            "increase": int(after.get(state, 0) - before.get(state, 0)),
        }
        for state in sorted(set(before) | set(after))
    }


def _uses_non_paper_port(args: argparse.Namespace) -> bool:
    return int(args.port) not in COMMON_IB_PAPER_PORTS


def _require_paper_execution_port(args: argparse.Namespace) -> None:
    if args.execute_orders and _uses_non_paper_port(args) and not args.allow_non_paper_port:
        allowed = ", ".join(str(port) for port in COMMON_IB_PAPER_PORTS)
        raise ValueError(
            "--execute-orders refuses non-paper IB port "
            f"{args.port}; use one of {allowed}, or pass --allow-non-paper-port."
        )


def _require_resolved_execution_quantities(args: argparse.Namespace, order_intents: Sequence) -> None:
    if not args.execute_orders:
        return
    unresolved = [
        intent.ticker
        for intent in order_intents
        if intent.quantity == ALL_AVAILABLE
    ]
    if unresolved:
        tickers = ", ".join(unresolved)
        raise ValueError(
            "--execute-orders requires resolved numeric quantities; "
            f"ALL quantity is unresolved for {tickers}."
        )


def _require_execution_order_source(args: argparse.Namespace) -> None:
    if args.execute_orders and not args.order and not args.auto_decide:
        raise ValueError("--execute-orders requires --order or --auto-decide")


def _paper_order_source(args: argparse.Namespace) -> str:
    has_manual_orders = bool(args.order)
    if has_manual_orders and args.auto_decide:
        return "manual_and_auto_decide"
    if args.auto_decide:
        return "auto_decide"
    if has_manual_orders:
        return "manual"
    return "none"


def _require_paper_auto_decide_args(args: argparse.Namespace) -> None:
    if not args.auto_decide:
        return
    if not args.use_account_cash and float(args.paper_cash) <= 0:
        raise ValueError("--auto-decide requires positive paper cash")
    if int(args.paper_vol_window) <= 0:
        raise ValueError("--paper-vol-window must be positive")
    if int(args.paper_percentile_window) <= 0:
        raise ValueError("--paper-percentile-window must be positive")
    if int(args.lookback_days) <= 0:
        raise ValueError("--lookback-days must be positive")


def _require_paper_connection_args(args: argparse.Namespace) -> None:
    if not str(args.host).strip():
        raise ValueError("--host must be non-empty")
    if not 1 <= int(args.port) <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if int(args.client_id) <= 0:
        raise ValueError("--client-id must be positive")


def _require_connection_smoke_args(args: argparse.Namespace) -> None:
    if not args.connection_smoke:
        return
    if args.dry_run:
        raise ValueError("--connection-smoke cannot be combined with --dry-run")
    if args.execute_orders:
        raise ValueError("--connection-smoke cannot be combined with --execute-orders")
    if args.auto_decide:
        raise ValueError("--connection-smoke cannot be combined with --auto-decide")
    if args.order:
        raise ValueError("--connection-smoke cannot be combined with --order")


def _view_to_payload(view: ExpertView) -> dict:
    return {
        "expert_id": view.expert_id,
        "direction": view.direction,
        "confidence": float(view.confidence),
        "reasoning": view.reasoning,
        "evidence": list(view.evidence),
        "risk_flags": list(view.risk_flags),
        "data_quality": view.data_quality,
    }


def _views_by_ticker_payload(views: dict[str, dict[str, ExpertView]]) -> dict:
    tickers = sorted({ticker for by_ticker in views.values() for ticker in by_ticker})
    return {
        ticker: {
            expert_id: _view_to_payload(by_ticker[ticker])
            for expert_id, by_ticker in sorted(views.items())
            if ticker in by_ticker
        }
        for ticker in tickers
    }


def _legacy_expert_views_payload(views: dict[str, dict[str, ExpertView]]) -> dict:
    out = {}
    for expert_id, by_ticker in sorted(views.items()):
        if not by_ticker:
            continue
        first = by_ticker[sorted(by_ticker)[0]]
        out[expert_id] = _view_to_payload(first)
    return out


def _long_symbols_from_aggregation(symbols: Sequence[str], aggregation_payload: dict | None) -> list[str]:
    if not aggregation_payload:
        return []
    by_ticker = aggregation_payload.get("by_ticker", {})
    return [
        symbol for symbol in symbols
        if by_ticker.get(symbol, {}).get("direction") == "long"
    ]


_LABEL_TO_DIRECTION = {"LONG": "long", "FLAT": "flat", "ABSTAIN": "neutral",
                       "NO DATA": "neutral", "ERROR": "neutral", "no view": "neutral"}


def _committee_read_path(args: argparse.Namespace) -> Path | None:
    """The recorded committee read this order run must follow, if one is configured."""
    raw = (getattr(args, "committee_read", "") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(
                f"--committee-read {path} holds no committee read. Run "
                "scripts/live_analysis.py --save-analysis first."
            )
        path = files[-1]
    if not path.exists():
        raise ValueError(f"--committee-read {path} not found.")
    return path


def _views_from_committee_read(path: Path, observation: Observation,
                               market_state: str) -> dict[str, dict[str, ExpertView]]:
    """Replay the recorded committee read as this run's expert views.

    The order path used to build views from ``build_static_experts()``, whose backend
    always answers "long". On 2026-08-03 that produced BUY previews on all twenty
    stocks, including the four the real committee had called BEARISH. The two paths were
    reaching different conclusions and the one placing orders was the one not listening.

    The committee read is the system: it runs the three real experts over live bars and
    live news and writes its result. Reading that file makes the order path a consumer
    of the same decision instead of a second, weaker copy of it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_date = str(payload.get("decision_date", ""))[:10]
    if recorded_date != observation.decision_date.isoformat():
        raise ValueError(
            f"the committee read in {path.name} is dated {recorded_date} but this run's "
            f"observation is {observation.decision_date.isoformat()}. Refusing to trade "
            "on a decision taken for a different session; re-run scripts/live_analysis.py."
        )
    by_symbol = {s["symbol"]: s for s in payload.get("stocks", [])}
    out: dict[str, dict[str, ExpertView]] = {}
    for ticker in observation.tickers:
        record = by_symbol.get(ticker)
        if record is None:
            raise ValueError(
                f"{path.name} carries no committee read for {ticker}. Re-run "
                "scripts/live_analysis.py so every traded symbol is covered."
            )
        if record.get("market_state") != market_state:
            raise ValueError(
                f"{ticker}: the committee read was taken in market state "
                f"{record.get('market_state')!r} and this run is in {market_state!r}. "
                "The weights would come from a different cell; re-run the read."
            )
        for expert in record.get("experts", []):
            label = str(expert.get("label", ""))
            direction = _LABEL_TO_DIRECTION.get(label)
            if direction is None:
                raise ValueError(
                    f"{ticker}: unrecognised expert label {label!r} in {path.name}."
                )
            confidence = expert.get("confidence")
            out.setdefault(expert["expert_id"], {})[ticker] = ExpertView(
                expert_id=expert["expert_id"],
                direction=direction,
                confidence=float(confidence) if confidence is not None else 0.0,
                reasoning=str(expert.get("reasoning", ""))[:400],
            )
    return out


def _paper_views(observation: Observation, market_state: str,
                 committee_read: Path | None = None) -> dict[str, dict[str, ExpertView]]:
    if committee_read is not None:
        return _views_from_committee_read(committee_read, observation, market_state)
    out: dict[str, dict[str, ExpertView]] = {}
    for expert in build_static_experts():
        if hasattr(expert, "views"):
            out[expert.expert_id] = expert.views(observation, market_state, context=None)
        else:
            view = expert.view(observation, market_state, context=None)
            out[expert.expert_id] = {ticker: view for ticker in observation.tickers}
    return out


def _state_has_learned_memory(market_state: str, memory_store) -> bool:
    """True when the frozen scorecard actually tilts the weights for this state."""
    if memory_store is None:
        return False
    return memory_store.support(market_state) >= memory_store._min_support


# Paper decisions use the FROZEN system threshold (theta=0, strict boundaries,
# 2026-07-17 settlement) — not the aggregation function's 0.5 signature default.
_PAPER_DECISION_THRESHOLD: float = 0.0
# The frozen action rule is HOLD semantics: enter on a score strictly above theta,
# exit on one strictly below the exit boundary, and MAINTAIN at a boundary, because a
# zero net spoken edge is evidence for nothing. That needs the account's current
# position. The paper bridge used to aggregate without it, which made every session a
# fresh entry vote: a held stock scoring exactly zero was read as "not long" and closed.
# On 2026-08-05 that was O, at score 0.000 on 78 shares. The account is read here, so
# the position is passed and the deployed rule is the registered one.
_PAPER_EXIT_THRESHOLD: float = 0.0


def _prev_directions(held: Mapping[str, float] | None,
                     tickers: Sequence[str]) -> dict[str, str] | None:
    """Map the account's holdings to the previous action, or None if it was not read.

    None keeps the entry-re-vote behaviour rather than asserting a position that was
    never observed; {} is a read account holding nothing.
    """
    if held is None:
        return None
    return {t: ("long" if float(held.get(t, 0.0)) > 0 else "flat") for t in tickers}


def _memory_application(
    market_state: str,
    memory_store,
    views: dict[str, dict[str, ExpertView]],
    tickers: Sequence[str] = (),
    held: Mapping[str, float] | None = None,
) -> tuple[dict | None, dict, float]:
    tickers = tuple(tickers)
    prev = _prev_directions(held, tickers)
    hold = dict(prev_directions=prev, exit_threshold=_PAPER_EXIT_THRESHOLD) if prev is not None else {}
    if not _state_has_learned_memory(market_state, memory_store):
        equal_result = aggregate_by_ticker(
            views, {}, tickers=tickers, threshold=_PAPER_DECISION_THRESHOLD, **hold
        )
        long_scores = [
            decision.score for decision in equal_result.by_ticker.values()
            if decision.direction == "long"
        ]
        top_score = max((decision.score for decision in equal_result.by_ticker.values()), default=0.0)
        direction = "long" if long_scores else "flat"
        return None, {
            "direction": direction,
            "score": float(max(long_scores, default=top_score)),
            "weights": dict(equal_result.weights_used),
            "by_ticker": {
                ticker: {"direction": item.direction, "score": float(item.score)}
                for ticker, item in equal_result.by_ticker.items()
            },
        }, 1.0
    weights = memory_store.get_weights(market_state)
    result = aggregate_by_ticker(
        views, weights, tickers=tickers, threshold=_PAPER_DECISION_THRESHOLD, **hold
    )
    long_scores = [
        decision.score for decision in result.by_ticker.values()
        if decision.direction == "long"
    ]
    top_score = max((decision.score for decision in result.by_ticker.values()), default=0.0)
    direction = "long" if long_scores else "flat"
    return {
        "market_state": market_state,
        "state_support": memory_store.support(market_state),
        "weights": dict(result.weights_used),
        "aggregation_score": float(max(long_scores, default=top_score)),
        "direction": direction,
        "by_ticker": {
            ticker: {"direction": item.direction, "score": float(item.score)}
            for ticker, item in result.by_ticker.items()
        },
    }, {
        "direction": direction,
        "score": float(max(long_scores, default=top_score)),
        "weights": dict(result.weights_used),
        "by_ticker": {
            ticker: {"direction": item.direction, "score": float(item.score)}
            for ticker, item in result.by_ticker.items()
        },
    }, (1.0 if direction == "long" else 0.0)   # binary deployment: no discretionary sizing (Ch3)


def _held_quantities(ib, symbols: Sequence[str]) -> dict[str, float] | None:
    """Quantity currently held for each traded symbol, or None if the account is unreadable.

    None and an empty dict mean different things downstream: None is "the account was
    not read, do not infer a position", {} is "read, and nothing is held".
    """
    try:
        rows = ib.positions()
    except Exception:
        return None
    wanted = {str(s).upper() for s in symbols}
    held: dict[str, float] = {}
    for row in rows or []:
        contract = getattr(row, "contract", None)
        symbol = str(getattr(contract, "symbol", "") or "").upper()
        if symbol in wanted:
            held[symbol] = held.get(symbol, 0.0) + float(getattr(row, "position", 0.0))
    return held


def _require_applicable_memory(market_state: str, memory_store) -> None:
    if not _state_has_learned_memory(market_state, memory_store):
        raise ValueError(
            "--require-memory requires the current market_state to have learned memory "
            "(scorecard support at or above the tilt threshold)"
        )


def _parse_exit_threshold(value: str):
    raw = str(value).strip().lower()
    if raw in {"none", "off"}:
        return None
    return float(value)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_symbols(value: str) -> list[str]:
    symbols = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not symbols:
        raise ValueError("--symbols requires at least one symbol")
    return symbols


def _write_json(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


def _require_frozen_memory_snapshot(payload: dict) -> None:
    if payload.get("status") != "frozen":
        raise ValueError("paper mode requires a frozen memory snapshot")



def _memory_snapshot_info(path: str | Path, payload: dict, entries: Sequence) -> dict:
    cards = payload.get("cards", [])
    return {
        "path": str(path),
        "status": payload.get("status"),
        "frozen_at": payload.get("frozen_at"),
        "scope": payload.get("scope"),
        "entries_loaded": len(cards),                 # number of scorecards loaded
        "learned_adjustments_loaded": len(entries),   # nonzero reflection deltas
        "adjustments": [
            {
                "expert_id": entry.expert_id,
                "market_state": entry.market_state,
                "weight_delta": float(entry.weight_delta),
                "status": entry.status,
            }
            for entry in entries
        ],
    }


def _latest_bar_report(frame) -> dict:
    last_date = None
    if not getattr(frame, "empty", True):
        try:
            last_date = str(frame["date"].iloc[-1])
        except (KeyError, TypeError, AttributeError, IndexError):
            last_date = None
    return {
        "rows": int(len(frame)),
        "last_date": last_date,
    }


def _latest_frame_value(frame, column: str, fallback: float | None = None) -> float:
    try:
        value = frame[column].iloc[-1]
    except (KeyError, TypeError, AttributeError, IndexError):
        if fallback is None:
            raise
        return float(fallback)
    return float(value)


def _frame_value_at(frame, column: str, index: int):
    series = frame[column]
    return series.iloc[index]


class _PaperPricePanel:
    """Thin adapter from IB daily bar frames to MarketStateClassifier's panel API."""

    def __init__(self, latest_frames: dict[str, object]) -> None:
        by_symbol: dict[str, dict[date, float]] = {}
        dates = set()
        for symbol, frame in latest_frames.items():
            prices: dict[date, float] = {}
            try:
                n_rows = int(len(frame))
            except TypeError:
                n_rows = 0
            for i in range(n_rows):
                try:
                    raw_date = _frame_value_at(frame, "date", i)
                    close = _frame_value_at(frame, "adjusted_close", i)
                except (KeyError, TypeError, AttributeError, IndexError):
                    try:
                        raw_date = _frame_value_at(frame, "date", i)
                        close = _frame_value_at(frame, "close", i)
                    except (KeyError, TypeError, AttributeError, IndexError):
                        continue
                try:
                    dt = date.fromisoformat(str(raw_date)[:10])
                    price = float(close)
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    prices[dt] = price
                    dates.add(dt)
            by_symbol[symbol] = prices
        self._by_symbol = by_symbol
        self.dates = sorted(dates)

    def _price_on(self, ticker: str, day) -> float | None:
        try:
            dt = date.fromisoformat(str(day)[:10]) if not isinstance(day, date) else day
        except ValueError:
            return None
        return self._by_symbol.get(ticker, {}).get(dt)


def _resolve_paper_market_state(
    args: argparse.Namespace,
    observation: Observation,
    latest_frames: dict[str, object],
    proxy_frame: object | None = None,
) -> tuple[str, str]:
    if args.paper_market_state != "auto":
        return args.paper_market_state, "manual_cli"
    config = MarketStateConfig(
        vol_window=args.paper_vol_window,
        percentile_window=args.paper_percentile_window,
        trend_window=getattr(args, "paper_trend_window", 200),
        trend_confirm_days=getattr(args, "paper_trend_confirm_days", 5),
    )
    # External market proxy (e.g. SPY): the traded name never enters the state label,
    # so a single-name run cannot leak its own trend/vol into the conditioning variable
    # (market_state_classifier.py docstring). When no proxy frame is available the old
    # self-referential panel is used, and the source label records the difference.
    proxy_levels: dict | None = None
    proxy_panel = None
    if proxy_frame is not None:
        proxy_panel = _PaperPricePanel({"__proxy__": proxy_frame})
        proxy_levels = proxy_panel._by_symbol.get("__proxy__") or None
    if proxy_levels:
        state = MarketStateClassifier(
            proxy_panel, config, proxy_levels=proxy_levels
        ).classify(observation)
        source = "classifier_proxy"
    else:
        # Without the external proxy the only prices to hand are the traded symbols'
        # own, which would put a stock inside its own conditioning variable. The
        # backtest refuses that outright; the live path must not do quietly what the
        # backtest refuses. Hold the frozen default instead and say why.
        return "up_calm", "proxy_unavailable_no_self_conditioning"
    if state in MARKET_STATES:
        return state, source
    return "up_calm", f"{source}_warmup_fallback"


def _fetch_market_proxy_frame(ib, args: argparse.Namespace, history_days: int):
    """Fetch the external market-proxy bars (e.g. SPY) used for market_state, or None.

    Returns None when the state is a manual override, when no proxy symbol is set, or
    when the fetch fails — in which case the caller falls back to the traded symbols'
    own bars and the source label records it.
    """
    symbol = (getattr(args, "market_proxy_symbol", "") or "").strip()
    if args.paper_market_state != "auto" or not symbol:
        return None
    try:
        frame = request_daily_bars(
            ib, symbol, duration=_paper_history_duration_arg(history_days)
        )
    except Exception:
        return None
    # An empty frame is the same outcome as a failed fetch and must be reported the
    # same way. IB returns one rather than raising when a request is refused, and a
    # silently empty proxy is what makes the state self-referential.
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _reject_partial_session(frames: dict, symbol_order: Sequence[str]) -> None:
    """Drop any bar for a session that has not closed, in place.

    ``request_daily_bars`` asks IB with ``endDateTime=""``, so during regular hours the
    last daily bar is TODAY'S PARTIAL bar: on 2026-08-03 at 09:50 ET one symbol came
    back with volume 30,155 against the previous session's 782,233. The protocol decides
    on a completed close, so that bar is not an input.

    Dropping it and deciding on the last completed session IS the protocol. What the
    protocol does not require is that the fill happen at the open: a mid-session fill is
    an execution deviation to disclose, not a look-ahead. Refusing the whole run, as this
    did when first written, confused the two.
    """
    from datetime import timedelta, timezone

    eastern = timezone(timedelta(hours=-4))  # EDT; the forward window is in August
    now = datetime.now(eastern)
    minutes = now.hour * 60 + now.minute
    if not (9 * 60 + 30 <= minutes < 16 * 60 and now.weekday() < 5):
        return
    today = now.date().isoformat()
    dropped = []
    for symbol in symbol_order:
        frame = frames.get(symbol)
        try:
            last = str(frame["date"].iloc[-1])[:10]
        except (KeyError, TypeError, AttributeError, IndexError):
            continue
        if last == today:
            frames[symbol] = frame.iloc[:-1].copy()
            dropped.append(symbol)
    if dropped:
        print(f"dropped the unfinished {today} bar on {dropped}: the session is still "
              f"open at {now:%H:%M} ET, so it is not a close. Deciding on the last "
              "completed session.")


def _paper_history_duration_days(args: argparse.Namespace) -> int:
    if getattr(args, "paper_market_state", "auto") != "auto":
        return int(args.lookback_days)
    # Enough trailing history for BOTH the RV percentile window and the trend SMA.
    # The trend axis needs its full window (frozen system: 200) or it never leaves
    # warmup, which would pin every live decision to the up_calm fallback.
    trend_window = int(getattr(args, "paper_trend_window", 200))
    required_trading_days = max(
        int(args.paper_vol_window) + int(args.paper_percentile_window),
        trend_window,
    ) + 2
    required_calendar_days = int(required_trading_days * 1.5) + 2
    return max(int(args.lookback_days), required_calendar_days)


def _paper_history_duration_arg(days: int) -> str:
    """IB duration string for ``days`` calendar days.

    IB rejects a day-unit duration much past a year: a request for "792 D" times out
    and returns an EMPTY frame rather than an error. That is the dangerous shape --
    for the market proxy an empty frame leaves the classifier with no series and it
    falls back to the traded stock's own history, which is the self-referential label
    the design forbids. Measured on the paper gateway: 1 Y -> 251 rows, 2 Y -> 501,
    3 Y -> 751, 792 D -> 0. Anything beyond a year is therefore asked for in years.
    """
    if days <= 365:
        return f"{days} D"
    return f"{-(-days // 365)} Y"  # ceil, so the window is never short


def _paper_observation(symbols: Sequence[str], latest_frames: dict[str, object]) -> Observation:
    bars: dict[str, Bar] = {}
    decision_date = date.today()
    for symbol in symbols:
        frame = latest_frames[symbol]
        close = _latest_frame_value(frame, "close")
        try:
            raw_date = frame["date"].iloc[-1]
            decision_date = date.fromisoformat(str(raw_date)[:10])
        except (KeyError, TypeError, AttributeError, IndexError, ValueError):
            pass
        bars[symbol] = Bar(
            open=_latest_frame_value(frame, "open", close),
            high=_latest_frame_value(frame, "high", close),
            low=_latest_frame_value(frame, "low", close),
            close=close,
            adjusted_close=_latest_frame_value(frame, "adjusted_close", close),
            volume=_latest_frame_value(frame, "volume", 0.0),
        )
    decision_time = datetime.combine(decision_date, time(16, 0))
    return Observation(
        decision_date=decision_date,
        decision_time=decision_time,
        availability_time=decision_time,
        bars=bars,
        tickers=list(symbols),
    )


def _backtest_config(args: argparse.Namespace, tickers: Sequence[str]) -> BacktestConfig:
    # A positive bps cost replaces the per-share commission model inside the
    # execution adapter; zero the commission parameters so the fingerprint and
    # manifest state the cost model actually charged.
    bps_model_active = float(args.headline_cost_bps_per_side) > 0
    return BacktestConfig(
        date_from=args.start,
        date_to=args.end,
        preload_date_from=args.preload_start,
        initial_cash=args.initial_cash,
        tickers=tuple(tickers),
        commission_per_share=0.0 if bps_model_active else args.commission_per_share,
        min_commission=0.0 if bps_model_active else args.min_commission,
        max_commission_rate=args.max_commission_rate,
        slippage_perc=args.slippage_perc,
        execution_cost_bps_per_side=args.headline_cost_bps_per_side,
        liquidity_cap_pct=args.liquidity_cap_pct,
        seed=args.seed,
    )


def _preload_start(start: date, calendar_days: int) -> date:
    return start - timedelta(days=max(0, int(calendar_days)))


def _require_backtest_date_window(args: argparse.Namespace) -> None:
    if args.start >= args.end:
        raise ValueError("--start must be before --end")


def _require_holdout_date_window(args: argparse.Namespace) -> None:
    validation_boundary = getattr(args, "validation_boundary", None)
    if validation_boundary is not None:
        if not (args.start < validation_boundary < args.boundary < args.end):
            raise ValueError("--start must be before --validation-boundary, before --boundary, and --boundary before --end")
        return
    if not (args.start < args.boundary < args.end):
        raise ValueError("--start must be before --boundary, and --boundary before --end")


def _require_positive_args(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        if getattr(args, name) <= 0:
            flag = f"--{name.replace('_', '-')}"
            raise ValueError(f"{flag} must be positive")


def _require_non_negative_args(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        if getattr(args, name) < 0:
            flag = f"--{name.replace('_', '-')}"
            raise ValueError(f"{flag} must be non-negative")


def _require_experiment_parameters(args: argparse.Namespace) -> None:
    _require_positive_args(
        args,
        (
            "names",
            "reflect_every",
            "reflection_window",
            "reflection_min_support",
            "initial_cash",
            "vol_window",
            "percentile_window",
        ),
    )
    _require_non_negative_args(
        args,
        (
            "commission_per_share",
            "min_commission",
            "max_commission_rate",
            "slippage_perc",
            "headline_cost_bps_per_side",
            "sensitivity_cost_bps_per_side",
            "liquidity_cap_pct",
            "preload_calendar_days",
        ),
    )


def _expert_model_revision(args: argparse.Namespace) -> str:
    raw = getattr(args, "expert_model_revision", None)
    if raw:
        return str(raw)
    backend = str(args.expert_backend)
    if backend in {"openai", "codex_sdk"}:
        return str(args.expert_model)
    return f"{backend}-deterministic"


def _require_expert_model_identity(args: argparse.Namespace) -> None:
    if not getattr(args, "expert_model", None):
        args.expert_model = (
            DEFAULT_CODEX_SDK_MODEL
            if str(args.expert_backend).strip() == "codex_sdk"
            else DEFAULT_EXPERT_MODEL
        )
    args.expert_model_revision = _expert_model_revision(args)
    if not str(args.expert_backend).strip():
        raise ValueError("--expert-backend is required")
    if not str(args.expert_model).strip():
        raise ValueError("--expert-model is required")
    if not str(args.expert_model_revision).strip():
        raise ValueError("--expert-model-revision is required")


def run_backtest_mode(args: argparse.Namespace) -> int:
    _require_backtest_date_window(args)
    _require_experiment_parameters(args)
    _require_expert_model_identity(args)
    args.preload_start = _preload_start(args.start, args.preload_calendar_days)
    data_root = str(resolve_data_root(args.data_root))
    factor_artifact_tickers = (
        available_tickers_from_sqlite_qlib_factors(args.alpha158_factors_db)
        if args.alpha158_factors_db
        else None
    )
    selection = select_fixed_universe(
        data_root,
        args.start,
        args.end,
        n=args.names,
        allowed_tickers=factor_artifact_tickers,
    )
    loader = load_panel(data_root, args.preload_start, args.end, tickers=selection.tickers, modalities=("price",))
    config = _backtest_config(args, selection.tickers)
    news_sentiment_backend = (
        OpenAIChatBackend(model=args.news_sentiment_model, temperature=0.0)
        if args.news_sentiment_cache
        else None
    )
    # Design D: technical = deterministic TradingView rating (same wiring as the holdout
    # launcher) computed point-in-time on the already-loaded OHLCV panel.
    from trading_agent.alpha.technical_rating import TradingViewRating
    from trading_agent.market import LoaderOHLCVPanel

    signal_sources = build_expert_signal_sources(
        data_root=data_root,
        start=args.preload_start,
        end=args.end,
        tickers=selection.tickers,
        alpha158_factors_db=args.alpha158_factors_db,
        qlib_provider_uri=args.qlib_provider_uri,
        news_sentiment_backend=news_sentiment_backend,
        news_sentiment_cache_path=args.news_sentiment_cache,
        include_news=not args.no_news,
        strict_news=args.strict_news,
        technical_rating=TradingViewRating(LoaderOHLCVPanel(loader)),
    )
    result = run_expert_loop(
        loader,
        config,
        build_experts(
            args.expert_backend,
            model=args.expert_model,
            model_revision=args.expert_model_revision,
        ),
        market_state_config=MarketStateConfig(
            vol_window=args.vol_window,
            percentile_window=args.percentile_window,
            trend_window=args.trend_window,
            trend_confirm_days=args.trend_confirm_days,
        ),
        reflect_every=args.reflect_every,
        reflection_window=args.reflection_window,
        reflection_min_support=args.reflection_min_support,
        exit_threshold=args.exit_threshold,
        allow_reflection=args.allow_reflection,
        signal_sources=signal_sources,
        run_id=args.run_id,
    )
    payload = {
        "mode": "backtest",
        "system": SYSTEM_IDENTITY,
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "data_root": data_root,
        "manifest": {
            "mode": "backtest",
            "system": SYSTEM_IDENTITY,
            "schema_version": REPORT_SCHEMA_VERSION,
            "launcher": "trading-agent backtest",
            "data_root": data_root,
            "universe_label": selection.label,
            "tickers": list(selection.tickers),
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "run_id": args.run_id,
            "seed": int(args.seed),
            "reflect_every": int(args.reflect_every),
            "reflection_window": int(args.reflection_window),
            "exit_threshold": args.exit_threshold,
            "reflection_min_support": int(args.reflection_min_support),
            "market_state": {
                "vol_window": int(args.vol_window),
                "percentile_window": int(args.percentile_window),
            },
            "preload": {
                "calendar_days": int(args.preload_calendar_days),
                "date_from": args.preload_start.isoformat(),
            },
            "execution": {
                # Effective BacktestConfig values: when the bps model is active
                # the commission parameters are zeroed because the adapter does
                # not charge them (bps replaces per-share commission).
                "initial_cash": float(config.initial_cash),
                "commission_per_share": float(config.commission_per_share),
                "min_commission": float(config.min_commission),
                "max_commission_rate": float(config.max_commission_rate),
                "slippage_perc": float(config.slippage_perc),
                "execution_cost_bps_per_side": float(config.execution_cost_bps_per_side),
                "liquidity_cap_pct": float(config.liquidity_cap_pct),
            },
            "cost_protocol": {
                "model": (
                    "bps_per_side_on_traded_notional"
                    if config.execution_cost_bps_per_side > 0
                    else "per_share_commission"
                ),
                # Same hardcoding as the holdout launcher had: true only while a
                # positive bps rate is charged, since that is what displaces the
                # per-share schedule.
                "bps_replaces_per_share_commission": (
                    config.execution_cost_bps_per_side > 0
                ),
                "headline_bps_per_side": float(args.headline_cost_bps_per_side),
                "sensitivity_bps_per_side": float(args.sensitivity_cost_bps_per_side),
                "arm_identical": True,
            },
            "expert_backend": args.expert_backend,
            "expert_model": args.expert_model,
            "expert_model_revision": args.expert_model_revision,
            **expert_backend_metadata(
                args.expert_backend,
                model=args.expert_model,
                model_revision=args.expert_model_revision,
            ),
            "alpha158_factors_db": args.alpha158_factors_db,
            "qlib_provider_uri": args.qlib_provider_uri,
            "news_enabled": not bool(args.no_news),
            "strict_news": bool(args.strict_news),
            "news_sentiment_cache": args.news_sentiment_cache,
            "news_sentiment_model": args.news_sentiment_model if args.news_sentiment_cache else None,
            "allow_reflection": bool(args.allow_reflection),
        },
        "universe": {
            "label": selection.label,
            "tickers": list(selection.tickers),
        },
        "config_fingerprint": result.backtest.config_fingerprint,
        "summary": dict(result.backtest.summary),
        "n_dates": result.n_dates,
        "n_active_dates": result.n_active_dates,
        "n_long_days": result.n_long_days,
        "participation_rate": result.participation_rate,
        "market_state_coverage": _state_counts(result),
        "per_name_episode_counts": _episode_counts_by_state_action(result),
        "episode_count_before_after": _episode_count_before_after(result),
        "expert_error_by_state": _error_summary_from_result(result),
        "per_state_metrics": _per_state_metrics(result),
        "n_reflection_rounds": result.n_reflection_rounds,
        "n_proposals": result.n_proposals,
        "n_applied": result.n_applied,
        "n_learned": result.n_learned,
        "reflection_trial_audit": {
            "round_count": len(result.reflection_trial_records),
            "proposal_count": result.n_proposals,
            "applied_count": result.n_applied,
            "rounds": result.reflection_trial_records,
        },
        "research_claims": {
            "run_role": "exploratory_backtest",
            "evidence_scope": "in_sample_only",
            "final_evidence_mode": "holdout",
            "reflection_learning": {
                "allow_reflection": bool(args.allow_reflection),
                "n_reflection_rounds": result.n_reflection_rounds,
                "n_learned": result.n_learned,
            },
        },
        "n_equity_points": len(result.backtest.equity_curve),
        "n_fills": len(result.backtest.fills),
        "n_rejected": len(result.backtest.rejected),
        "allow_reflection": bool(args.allow_reflection),
    }
    _write_json(args.output, payload)
    print(f"WROTE {args.output}", flush=True)
    return 0


def run_holdout_script(argv: Sequence[str]) -> int:
    from trading_agent.experiments.expert_weight_holdout_launcher import main as holdout_main

    return holdout_main(argv)


def run_holdout_mode(args: argparse.Namespace) -> int:
    _require_holdout_date_window(args)
    _require_experiment_parameters(args)
    _require_non_negative_args(args, ("embargo_days",))
    _require_expert_model_identity(args)
    forwarded = [
        "--data-root", args.data_root,
        "--start", args.start.isoformat(),
        "--boundary", args.boundary.isoformat(),
        "--end", args.end.isoformat(),
        "--names", str(args.names),
        "--embargo-days", str(args.embargo_days),
        "--reflect-every", str(args.reflect_every),
        "--reflection-window", str(args.reflection_window),
        "--reflection-min-support", str(args.reflection_min_support),
        *(["--decision-thresholds", args.decision_thresholds]
          if args.decision_thresholds is not None else []),
        "--exit-threshold", str(args.exit_threshold),
        "--initial-cash", str(args.initial_cash),
        "--commission-per-share", str(args.commission_per_share),
        "--min-commission", str(args.min_commission),
        "--max-commission-rate", str(args.max_commission_rate),
        "--slippage-perc", str(args.slippage_perc),
        "--headline-cost-bps-per-side", str(args.headline_cost_bps_per_side),
        "--sensitivity-cost-bps-per-side", str(args.sensitivity_cost_bps_per_side),
        "--preload-calendar-days", str(args.preload_calendar_days),
        "--liquidity-cap-pct", str(args.liquidity_cap_pct),
        "--seed", str(args.seed),
        "--vol-window", str(args.vol_window),
        "--percentile-window", str(args.percentile_window),
        "--trend-window", str(args.trend_window),
        "--trend-confirm-days", str(args.trend_confirm_days),
        "--launcher", "trading-agent holdout",
        "--output", args.output,
    ]
    if args.validation_boundary:
        forwarded.extend(["--validation-boundary", args.validation_boundary.isoformat()])
    if args.alpha158_factors_db:
        forwarded.extend(["--alpha158-factors-db", args.alpha158_factors_db])
    if args.qlib_provider_uri:
        forwarded.extend(["--qlib-provider-uri", args.qlib_provider_uri])
    if args.no_news:
        forwarded.append("--no-news")
    if args.strict_news:
        forwarded.append("--strict-news")
    if args.news_sentiment_cache:
        forwarded.extend(["--news-sentiment-cache", args.news_sentiment_cache])
        forwarded.extend(["--news-sentiment-model", args.news_sentiment_model])
    if args.memory_output:
        forwarded.extend(["--memory-output", args.memory_output])
    forwarded.extend(["--expert-backend", args.expert_backend])
    forwarded.extend(["--expert-model", args.expert_model])
    forwarded.extend(["--expert-model-revision", args.expert_model_revision])
    return run_holdout_script(forwarded)


def run_paper_mode(args: argparse.Namespace) -> int:
    if args.dry_run and args.execute_orders:
        raise ValueError("--dry-run cannot be combined with --execute-orders")
    _require_connection_smoke_args(args)
    _require_paper_connection_args(args)
    _require_paper_execution_port(args)
    symbols = [] if args.connection_smoke and not args.symbols else _parse_symbols(args.symbols)
    order_intents = [parse_order_spec(spec) for spec in args.order]
    _require_execution_order_source(args)
    _require_resolved_execution_quantities(args, order_intents)
    _require_paper_auto_decide_args(args)
    order_source = _paper_order_source(args)
    gateway_config = IBGatewayConfig(host=args.host, port=args.port, client_id=args.client_id)
    non_paper_port_override = bool(
        args.execute_orders and _uses_non_paper_port(args) and args.allow_non_paper_port
    )
    memory_info = None
    memory_entries = []
    memory_store = None
    if args.require_memory and not args.auto_decide:
        raise ValueError("--require-memory requires --auto-decide")
    if args.require_memory and not args.memory_snapshot:
        raise ValueError("--require-memory requires --memory-snapshot")
    if args.memory_snapshot:
        memory_payload = load_snapshot_payload(args.memory_snapshot)
        _require_frozen_memory_snapshot(memory_payload)
        memory_store = ExpertScorecardStore.from_snapshot(memory_payload)
        memory_entries = snapshot_entries(memory_payload)
        if args.require_memory and memory_store.learned_state_count() == 0:
            raise ValueError("--require-memory requires at least one learned market state")
        memory_info = _memory_snapshot_info(args.memory_snapshot, memory_payload, memory_entries)
    from data_collection.config import Settings

    if args.dry_run:
        latest_frames = {}
        latest_bars = {}
        expert_views = {}
        aggregation_payload = None
        memory_applied = None
        deployment_fraction = 1.0
        gateway_connected = False
        paper_cash = float(args.paper_cash)
        cash_source = "manual_paper_cash"
        if args.auto_decide:
            if connect_ib is None or request_daily_bars is None:
                raise RuntimeError("IB connector is unavailable in this environment.")
            settings = Settings(
                ib_host=args.host,
                ib_port=args.port,
                ib_client_id=args.client_id,
                data_dir=Path("data"),
                reports_dir=Path("reports"),
                rapidapi_key=None,
                rapidapi_host="seeking-alpha-finance.p.rapidapi.com",
            )
            ib = connect_ib(settings)
            gateway_connected = True
            try:
                if args.use_account_cash:
                    paper_cash = account_cash_from_summary(ib.accountSummary())
                    cash_source = "ib_account_available_funds"
                history_days = _paper_history_duration_days(args)
                duration = _paper_history_duration_arg(history_days)
                for symbol in symbols:
                    frame = request_daily_bars(ib, symbol, duration=duration)
                    if frame is None or getattr(frame, "empty", True):
                        raise ValueError(
                            f"IB returned no bars for {symbol} at duration {duration!r}. "
                            "Refusing to continue: an empty frame would make the market "
                            "state fall back to the traded stock's own history."
                        )
                    latest_frames[symbol] = frame
                    latest_bars[symbol] = _latest_bar_report(frame)
                _reject_partial_session(latest_frames, symbols)
                proxy_frame = _fetch_market_proxy_frame(ib, args, history_days)
                # Read before the disconnect below, so the preview shows the orders the
                # live run would actually place against the current account.
                held = _held_quantities(ib, symbols)
                # A configured proxy that returns nothing is not a soft failure on a
                # deployment run: the classifier would fall back to the traded stock's
                # own history and label the state from the instrument being traded.
                # Smoke and dry runs keep the old fail-soft behaviour.
                if (
                    args.require_memory
                    and args.paper_market_state == "auto"
                    and (getattr(args, "market_proxy_symbol", "") or "").strip()
                    and proxy_frame is None
                ):
                    raise ValueError(
                        f"market proxy {args.market_proxy_symbol!r} returned no bars. "
                        "Refusing to continue: the market state would be computed from "
                        "the traded stock's own history."
                    )
            finally:
                ib.disconnect()
            observation = _paper_observation(symbols, latest_frames)
            market_state, market_state_source = _resolve_paper_market_state(
                args, observation, latest_frames, proxy_frame=proxy_frame
            )
            if args.require_memory:
                _require_applicable_memory(market_state, memory_store)
            expert_views = _paper_views(observation, market_state,
                                        committee_read=_committee_read_path(args))
            memory_applied, aggregation_payload, deployment_fraction = _memory_application(
                market_state,
                memory_store,
                expert_views,
                symbols,
                held=held,
            )
            long_symbols = _long_symbols_from_aggregation(symbols, aggregation_payload)
            order_intents.extend(auto_decide_order_intents(
                long_symbols,
                latest_frames,
                paper_cash=paper_cash,
                deployment_fraction=deployment_fraction,
                max_order_value=args.max_order_value,
                held=held,
            ))
        payload = build_paper_manifest(
            config=gateway_config,
            symbols=symbols,
            order_intents=order_intents,
            dry_run=True,
            execute_orders=args.execute_orders,
            gateway_connected=gateway_connected,
            latest_bars=latest_bars,
            orders_submitted=0,
            require_memory=args.require_memory,
            connection_smoke=args.connection_smoke,
            non_paper_port_override=non_paper_port_override,
            launcher="trading-agent paper",
        )
        payload["order_source"] = order_source
        payload["non_paper_port_override"] = non_paper_port_override
        if args.auto_decide:
            # Name the seat the views actually came from. The label was hardcoded to
            # the static backend and stayed that way after --committee-read was added,
            # so a manifest recording a real committee decision read as though it had
            # come from stubs.
            _read = _committee_read_path(args)
            payload["decision_source"] = (
                f"committee_read:{_read.name}" if _read is not None else "three_expert_static"
            )
            payload["market_state"] = market_state
            payload["market_state_source"] = market_state_source
            payload["cash_basis"] = {
                "source": cash_source,
                "amount": float(paper_cash),
            }
            payload["expert_views"] = _legacy_expert_views_payload(expert_views)
            payload["expert_views_by_ticker"] = _views_by_ticker_payload(expert_views)
            payload["aggregation"] = aggregation_payload
            if args.max_order_value is not None:
                payload["risk_limits"] = {"max_order_value": float(args.max_order_value)}
            if memory_applied is not None:
                payload["memory_applied"] = memory_applied
        if memory_info is not None:
            payload["memory_snapshot"] = memory_info
        payload["safety_note"] = (
            "Dry run only; no orders submitted. "
            "Auto-decision may connect for market data/account cash."
            if args.auto_decide
            else "Dry run only; no orders submitted and no IB connection opened."
        )
        _write_json(args.output, payload)
        if gateway_connected:
            print(
                f"DRY RUN: connected to IB Gateway {args.host}:{args.port} "
                f"clientId={args.client_id} for market data only.",
                flush=True,
            )
        else:
            print(
                f"DRY RUN: would connect to IB Gateway {args.host}:{args.port} "
                f"clientId={args.client_id}",
                flush=True,
            )
        print(f"WROTE {args.output}", flush=True)
        return 0

    if connect_ib is None or request_daily_bars is None:
        raise RuntimeError("IB connector is unavailable in this environment.")

    settings = Settings(
        ib_host=args.host,
        ib_port=args.port,
        ib_client_id=args.client_id,
        data_dir=Path("data"),
        reports_dir=Path("reports"),
        rapidapi_key=None,
        rapidapi_host="seeking-alpha-finance.p.rapidapi.com",
    )
    try:
        ib = connect_ib(settings)
    except Exception as exc:
        if not args.connection_smoke:
            raise
        payload = build_paper_manifest(
            config=gateway_config,
            symbols=symbols,
            order_intents=[],
            dry_run=False,
            execute_orders=False,
            gateway_connected=False,
            orders_submitted=0,
            require_memory=args.require_memory,
            connection_smoke=args.connection_smoke,
            non_paper_port_override=non_paper_port_override,
            launcher="trading-agent paper",
        )
        payload["connection_smoke"] = True
        payload["smoke_status"] = "failed"
        payload["order_source"] = order_source
        payload["error"] = str(exc)
        payload["safety_note"] = "Connection smoke failed before any market data request or order submission."
        _write_json(args.output, payload)
        raise RuntimeError(
            "IB Gateway paper connection smoke failed for "
            f"{args.host}:{args.port}. Start IB Gateway/TWS paper, enable API access, "
            "and confirm the paper API port is 4002 or 7497."
        ) from exc
    try:
        bars = {}
        latest_frames = {}
        expert_views = {}
        aggregation_payload = None
        memory_applied = None
        deployment_fraction = 1.0
        paper_cash = float(args.paper_cash)
        cash_source = "manual_paper_cash"
        account_summary_rows = None
        if args.use_account_cash:
            account_summary = ib.accountSummary()
            account_summary_rows = len(account_summary)
            paper_cash = account_cash_from_summary(account_summary)
            cash_source = "ib_account_available_funds"
        elif not args.execute_orders:
            account_summary_rows = len(ib.accountSummary())
        if args.auto_decide:
            history_days = _paper_history_duration_days(args)
            # A day-unit duration past a year returns an EMPTY frame instead of an
            # error (see _paper_history_duration_arg). The frozen windows put this at
            # 792 days, which is exactly the case that fails, and the connected branch
            # was asking for it directly while the dry-run branch converted correctly.
            duration = _paper_history_duration_arg(history_days)
            for symbol in symbols:
                frame = request_daily_bars(ib, symbol, duration=duration)
                if frame is None or getattr(frame, "empty", True):
                    raise ValueError(
                        f"IB returned no bars for {symbol} at duration {duration!r}. "
                        "Refusing to continue: an empty frame would make the market "
                        "state fall back to the traded stock's own history."
                    )
                latest_frames[symbol] = frame
                bars[symbol] = _latest_bar_report(frame)
            _reject_partial_session(latest_frames, symbols)
            proxy_frame = _fetch_market_proxy_frame(ib, args, history_days)
            # The dry-run branch refuses a deployment run whose proxy came back empty.
            # This branch submits orders, so it cannot be the more permissive of the
            # two: without the proxy every stock silently takes the up_calm default,
            # and a state-armed stock never reaches the cell it was deployed for.
            if (
                args.require_memory
                and args.paper_market_state == "auto"
                and (getattr(args, "market_proxy_symbol", "") or "").strip()
                and proxy_frame is None
            ):
                raise ValueError(
                    f"market proxy {args.market_proxy_symbol!r} returned no bars. "
                    "Refusing to continue: the market state would fall back to a "
                    "default and no state-conditioned cell could be reached."
                )
            observation = _paper_observation(symbols, latest_frames)
            market_state, market_state_source = _resolve_paper_market_state(
                args, observation, latest_frames, proxy_frame=proxy_frame
            )
            if args.require_memory:
                _require_applicable_memory(market_state, memory_store)
            expert_views = _paper_views(observation, market_state,
                                        committee_read=_committee_read_path(args))
            # Read what the account already holds BEFORE aggregating. The position is
            # needed twice: the action rule maintains a held stock at a boundary score,
            # and the order builder turns a repeated long into a hold and a stock the
            # committee has turned flat on into an exit. Without it the bridge only ever
            # bought, which over a ten-day run bought the same stock every day and never
            # closed anything.
            held = _held_quantities(ib, symbols)
            memory_applied, aggregation_payload, deployment_fraction = _memory_application(
                market_state,
                memory_store,
                expert_views,
                symbols,
                held=held,
            )
            long_symbols = _long_symbols_from_aggregation(symbols, aggregation_payload)
            order_intents.extend(auto_decide_order_intents(
                long_symbols,
                latest_frames,
                paper_cash=paper_cash,
                deployment_fraction=deployment_fraction,
                max_order_value=args.max_order_value,
                held=held,
            ))
        order_previews = build_order_previews(order_intents, allow_all_available_sell=True)
        submission_reports = []
        if args.execute_orders:
            submission_reports = submit_order_previews(ib, order_previews)
        payload = build_paper_manifest(
            config=gateway_config,
            symbols=symbols,
            order_intents=order_intents,
            dry_run=False,
            execute_orders=args.execute_orders,
            gateway_connected=True,
            latest_bars=bars,
            orders_submitted=len(submission_reports),
            require_memory=args.require_memory,
            connection_smoke=args.connection_smoke,
            non_paper_port_override=non_paper_port_override,
            launcher="trading-agent paper",
        )
        payload["order_source"] = order_source
        payload["non_paper_port_override"] = non_paper_port_override
        payload["submission_reports"] = submission_reports
        if not args.execute_orders:
            payload["connection_smoke"] = bool(args.connection_smoke)
            payload["gateway_smoke"] = {
                "account_summary_checked": True,
                "account_summary_rows": int(account_summary_rows or 0),
            }
            if args.connection_smoke:
                payload["smoke_status"] = "passed"
        if args.auto_decide:
            # Name the seat the views actually came from. The label was hardcoded to
            # the static backend and stayed that way after --committee-read was added,
            # so a manifest recording a real committee decision read as though it had
            # come from stubs.
            _read = _committee_read_path(args)
            payload["decision_source"] = (
                f"committee_read:{_read.name}" if _read is not None else "three_expert_static"
            )
            payload["market_state"] = market_state
            payload["market_state_source"] = market_state_source
            payload["cash_basis"] = {
                "source": cash_source,
                "amount": float(paper_cash),
            }
            payload["expert_views"] = _legacy_expert_views_payload(expert_views)
            payload["expert_views_by_ticker"] = _views_by_ticker_payload(expert_views)
            payload["aggregation"] = aggregation_payload
            if args.max_order_value is not None:
                payload["risk_limits"] = {"max_order_value": float(args.max_order_value)}
            if memory_applied is not None:
                payload["memory_applied"] = memory_applied
        if memory_info is not None:
            payload["memory_snapshot"] = memory_info
        payload["safety_note"] = (
            "Submitted paper orders through IB Gateway."
            if args.execute_orders
            else "Gateway/data smoke complete; no orders submitted without --execute-orders."
        )
    finally:
        ib.disconnect()
    _write_json(args.output, payload)
    if args.connection_smoke:
        rows = payload["gateway_smoke"]["account_summary_rows"]
        print(f"CONNECTION SMOKE PASSED account_summary_rows={rows}", flush=True)
    print(f"WROTE {args.output}", flush=True)
    return 0


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission-per-share", type=float, default=0.0049)
    parser.add_argument("--min-commission", type=float, default=0.99)
    parser.add_argument("--max-commission-rate", type=float, default=0.01)
    parser.add_argument("--slippage-perc", type=float, default=0.0005)
    parser.add_argument("--headline-cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--sensitivity-cost-bps-per-side", type=float, default=25.0)
    parser.add_argument("--preload-calendar-days", type=int, default=900)
    parser.add_argument("--liquidity-cap-pct", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="Run the regime-conditioned expert reflection trading system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Modes:
              backtest: run historical committee backtests.
              holdout: train memory, freeze it, and test it on a later window.
              paper: connect to IB Gateway/TWS paper trading.

            Example:
              trading-agent holdout --data-root <FINSABER_PARQUET_ROOT> --start 2020-01-01 --validation-boundary 2021-07-01 --boundary 2022-01-01 --end 2023-01-01 --memory-output reports/expert_weight_memory.json
            """
        ).strip(),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"trading-agent {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    backtest = subparsers.add_parser("backtest", help="Run a deterministic historical backtest.")
    backtest.add_argument("--data-root", required=True)
    backtest.add_argument("--start", type=_parse_date, required=True)
    backtest.add_argument("--end", type=_parse_date, required=True)
    backtest.add_argument("--names", type=int, default=10)
    backtest.add_argument("--run-id", default="cli-backtest")
    backtest.add_argument("--reflect-every", type=int, default=21)
    backtest.add_argument(
        "--exit-threshold", type=_parse_exit_threshold, default=0.0,
        help="HOLD semantics (default 0): enter long at score >= 0.5, hold while "
        "score stays above this value, exit at or below it. 'none' disables.",
    )
    backtest.add_argument("--reflection-window", type=int, default=63)
    backtest.add_argument("--reflection-min-support", type=int, default=5)
    backtest.add_argument("--vol-window", type=int, default=21)
    backtest.add_argument("--percentile-window", type=int, default=504)
    backtest.add_argument("--trend-window", type=int, default=200)
    backtest.add_argument("--trend-confirm-days", type=int, default=5)
    backtest.add_argument("--alpha158-factors-db", default=None)
    backtest.add_argument("--qlib-provider-uri", default=None)
    backtest.add_argument("--no-news", action="store_true")
    backtest.add_argument("--strict-news", action="store_true")
    backtest.add_argument("--expert-backend", choices=SUPPORTED_EXPERT_BACKENDS, default="static")
    backtest.add_argument("--expert-model", default=None)
    backtest.add_argument("--expert-model-revision", default=None)
    backtest.add_argument("--news-sentiment-cache", default=None)
    backtest.add_argument("--news-sentiment-model", default="gpt-4.1-mini")
    backtest.add_argument("--allow-reflection", action="store_true")
    backtest.add_argument("--output", default="reports/trading_agent_backtest.json")
    _add_execution_args(backtest)
    backtest.set_defaults(func=run_backtest_mode)

    holdout = subparsers.add_parser("holdout", help="Train reflection memory, freeze it, then test on holdout.")
    holdout.add_argument("--data-root", required=True)
    holdout.add_argument("--start", type=_parse_date, required=True)
    holdout.add_argument("--validation-boundary", type=_parse_date, default=None)
    holdout.add_argument("--boundary", type=_parse_date, required=True)
    holdout.add_argument("--end", type=_parse_date, required=True)
    holdout.add_argument("--names", type=int, default=10)
    holdout.add_argument("--embargo-days", type=int, default=5)
    holdout.add_argument("--reflect-every", type=int, default=21)
    holdout.add_argument(
        "--exit-threshold", default="0",
        help="HOLD semantics (default 0): hold a long while the score stays above "
        "this value; exit at or below it. 'none' disables (daily re-vote).",
    )
    holdout.add_argument("--reflection-window", type=int, default=63)
    holdout.add_argument("--reflection-min-support", type=int, default=5)
    # None -> not forwarded, so the launcher's FROZEN default (theta=0, single
    # point since 2026-07-17) is the single source of truth. Passing a grid here
    # is a diagnostic override only.
    holdout.add_argument("--decision-thresholds", default=None)
    holdout.add_argument("--vol-window", type=int, default=21)
    holdout.add_argument("--percentile-window", type=int, default=504)
    holdout.add_argument("--trend-window", type=int, default=200)
    holdout.add_argument("--trend-confirm-days", type=int, default=5)
    holdout.add_argument("--alpha158-factors-db", default=None)
    holdout.add_argument("--qlib-provider-uri", default=None)
    holdout.add_argument("--no-news", action="store_true")
    holdout.add_argument("--strict-news", action="store_true")
    holdout.add_argument("--expert-backend", choices=SUPPORTED_EXPERT_BACKENDS, default="static")
    holdout.add_argument("--expert-model", default=None)
    holdout.add_argument("--expert-model-revision", default=None)
    holdout.add_argument("--news-sentiment-cache", default=None)
    holdout.add_argument("--news-sentiment-model", default="gpt-4.1-mini")
    holdout.add_argument("--output", default="reports/expert_weight_holdout")
    holdout.add_argument("--memory-output", default=None)
    _add_execution_args(holdout)
    holdout.set_defaults(func=run_holdout_mode)

    paper = subparsers.add_parser("paper", help="Start IB Gateway paper-trading bridge.")
    paper.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols, e.g. AAPL,MSFT. Optional only with --connection-smoke.",
    )
    paper.add_argument("--host", default="127.0.0.1")
    paper.add_argument("--port", type=int, default=4002)
    paper.add_argument("--client-id", type=int, default=1)
    paper.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help=(
            "Minimum IB daily-bar history to request. Auto market-state mode may "
            "increase this to cover the classifier windows."
        ),
    )
    paper.add_argument("--paper-vol-window", type=int, default=5)
    paper.add_argument("--paper-percentile-window", type=int, default=20)
    paper.add_argument(
        "--paper-trend-window", type=int, default=200,
        help="Trailing SMA window for the market_state trend axis (frozen system: 200).",
    )
    paper.add_argument(
        "--paper-trend-confirm-days", type=int, default=5,
        help="Consecutive closes on the new side before the trend flips (frozen system: 5).",
    )
    paper.add_argument(
        "--market-proxy-symbol",
        default="SPY",
        help=(
            "External symbol whose bars define market_state (default SPY), so a "
            "single-name run never conditions on its own trend. Empty string falls "
            "back to the traded symbols' own bars."
        ),
    )
    paper.add_argument(
        "--auto-decide",
        action="store_true",
        help="Generate paper order previews from the system's three-expert decision path.",
    )
    paper.add_argument("--paper-cash", type=float, default=10_000.0)
    paper.add_argument(
        "--max-order-value",
        type=float,
        default=None,
        help="Optional per-symbol notional cap for auto-decision order previews.",
    )
    paper.add_argument(
        "--use-account-cash",
        action="store_true",
        help="Use IB account AvailableFunds as the paper cash budget for auto-decision.",
    )
    paper.add_argument("--memory-snapshot", default=None)
    paper.add_argument(
        "--committee-read",
        default=None,
        metavar="PATH",
        help="Replay a recorded committee read (a file, or a directory whose newest "
             "file is taken) as this run's expert views. Without it the views come "
             "from the static backend, which always answers long: on 2026-08-03 that "
             "produced BUY previews on all twenty stocks including the four the real "
             "committee had called bearish.",
    )
    paper.add_argument(
        "--require-memory",
        action="store_true",
        help="Reject auto-decision deployment unless a frozen memory snapshot is supplied.",
    )
    paper.add_argument(
        "--paper-market-state",
        default="auto",
        choices=("auto", "up_calm", "up_stress", "down_calm", "down_stress"),
        help=(
            "Market state used when applying a frozen memory snapshot in paper mode. "
            "Default 'auto' classifies latest IB daily bars; explicit states are manual overrides."
        ),
    )
    paper.add_argument(
        "--order",
        action="append",
        default=[],
        help="Optional order preview in SYMBOL:buy|sell:QTY form; repeat for multiple orders.",
    )
    paper.add_argument("--dry-run", action="store_true", default=False)
    paper.add_argument(
        "--connection-smoke",
        action="store_true",
        help=(
            "Connect to IB Gateway, read account summary, disconnect, and submit no orders; "
            "cannot be combined with dry-run, auto-decision, manual orders, or execution."
        ),
    )
    paper.add_argument(
        "--execute-orders",
        action="store_true",
        help="Submit market orders to the connected IB paper account; cannot be combined with --dry-run.",
    )
    paper.add_argument(
        "--allow-non-paper-port",
        action="store_true",
        help=(
            "Allow --execute-orders on a port outside the common IB paper "
            "ports 4002 and 7497."
        ),
    )
    paper.add_argument("--output", default="reports/trading_agent_paper.json")
    paper.set_defaults(func=run_paper_mode)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Load secrets from a gitignored .env (OPENAI_API_KEY etc.) before any backend
    # runs, matching the holdout launcher. setdefault, so an exported var still wins.
    try:
        from data_collection.config import load_env_file
        load_env_file()
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
