"""Phase 3 — minimal end-to-end smoke runner.

A plain Python orchestration loop (NO LangGraph). Each cycle:
    open bar (next_open fills) -> observation -> signals -> target positions ->
    order intents -> submit orders (next-bar) -> persist -> advance.
Deterministic given (config.seed, config), produces an equity curve via BacktestResult.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, Callable, Mapping, Optional, Sequence

from trading_agent.contracts import (
    ALL_AVAILABLE,
    AlphaEngine,
    BacktestConfig,
    BacktestResult,
    Observation,
    Order,
    OrderIntent,
    PortfolioEngine,
    RiskEngine,
    is_abstention_view,
)
from trading_agent.execution.finsaber_adapter import FinsaberBacktestAdapter
from trading_agent.expert_context import ExpertSignalContextBuilder, ExpertSignalSources
from trading_agent.experts.base_expert import is_safe_default_view, safe_default_reason
from trading_agent.persistence import RunLogger


def run_backtest(
    loader,
    config: BacktestConfig,
    alpha: AlphaEngine,
    risk: RiskEngine,
    portfolio: PortfolioEngine,
    run_id: Optional[str] = None,
    logger: Optional[RunLogger] = None,
) -> BacktestResult:
    # Determinism hygiene (no stochastic components in the baseline, but seed anyway).
    random.seed(config.seed)
    try:  # numpy is a transitive dep of the loaders; seed if present.
        import numpy as np

        np.random.seed(config.seed)
    except Exception:  # pragma: no cover
        pass

    adapter = FinsaberBacktestAdapter(loader, config)
    run_id = run_id or f"run-{config.fingerprint()[:12]}"
    order_seq = 0

    for decision_date in adapter.dates:
        adapter.open_bar(decision_date)
        account = adapter.get_account()
        positions = adapter.get_positions()
        observation = adapter.get_observation(decision_date)

        signals = alpha.score(observation)
        targets = risk.size(signals, account, observation)
        intents = portfolio.build_orders(targets, account, positions, observation)

        decision_id = f"{run_id}:{decision_date.isoformat()}"
        orders: list[Order] = []
        for intent in intents:
            order_seq += 1
            bar = observation.bars.get(intent.ticker)
            ref_price = None
            if bar is not None:
                ref_price = intent.limit_price if intent.limit_price is not None else bar.adjusted_close
            order = Order(
                order_id=f"{decision_id}:{order_seq}",
                decision_id=decision_id,
                signal_date=decision_date,
                ticker=intent.ticker,
                side=intent.side,
                quantity=intent.quantity,
                reference_price=ref_price,
            )
            adapter.submit_order(order)
            orders.append(order)

        if logger is not None:
            logger.log(
                "decision",
                run_id,
                decision_id,
                {
                    "decision_date": observation.decision_date,
                    "decision_time": observation.decision_time,
                    "availability_time": observation.availability_time,
                    "config_fingerprint": config.fingerprint(),
                    "account": account,
                    "positions": list(positions.positions),
                    "signals": list(signals),
                    "targets": list(targets),
                    "orders": orders,
                },
            )

        adapter.advance()

    adapter.finalize()
    result = adapter.result(run_id)

    if logger is not None:
        logger.log(
            "summary",
            run_id,
            "summary",
            {
                "config_fingerprint": result.config_fingerprint,
                "summary": dict(result.summary),
                "equity_curve": [(d.isoformat(), e) for d, e in result.equity_curve],
                "final_account": result.final_account,
            },
        )

    return result


# ======================================================================================
# End-to-end expert loop (Phase 1 wiring)
#
#   observation -> market_state -> experts -> get_weights(market_state)
#   -> aggregate -> deterministic risk layer -> execute -> attribution
#   -> periodic reflection -> learned memory stored and reused.
#
# Goal of this wiring is only that the loop runs end to end once and returns a
# populated result. The risk layer here is intentionally simple (the LLM/aggregation
# emits only a direction; this layer owns positions). Evaluation on an untouched
# holdout set is the evidence step; the main path uses train-reflect-freeze-test.
# ======================================================================================


@dataclass
class ExpertLoopResult:
    """Populated result of one end-to-end run."""

    run_id: str
    backtest: BacktestResult
    n_dates: int
    n_active_dates: int
    market_states: list[tuple]              # (Date, market_state)
    n_long_days: int
    participation_rate: float
    n_reflection_rounds: int
    n_proposals: int
    n_applied: int
    n_learned: int
    decision_threshold: float = 0.5
    confidence_diagnostics: dict = field(default_factory=dict)
    expert_view_records: dict = field(default_factory=dict)
    equal_weight_score_records: list = field(default_factory=list)
    context_records: list = field(default_factory=list)
    episodes: list = field(default_factory=list)  # list[AttributedEpisode]
    reflection_trial_records: list = field(default_factory=list)


def _confidence_histogram(values: Sequence[float]) -> dict[str, int]:
    bins = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    for value in values:
        v = max(0.0, min(1.0, float(value)))
        if v < 0.2:
            bins["0.0-0.2"] += 1
        elif v < 0.4:
            bins["0.2-0.4"] += 1
        elif v < 0.6:
            bins["0.4-0.6"] += 1
        elif v < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def _confidence_summary(records: dict[str, list[dict]]) -> dict[str, dict]:
    out = {}
    for expert_id, rows in sorted(records.items()):
        values = [float(row["confidence"]) for row in rows]
        safe_rows = [row for row in rows if row["safe_default"]]
        reason_counts = {"data_unavailable": 0, "parse_error": 0, "unknown": 0}
        for row in safe_rows:
            reason = row.get("safe_default_reason") or "unknown"
            reason_counts[reason if reason in reason_counts else "unknown"] += 1
        if values:
            mean = sum(values) / len(values)
            minimum = min(values)
            maximum = max(values)
        else:
            mean = minimum = maximum = 0.0
        real_count = len(rows) - len(safe_rows)
        out[expert_id] = {
            "n": len(rows),
            "mean": round(mean, 6),
            "min": round(minimum, 6),
            "max": round(maximum, 6),
            "histogram": _confidence_histogram(values),
            "long_count": sum(1 for row in rows if row["direction"] == "long"),
            "flat_count": sum(1 for row in rows if row["direction"] == "flat"),
            "safe_default_count": len(safe_rows),
            "safe_default_reasons": reason_counts,
            "real_judgment_count": real_count,
            "real_judgment_rate": (real_count / len(rows)) if rows else 0.0,
        }
    return out


#: Family E2 (prereg Amendment 3): rebalance only when the target fraction moves by
#: more than this share of the name's sleeve. Registered a priori, not tuned.
REBALANCE_BAND: float = 0.10


def _direction_to_intents(
    directions_by_ticker: Mapping[str, str],
    observation: Observation,
    account,
    positions,
    *,
    fractions_by_ticker: Mapping[str, float] | None = None,
    rebalance_band: float = REBALANCE_BAND,
) -> list[OrderIntent]:
    """Deterministic risk layer: equal-weight passing tickers; sell failed names.

    ``fractions_by_ticker`` (position_sizing="proportional", Family E2) scales each
    long name's exposure to a target fraction of its sleeve instead of full
    investment; entry/exit dates are decided by ``directions_by_ticker`` exactly as
    in the binary layer, so only exposure while long differs.
    """
    intents: list[OrderIntent] = []
    held = positions.by_ticker()
    long_tickers = [
        tk for tk in observation.tickers
        if directions_by_ticker.get(tk) == "long" and tk in observation.bars
    ]
    if long_tickers:
        for tk, pos in held.items():
            if tk not in long_tickers and pos.quantity > 0:
                intents.append(OrderIntent(ticker=tk, side="sell", quantity=ALL_AVAILABLE))
        n = max(1, len(long_tickers))
        if fractions_by_ticker is None:
            budget_each = account.cash / n
            for tk in long_tickers:
                if tk in held:
                    continue
                bar = observation.bars.get(tk)
                if bar is None or bar.adjusted_close <= 0:
                    continue
                qty = int(budget_each / bar.adjusted_close)
                if qty > 0:
                    intents.append(OrderIntent(ticker=tk, side="buy", quantity=qty))
        else:
            held_value = sum(
                held[tk].quantity * observation.bars[tk].adjusted_close
                for tk in long_tickers
                if tk in held and observation.bars.get(tk) is not None
            )
            sleeve_each = (account.cash + held_value) / n
            for tk in long_tickers:
                bar = observation.bars.get(tk)
                if bar is None or bar.adjusted_close <= 0 or sleeve_each <= 0:
                    continue
                frac = max(0.0, min(1.0, float(fractions_by_ticker.get(tk, 1.0))))
                cur_qty = held[tk].quantity if tk in held else 0
                cur_value = cur_qty * bar.adjusted_close
                target_value = frac * sleeve_each
                if abs(target_value - cur_value) / sleeve_each <= rebalance_band:
                    continue
                delta_qty = int(target_value / bar.adjusted_close) - cur_qty
                if delta_qty > 0:
                    intents.append(OrderIntent(ticker=tk, side="buy", quantity=delta_qty))
                elif delta_qty < 0:
                    intents.append(OrderIntent(ticker=tk, side="sell", quantity=-delta_qty))
    else:  # flat -> sell everything held
        for tk, pos in held.items():
            if pos.quantity > 0:
                intents.append(OrderIntent(ticker=tk, side="sell", quantity=ALL_AVAILABLE))
    return intents


def _apply_stale_view_abstention(
    views_by_expert: Mapping[str, Mapping[str, Any]],
    direction_runs: dict[tuple[str, str], tuple[str, int]],
    tickers: Sequence[str],
    *,
    window: int,
) -> dict[str, dict[str, Any]]:
    """Drop views whose direction has repeated for longer than ``window`` active dates.

    ``direction_runs`` maps (expert_id, ticker) -> (last direction, run length INCLUDING
    this date) and is updated in place, so the decision uses only past and present
    directions. Abstention here means the view is absent from the returned mapping,
    which is the aggregation's existing sleeping-expert path: the remaining weights
    renormalise, and silence never forces an exit. A view speaks for ``window`` active
    dates after its direction changes and is muted from then until it changes again;
    the registered variants differ only in ``window`` (F1 = 63, F2 = 21).
    """
    filtered: dict[str, dict[str, Any]] = {}
    for expert_id, by_ticker in views_by_expert.items():
        kept: dict[str, Any] = {}
        for ticker in tickers:
            view = by_ticker.get(ticker)
            if view is None:
                continue
            key = (expert_id, ticker)
            direction = str(view.direction)
            previous, run = direction_runs.get(key, (None, 0))
            run = run + 1 if direction == previous else 1
            direction_runs[key] = (direction, run)
            # A view that is already an abstention (neutral / safe default) is left
            # exactly as it is: the aggregation drops it either way, and re-deciding it
            # here would double-count the abstention in the audit fields.
            if is_abstention_view(view):
                kept[ticker] = view
                continue
            if run > window:
                continue
            kept[ticker] = view
        if kept:
            filtered[expert_id] = kept
    return filtered


ATTRIBUTION_MODES: tuple[str, ...] = ("raw", "market_adjusted")


def _attribute(
    decision_date, market_state, views_by_expert, weights, observation, panel,
    decisions_by_ticker, *, attribution_mode="raw", market_return=None,
    learning_horizon=1, learning_market_return=None,
):
    """Build per-name AttributedEpisodes carrying two separate signals.

    LEARNING evidence (``signal_attribution``) is each expert's COUNTERFACTUAL contribution
    (Chapter 3, Eq. contribution): ``g = v * r`` where ``v = +1`` for a long view and ``-1``
    for a flat view, and ``r`` is the next-period return. It scores the expert on its OWN
    view (a +/-1 vote), independent of the ensemble weight it actually held and of the
    system's action, so the evidence does not depend on the weights the system held when the
    view was expressed and the reflection feedback path is removed by construction.
    Transaction costs are charged once at the portfolio execution layer and do NOT enter the
    per-expert attribution, so the learning signal reflects directional accuracy rather than
    trading frequency. The default ``raw`` mode does not demean ``r``; ``market_adjusted``
    subtracts the market proxy and is retained only for the E2 ablation.

    REPORTING (``attribution``) is the realized market/selection decomposition of the
    system's actual position: the market proxy plus the name's excess when the system is
    long, and zero when flat. It is kept separate from the learning evidence so the two
    cannot be conflated; the full market/style/selection split lives at this reporting layer.

    The market baseline is an external proxy (e.g. SPY) when provided -- required for
    single-name runs, where the universe mean would be the traded stock itself -- otherwise
    the equal-weight universe forward return. Returns [] during warmup or with no forward bar.
    """
    from trading_agent.memory.expert_reflection import ATTRIBUTION_COMPONENTS, AttributedEpisode
    from trading_agent.memory.expert_weights import MARKET_STATES
    from trading_agent.memory.expert_error import build_expert_error_records

    if market_state not in MARKET_STATES:
        return []
    realized_by_tk: dict[str, float] = {}
    learning_by_tk: dict[str, float] = {}
    for tk in observation.tickers:
        p0 = panel._price_on(tk, decision_date)
        p1 = panel.price_after(tk, decision_date, 1)
        if p0 and p1 and p0 > 0:
            realized_by_tk[tk] = p1 / p0 - 1.0
            # LEARNING return at the attribution horizon h (>= 1): experts' edge is
            # multi-day (h=21 monthly), so the counterfactual g = v * r_{t->t+h} can be
            # scored at h != 1. The h-bar forward price is realized only at t+h; the loop
            # EMBARGOES the episode by h active dates so no decision at or before t+h reads
            # it (see run_expert_loop). Reporting attribution stays 1-bar (the actual
            # holding-period P&L). At h=1 this collapses to the 1-bar return exactly.
            ph = panel.price_after(tk, decision_date, learning_horizon)
            if ph and p0 > 0:
                learning_by_tk[tk] = ph / p0 - 1.0
            # A missing h-day target means the forward window never elapsed
            # for this ticker: it contributes NO learning evidence (the
            # thesis defines g over the elapsed window; a silent 1-bar
            # stand-in would mix horizons).
    if not realized_by_tk:
        return []
    if not learning_by_tk:
        learning_by_tk = {}
    if market_return is not None:
        market_proxy = float(market_return)
    else:
        market_proxy = sum(realized_by_tk.values()) / len(realized_by_tk)
    if learning_market_return is not None:
        learning_market = float(learning_market_return)
    elif market_return is not None:
        # h=1 path passes only market_return; the 1-bar learning baseline is the
        # same external proxy as the reporting baseline (learning collapses to the
        # 1-bar return at h=1). Falling back to the universe mean here would demean a
        # single-name return by itself and degenerate the market_adjusted score to 0.
        learning_market = float(market_return)
    else:
        learning_market = (
            sum(learning_by_tk.values()) / len(learning_by_tk)
            if learning_by_tk else 0.0
        )

    episodes = []
    for tk, realized in realized_by_tk.items():
        learning = learning_by_tk.get(tk)
        score_return = (
            None if learning is None
            else (learning - learning_market if attribution_mode == "market_adjusted" else learning)
        )
        action = decisions_by_ticker.get(tk, "flat")
        ticker_views = {
            expert_id: by_ticker[tk]
            for expert_id, by_ticker in views_by_expert.items()
            if tk in by_ticker
        }
        # Learning evidence: per-expert counterfactual contribution g = v * r, v in {+1, -1}
        # from the expert's OWN direction, NOT scaled by its ensemble weight and independent
        # of the system's action (this removes the feedback path). No transaction cost enters
        # the per-expert signal. SAFE-DEFAULT views (data unavailable / parse error) are
        # excluded: a fallback flat is not a judgment, and scoring it would penalise an
        # expert for missing data (e.g. the news expert over pre-coverage years). The
        # safe-default still votes flat in the aggregation as conservative behaviour; it
        # just must not teach the scorecard anything.
        signal_attribution: dict[str, float] = {}
        for expert_id, view in ticker_views.items():
            if score_return is None:
                break
            if is_safe_default_view(view):
                continue
            if view.direction == "neutral":
                # Deliberate abstention (design D3): not a directional judgment, so it
                # must not enter the learning evidence — else selective experts (right
                # when they speak, silent otherwise) would be diluted toward zero.
                # Trust semantics become P(correct | speaks).
                continue
            v = 1.0 if view.direction == "long" else -1.0
            signal_attribution[expert_id] = v * score_return
        # Reporting: the system's realized market/selection decomposition (long only).
        attribution = {c: 0.0 for c in ATTRIBUTION_COMPONENTS}
        if action == "long":
            attribution["market"] = market_proxy
            attribution["signal"] = realized - market_proxy
        expert_errors = tuple(build_expert_error_records(
            market_state=market_state,
            views=ticker_views,
            weights=weights,
            # Audit records score direction against the return they describe;
            # when the h-day learning window never elapsed they fall back to
            # the realized 1-bar return (reporting basis), matching the
            # episode's own attribution decomposition.
            realized_next_return=score_return if score_return is not None else realized,
        ))
        episodes.append(AttributedEpisode(
            decision_date=decision_date,
            market_state=market_state,
            attribution=attribution,
            signal_attribution=signal_attribution,
            expert_errors=expert_errors,
            action=action,
            ticker=tk,
        ))
    return episodes


def run_expert_loop(
    loader,
    config: BacktestConfig,
    experts: Sequence,
    *,
    market_state_config=None,
    weight_store=None,
    reflection=None,
    reflect_every: int = 21,
    reflection_window: int = 63,
    reflection_min_support: int = 5,
    decision_threshold: float = 0.0,
    exit_threshold: float | None = 0.0,
    allow_reflection: bool = True,
    attribution_mode: str = "raw",
    attribution_horizon: int = 1,
    market_proxy=None,
    signal_sources: ExpertSignalSources | None = None,
    run_id: Optional[str] = None,
    progress_callback: Callable[[dict], None] | None = None,
    parallel_expert_calls: bool = False,
    confidence_weighted_vote: bool = False,
    soft_abstain_weight: float = 0.0,
    decision_cadence: int = 1,
    position_sizing: str = "binary",
    stale_view_abstain: int = 0,
) -> ExpertLoopResult:
    """Run the expert loop end to end once and return a populated result.

    ``experts`` is the expert committee (each exposes ``expert_id`` and
    ``view(observation, market_state, context=None)``). ``weight_store`` may be
    pre-seeded with learned entries to exercise the reuse path.

    ``reflect_every`` and ``reflection_window`` are measured in ACTIVE EPISODE
    DATES, not per-name episode counts, so the reflection cadence does not
    speed up with universe size.
    """
    from trading_agent.alpha.market_state_classifier import MarketStateClassifier, MarketStateConfig
    from trading_agent.aggregation.soft_weighting import aggregate_by_ticker
    from trading_agent.market import LoaderPricePanel
    from trading_agent.memory.expert_reflection import DeterministicExpertReflection
    from trading_agent.memory.expert_weights import MARKET_STATES
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    if reflect_every < 1:
        raise ValueError("reflect_every must be >= 1")
    if reflection_window < 1:
        raise ValueError("reflection_window must be >= 1")
    if reflection_min_support < 1:
        raise ValueError("reflection_min_support must be >= 1")
    if attribution_horizon < 1:
        raise ValueError("attribution_horizon must be >= 1")
    if position_sizing not in ("binary", "proportional"):
        raise ValueError("position_sizing must be 'binary' or 'proportional'")
    if stale_view_abstain < 0:
        raise ValueError("stale_view_abstain must be >= 0 (0 disables it)")
    # A = sum w*v in [-1, 1] (v = +-1 over the floored simplex). A negative
    # entry threshold is a valid participation-biased rule (go long unless the
    # committee is clearly bearish); a positive one is conviction-biased.
    if not -1.0 <= float(decision_threshold) <= 1.0:
        raise ValueError("decision_threshold must be in [-1, 1]")
    if attribution_mode not in ATTRIBUTION_MODES:
        raise ValueError(f"attribution_mode must be one of {ATTRIBUTION_MODES}")

    def emit(event: str, **payload) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, **payload})

    random.seed(config.seed)

    adapter = FinsaberBacktestAdapter(loader, config)
    panel = LoaderPricePanel(adapter._window)
    # The market proxy (e.g. SPY) drives BOTH the 2D regime state (trend from its levels,
    # risk from its log returns) and the attribution market baseline (its simple returns).
    # For a single name this proxy MUST be external, never the traded stock itself, or the
    # stock's own trend would leak into the regime label. With no proxy the classifier
    # builds the equal-weight universe proxy internally (multi-name runs).
    proxy_returns = market_proxy.returns_by_date(panel.dates) if market_proxy is not None else None
    proxy_levels = market_proxy.levels_by_date(panel.dates) if market_proxy is not None else None
    classifier = MarketStateClassifier(
        panel, market_state_config or MarketStateConfig(),
        proxy_levels=proxy_levels,
    )
    store = weight_store if weight_store is not None else ExpertScorecardStore()
    reflection = reflection or DeterministicExpertReflection(min_support=reflection_min_support)
    run_id = run_id or f"expert-{config.fingerprint()[:12]}"
    context_builder = ExpertSignalContextBuilder(
        panel=panel,
        store=store,
        sources=signal_sources or ExpertSignalSources(),
    )
    shared_cache = getattr(context_builder.sources, "context_cache", None)
    state_cfg = classifier._cfg
    state_cache_config = (
        state_cfg.vol_window,
        state_cfg.percentile_window,
        state_cfg.annualization,
        state_cfg.trend_window,
        state_cfg.trend_confirm_days,
        state_cfg.stress_percentile,
    )

    market_states: list[tuple] = []
    episodes: list = []
    n_long_days = 0
    n_long_ticker_days = 0
    n_active_ticker_days = 0
    n_active_dates = 0
    n_reflection_rounds = 0
    n_proposals = 0
    n_applied = 0
    n_learned = 0
    reflection_trial_records: list = []
    confidence_records: dict[str, list[dict]] = {}
    equal_weight_score_records: list[dict] = []
    # Previous decision per ticker, carried across dates for the optional
    # HOLD semantics (exit_threshold); empty when hysteresis is off.
    prev_directions_by_ticker: dict[str, str] = {}
    # Previous target fraction per ticker (position_sizing="proportional"): the
    # A = 0 atom and cadence-carried days MAINTAIN the last spoken size, mirroring
    # the D3 maintain semantics of the direction layer.
    prev_fractions_by_ticker: dict[str, float] = {}
    # (expert_id, ticker) -> (last direction, consecutive active dates at it), for the
    # Family F stale-view abstention; empty and unused when the rule is off.
    direction_runs: dict[tuple[str, str], tuple[str, int]] = {}
    if decision_cadence < 1:
        raise ValueError("decision_cadence must be >= 1")
    decision_day_idx = 0   # counts processed dates for the coarser decision cadence
    context_records: list[dict] = []
    episode_dates: list = []          # distinct decision dates that produced episodes
    active_dates_since_reflection = 0
    order_seq = 0
    # Horizon-attribution embargo (only engaged when attribution_horizon > 1). Episodes
    # attributed at h-bar forward returns can only enter memory h active dates later, so
    # no decision at or before t+h reads a signal derived from price_{t+h}. Each entry is
    # (mature_at_active_index, snapshot dict). The h == 1 path never touches this queue and
    # is byte-identical to the pre-horizon loop.
    from collections import deque as _deque
    pending_attribution: "_deque" = _deque()

    for decision_date in adapter.dates:
        emit("date_start", date=decision_date, run_id=run_id)
        adapter.open_bar(decision_date)
        account = adapter.get_account()
        positions = adapter.get_positions()
        observation = adapter.get_observation(decision_date)

        state_cache_key = None
        if shared_cache is not None:
            state_cache_key = (
                "market_state",
                decision_date,
                tuple(observation.tickers),
                state_cache_config,
            )
            market_state = shared_cache.get(state_cache_key)
        else:
            market_state = None
        if market_state is None:
            market_state = classifier.classify(observation)
            if shared_cache is not None and state_cache_key is not None:
                shared_cache[state_cache_key] = market_state
        market_states.append((decision_date, market_state))

        if market_state not in MARKET_STATES:
            emit("date_done", date=decision_date, market_state=market_state, active=False, run_id=run_id)
            adapter.advance()
            continue

        n_active_dates += 1

        # Horizon embargo release (h > 1 only): fold any episode whose h-bar forward
        # window has fully elapsed BEFORE this date's get_weights, so the decision below
        # never reads a signal derived from a price at or after its own date. An episode
        # created at active index k matures at k + h; releasing at index >= k + h mirrors
        # the h == 1 convention (fold after date t, first read at t + h) exactly.
        if attribution_horizon > 1:
            while pending_attribution and pending_attribution[0][0] <= n_active_dates:
                _, snap = pending_attribution.popleft()
                matured = _attribute(
                    snap["decision_date"], snap["market_state"], snap["views_by_expert"],
                    snap["weights"], snap["observation"], panel, snap["directions_by_ticker"],
                    attribution_mode=attribution_mode,
                    market_return=(market_proxy.forward_return(snap["decision_date"], 1)
                                   if market_proxy is not None else None),
                    learning_horizon=attribution_horizon,
                    learning_market_return=(market_proxy.forward_return(snap["decision_date"], attribution_horizon)
                                            if market_proxy is not None else None),
                )
                episodes.extend(matured)
                if matured:
                    episode_dates.append(snap["decision_date"])
                    active_dates_since_reflection += 1
                if allow_reflection and hasattr(store, "observe"):
                    for ep in matured:
                        store.observe(ep.market_state, ep.signal_attribution)

        # experts -> per-ticker views; signal blocks are audit-only context
        expert_context = context_builder.build(observation, market_state)
        context_records.append(expert_context)
        views_cache_key = None
        if shared_cache is not None:
            views_cache_key = (
                "expert_views",
                decision_date,
                tuple(observation.tickers),
                tuple(e.expert_id for e in experts),
            )
            views_by_expert = shared_cache.get(views_cache_key)
        else:
            views_by_expert = None
        if views_by_expert is None:
            views_by_expert = {}

            def call_expert(e):
                emit(
                    "expert_start",
                    date=decision_date,
                    market_state=market_state,
                    expert_id=e.expert_id,
                    run_id=run_id,
                )
                views = e.views(
                    observation,
                    market_state,
                    context=expert_context,
                )
                emit(
                    "expert_done",
                    date=decision_date,
                    market_state=market_state,
                    expert_id=e.expert_id,
                    n_tickers=len(views),
                    run_id=run_id,
                )
                return e.expert_id, views

            if parallel_expert_calls:
                with ThreadPoolExecutor(max_workers=len(experts)) as pool:
                    futures = [pool.submit(call_expert, e) for e in experts]
                    for future in as_completed(futures):
                        expert_id, views = future.result()
                        views_by_expert[expert_id] = views
            else:
                for e in experts:
                    expert_id, views = call_expert(e)
                    views_by_expert[expert_id] = views
            if shared_cache is not None and views_cache_key is not None:
                shared_cache[views_cache_key] = views_by_expert
        for expert_id, by_ticker in views_by_expert.items():
            rows = confidence_records.setdefault(expert_id, [])
            for ticker, view in by_ticker.items():
                rows.append({
                    "date": decision_date,
                    "ticker": ticker,
                    "market_state": market_state,
                    "confidence": float(view.confidence),
                    "direction": view.direction,
                    "safe_default": is_safe_default_view(view),
                    "safe_default_reason": safe_default_reason(view),
                    "reasoning": str(view.reasoning or ""),
                })

        # learned memory reused here (equal-weight fallback when empty)
        weights = store.get_weights(market_state)

        # Family F (prereg Amendment 4): a view that has repeated the same direction for
        # longer than the expert's evidence actually refreshes carries no information, yet
        # it still tilts A = sum w*v. Such a view leaves the VOTE through the existing D3
        # abstention path. ``views_by_expert`` itself is untouched, so attribution and
        # learning evidence -- which read the expert's own direction and are
        # action-independent -- stay byte-identical and the grid-invariance contract holds.
        voting_views = views_by_expert
        if stale_view_abstain:
            voting_views = _apply_stale_view_abstention(
                views_by_expert,
                direction_runs,
                observation.tickers,
                window=int(stale_view_abstain),
            )

        equal_agg = aggregate_by_ticker(
            voting_views,
            {},
            tickers=tuple(observation.tickers),
            threshold=float(decision_threshold),
            confidence_weighted=confidence_weighted_vote,
            soft_abstain_weight=soft_abstain_weight,
        )
        for ticker, decision in equal_agg.by_ticker.items():
            equal_weight_score_records.append({
                "date": decision_date,
                "ticker": ticker,
                "market_state": market_state,
                "score": float(decision.score),
                "direction": decision.direction,
            })

        agg = aggregate_by_ticker(
            voting_views,
            weights,
            tickers=tuple(observation.tickers),
            threshold=float(decision_threshold),
            exit_threshold=exit_threshold,
            prev_directions=prev_directions_by_ticker,
            confidence_weighted=confidence_weighted_vote,
            soft_abstain_weight=soft_abstain_weight,
        )
        if decision_cadence > 1 and (decision_day_idx % decision_cadence) != 0:
            # Coarser decision cadence (monthly or weekly horizons): the
            # committee re-votes only every Nth processed date; positions carry between.
            # The aggregation above still runs daily so views, weights_used, and hence
            # the attribution/learning evidence stay byte-identical (action-independent);
            # only the APPLIED position changes.
            directions_by_ticker = {
                ticker: ("long" if prev_directions_by_ticker.get(ticker) == "long" else "flat")
                for ticker in observation.tickers
            }
        else:
            directions_by_ticker = {
                ticker: decision.direction
                for ticker, decision in agg.by_ticker.items()
            }
        fractions_by_ticker: dict[str, float] | None = None
        if position_sizing == "proportional":
            fresh_vote = not (decision_cadence > 1 and (decision_day_idx % decision_cadence) != 0)
            fractions_by_ticker = {}
            for ticker in observation.tickers:
                if directions_by_ticker.get(ticker) != "long":
                    fractions_by_ticker[ticker] = 0.0
                    continue
                decision = agg.by_ticker.get(ticker)
                score = float(decision.score) if decision is not None else 0.0
                if fresh_vote and score > 0.0:
                    fractions_by_ticker[ticker] = min(1.0, score)
                else:
                    # A = 0 with a held long, or a cadence-carried day: maintain.
                    fractions_by_ticker[ticker] = prev_fractions_by_ticker.get(ticker, 1.0)
            prev_fractions_by_ticker.update(fractions_by_ticker)
        decision_day_idx += 1
        prev_directions_by_ticker.update(directions_by_ticker)
        active_tickers = len(observation.tickers)
        n_active_ticker_days += active_tickers
        long_count = sum(1 for direction in directions_by_ticker.values() if direction == "long")
        n_long_ticker_days += long_count
        if long_count > 0:
            n_long_days += 1

        # deterministic risk layer -> orders -> execute
        intents = _direction_to_intents(
            directions_by_ticker, observation, account, positions,
            fractions_by_ticker=fractions_by_ticker,
        )
        decision_id = f"{run_id}:{decision_date.isoformat()}"
        for intent in intents:
            order_seq += 1
            bar = observation.bars.get(intent.ticker)
            ref_price = bar.adjusted_close if bar is not None else None
            adapter.submit_order(Order(
                order_id=f"{decision_id}:{order_seq}",
                decision_id=decision_id,
                signal_date=decision_date,
                ticker=intent.ticker,
                side=intent.side,
                quantity=intent.quantity,
                reference_price=ref_price,
            ))

        # attribution (post-execution; feeds reflection, never the decision)
        if attribution_horizon == 1:
            market_return = (
                market_proxy.forward_return(decision_date, 1) if market_proxy is not None else None
            )
            new_episodes = _attribute(
                decision_date, market_state, views_by_expert, agg.weights_used,
                observation, panel, directions_by_ticker,
                attribution_mode=attribution_mode, market_return=market_return,
            )
            episodes.extend(new_episodes)
            if new_episodes:
                episode_dates.append(decision_date)
                active_dates_since_reflection += 1

            # Fold this date's per-expert counterfactual contributions into the scorecard.
            # CAUSAL ORDERING: this runs AFTER get_weights (the decision for this date) and
            # after execution and attribution, so the weights used on date t depend only on
            # observations through t-1; date t never sees its own outcome. Controlled by
            # allow_reflection so the frozen test window (allow_reflection=False) does not learn.
            if allow_reflection and hasattr(store, "observe"):
                for ep in new_episodes:
                    store.observe(ep.market_state, ep.signal_attribution)
        else:
            # h > 1: defer attribution until the h-bar window elapses (embargo release
            # above). Snapshot the decision-time inputs; the forward price and market
            # return are read at maturity, never now, so nothing here touches the future.
            pending_attribution.append((n_active_dates + attribution_horizon, {
                "decision_date": decision_date,
                "market_state": market_state,
                "views_by_expert": views_by_expert,
                "weights": agg.weights_used,
                "observation": observation,
                "directions_by_ticker": directions_by_ticker,
            }))

        # periodic reflection -> learned memory. Honesty is tested at the frozen
        # system level in holdout, not by scoring individual adjustments here.
        # Cadence is in ACTIVE EPISODE DATES (not per-name episode counts, which
        # scale with universe size): reflect every `reflect_every` new dates over
        # the per-name episodes of the most recent `reflection_window` dates.
        # MIN_SUPPORT is checked per (expert_id, market_state) on non-default
        # expert-error evidence.
        if allow_reflection and episodes and active_dates_since_reflection >= reflect_every:
            n_reflection_rounds += 1
            active_dates_since_reflection = 0
            participation = (n_long_ticker_days / n_active_ticker_days) if n_active_ticker_days else 0.0
            window_start = episode_dates[max(0, len(episode_dates) - reflection_window)]
            window_episodes = [ep for ep in episodes if ep.decision_date >= window_start]
            proposals = reflection.reflect(window_episodes, decision_date, participation)
            n_proposals += len(proposals)
            applied = []
            for proposal in proposals:
                store.add_learned_adjustment(proposal)
                applied.append(proposal.entry_id)
                n_applied += 1
                n_learned += 1
            reflection_trial_records.append({
                "decision_date": decision_date,
                "n_proposals": len(proposals),
                "n_applied": len(applied),
                "proposal_ids": applied,
            })

        adapter.advance()
        emit(
            "date_done",
            date=decision_date,
            market_state=market_state,
            active=True,
            long_count=long_count,
            n_tickers=active_tickers,
            run_id=run_id,
        )

    adapter.finalize()
    backtest = adapter.result(run_id)
    for expert in experts:
        close = getattr(getattr(expert, "_backend", None), "close", None)
        if callable(close):
            close()

    return ExpertLoopResult(
        run_id=run_id,
        backtest=backtest,
        n_dates=len(adapter.dates),
        n_active_dates=n_active_dates,
        market_states=market_states,
        n_long_days=n_long_days,
        participation_rate=(n_long_ticker_days / n_active_ticker_days) if n_active_ticker_days else 0.0,
        n_reflection_rounds=n_reflection_rounds,
        n_proposals=n_proposals,
        n_applied=n_applied,
        n_learned=store.count_learned(),
        decision_threshold=float(decision_threshold),
        confidence_diagnostics=_confidence_summary(confidence_records),
        expert_view_records=confidence_records,
        equal_weight_score_records=equal_weight_score_records,
        context_records=context_records,
        episodes=episodes,
        reflection_trial_records=reflection_trial_records,
    )
