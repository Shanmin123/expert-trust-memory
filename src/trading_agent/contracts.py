"""Phase 1 — backtesting contracts.

Plain Python dataclasses + Protocols. NO dependency on LangGraph / asyncio / any
orchestration framework (hard invariant). All value objects are frozen (immutable
snapshots), which is how we structurally enforce that the decision layer (and, later,
the LLM) can never mutate cash, positions, fills, or account state.

Two structurally separate families
----------------------------------
* Decision-side (LLM-facing later): ``Signal``, ``TargetPosition``, ``OrderIntent``.
  These carry NO authority over cash/holdings — no cash, equity, or realized-position
  fields exist on them.
* State / engine-owned: ``Order``, ``Fill``, ``RejectedOrder``, ``Position``,
  ``PortfolioState``, ``AccountState``. Only the deterministic execution engine produces
  these; the decision side merely reads the read-only snapshots.

The set of cash/holdings-bearing field names lives in ``STATE_AUTHORITY_FIELDS`` and is
asserted against the decision-side types by the contract test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date, datetime
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable

Side = Literal["buy", "sell"]

#: Field names that confer authority over cash/holdings. Decision-side types must expose
#: none of these (enforced by test_trading_contracts.py).
STATE_AUTHORITY_FIELDS = frozenset(
    {"cash", "initial_cash", "equity", "positions_value", "fill_price", "commission"}
)

# Rejection reason codes are reused verbatim from FINSABER's execution model.
REJECTION_REASONS = frozenset(
    {
        "invalid_price",
        "insufficient_liquidity_history",
        "zero_quantity",
        "insufficient_cash",
        "insufficient_holdings",
        "no_future_bar",
    }
)

#: Sentinel quantity meaning "all-in" (buy: int(cash/price)) / "all-out" (sell: full holding),
#: matching FINSABER ``_resolve_quantity``.
ALL_AVAILABLE = -1


# --------------------------------------------------------------------------------------
# Market data (point-in-time)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    """A single point-in-time OHLCV bar (decision-facing view of one ticker/date)."""

    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float
    adjusted_open: Optional[float] = None
    adjusted_high: Optional[float] = None
    adjusted_low: Optional[float] = None


@dataclass(frozen=True)
class Observation:
    """Everything the decision layer may legally see at ``decision_date``.

    ``availability_time`` is the latest availability timestamp of any datum in this
    observation. The hard leakage invariant is ``availability_time <= decision_time``
    (date-level data is treated as available at the decision date; we act next_open).
    """

    decision_date: Date
    decision_time: datetime
    availability_time: datetime
    bars: Mapping[str, Bar]
    tickers: Sequence[str]


# --------------------------------------------------------------------------------------
# Decision-side types (no state authority)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    """An alpha-engine output: a value with a source id."""

    ticker: str
    value: float
    source_id: str
    as_of: Date


@dataclass(frozen=True)
class TargetPosition:
    """Desired exposure for a ticker, expressed as a portfolio weight in [0, 1] (long/flat)."""

    ticker: str
    target_weight: float
    as_of: Date


@dataclass(frozen=True)
class OrderIntent:
    """A requested order before it is accepted as an ``Order`` and submitted."""

    ticker: str
    side: Side
    quantity: int  # shares, or ALL_AVAILABLE sentinel
    limit_price: Optional[float] = None


# --------------------------------------------------------------------------------------
# State / engine-owned types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Order:
    """An accepted order submitted to the ExecutionAdapter (next_open queue)."""

    order_id: str
    decision_id: str
    signal_date: Date
    ticker: str
    side: Side
    quantity: int
    reference_price: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    """A completed execution. Mirrors FINSABER's history record fields."""

    order_id: str
    ticker: str
    side: Side
    signal_date: Date
    execution_date: Date
    reference_price: float
    fill_price: float
    quantity: int
    commission: float
    slippage_cost: float
    average_volume: Optional[float]
    volume_observations: int
    participation_rate: Optional[float]


@dataclass(frozen=True)
class RejectedOrder:
    order_id: str
    ticker: str
    side: Side
    signal_date: Date
    execution_date: Optional[Date]
    requested_quantity: int
    reason: str


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: int
    avg_price: float


@dataclass(frozen=True)
class PortfolioState:
    as_of: Date
    positions: Sequence[Position]

    def by_ticker(self) -> Mapping[str, Position]:
        return {p.ticker: p for p in self.positions}


@dataclass(frozen=True)
class AccountState:
    as_of: Date
    cash: float
    initial_cash: float
    positions_value: float
    equity: float


# --------------------------------------------------------------------------------------
# Run configuration & result
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    """Deterministic run configuration. Execution params mirror FINSABER ``TradeConfig``."""

    date_from: Date
    date_to: Date
    initial_cash: float = 100_000.0
    tickers: Optional[Sequence[str]] = None  # None => all tickers in the loader window
    execution_timing: Literal["next_open"] = "next_open"
    price_field: str = "adjusted_close"
    commission_per_share: float = 0.0049
    min_commission: float = 0.99
    max_commission_rate: float = 0.01
    slippage_perc: float = 0.0
    slippage_impact: float = 0.0
    execution_cost_bps_per_side: float = 0.0
    liquidity_lookback_days: int = 20
    liquidity_min_history_days: int = 1
    liquidity_cap_pct: float = 0.0
    preload_date_from: Optional[Date] = None
    seed: int = 0

    def fingerprint(self) -> str:
        """Stable hash of the config for run manifests / idempotent persistence keys."""
        import hashlib
        import json

        payload = {
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "initial_cash": self.initial_cash,
            "tickers": list(self.tickers) if self.tickers is not None else None,
            "execution_timing": self.execution_timing,
            "price_field": self.price_field,
            "commission_per_share": self.commission_per_share,
            "min_commission": self.min_commission,
            "max_commission_rate": self.max_commission_rate,
            "slippage_perc": self.slippage_perc,
            "slippage_impact": self.slippage_impact,
            "execution_cost_bps_per_side": self.execution_cost_bps_per_side,
            "liquidity_lookback_days": self.liquidity_lookback_days,
            "liquidity_min_history_days": self.liquidity_min_history_days,
            "liquidity_cap_pct": self.liquidity_cap_pct,
            "preload_date_from": self.preload_date_from.isoformat() if self.preload_date_from else None,
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    config_fingerprint: str
    equity_curve: Sequence[tuple]  # (Date, equity_float)
    fills: Sequence[Fill]
    rejected: Sequence[RejectedOrder]
    final_account: AccountState
    summary: Mapping[str, float]


# --------------------------------------------------------------------------------------
# Protocols (interfaces) — no orchestration-framework dependency
# --------------------------------------------------------------------------------------
@runtime_checkable
class AlphaEngine(Protocol):
    def score(self, observation: Observation) -> Sequence[Signal]: ...


@runtime_checkable
class RiskEngine(Protocol):
    def size(
        self,
        signals: Sequence[Signal],
        account: AccountState,
        observation: Observation,
    ) -> Sequence[TargetPosition]: ...


@runtime_checkable
class PortfolioEngine(Protocol):
    def build_orders(
        self,
        targets: Sequence[TargetPosition],
        account: AccountState,
        positions: PortfolioState,
        observation: Observation,
    ) -> Sequence[OrderIntent]: ...


# --------------------------------------------------------------------------------------
# Expert types
# --------------------------------------------------------------------------------------
# Ternary vote (design D3): "neutral" is a DELIBERATE no-edge judgment — it abstains from
# the committee vote (weights renormalise over the speakers) and never enters learning
# evidence, exactly like the sleeping-experts / specialists setting (Freund et al., 1997;
# Cesa-Bianchi & Lugosi, 2006). "flat" remains the bearish vote (identifiable downside).
ExpertDirection = Literal["long", "neutral", "flat"]
ExpertDataQuality = Literal["high", "medium", "low", "unknown"]


@dataclass(frozen=True)
class ExpertView:
    """One expert's assessment for one ticker on a single rebalance date.

    Direction and confidence are the only machine-read fields in aggregation.
    Reasoning and report-style fields are short audit strings, never parsed by
    aggregation.
    """

    expert_id: str
    direction: ExpertDirection
    confidence: float  # [0, 1]
    reasoning: str
    evidence: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    data_quality: ExpertDataQuality = "unknown"
    # Design D3 comparison instrumentation (2026-07-17): the binary judgment the
    # expert would have given had "neutral" not been available, captured in the
    # SAME response (round-2 review: the only way to identify the effect of the
    # ternary vote itself is a frozen forced-binary replay over one corpus).
    # None when not provided. NEVER read by ternary aggregation or learning.
    forced_direction: Optional[Literal["long", "flat"]] = None


#: Reasons an expert emits its fallback view instead of a judgment.
SAFE_DEFAULT_REASONS = frozenset({"parse_error", "data_unavailable"})


def is_safe_default_view(view: "ExpertView") -> bool:
    """True for parser/data-unavailable fallbacks that are NOT judgments.

    Safe defaults are excluded from learning evidence and from the aggregation
    vote (the remaining experts' weights are renormalised); they are audit
    records, nothing more.
    """
    return (
        view.direction == "flat"
        and float(view.confidence) == 0.0
        and view.reasoning in SAFE_DEFAULT_REASONS
        and view.reasoning in view.risk_flags
        and view.data_quality == "low"
    )


def is_abstention_view(view: "ExpertView") -> bool:
    """True when the view abstains from the vote: a deliberate neutral OR a safe default.

    Both leave the committee vote (weights renormalise over the speakers) and neither
    enters learning evidence. They stay distinguishable for audit: a neutral is a
    JUDGMENT ("no directional edge"), a safe default is a fallback (missing data /
    parse failure).
    """
    return view.direction == "neutral" or is_safe_default_view(view)


@runtime_checkable
class Expert(Protocol):
    """Structural interface every expert must satisfy.

    ``context`` carries pre-computed signals specific to this expert type (keyed by
    expert_id). It is ``None`` when no upstream signals are available; experts must
    handle that gracefully and return a safe default (direction="flat", confidence=0).
    """

    expert_id: str

    def views(
        self,
        observation: "Observation",
        regime_label: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Mapping[str, ExpertView]: ...


@runtime_checkable
class ExecutionAdapter(Protocol):
    """The single contract shared by FinsaberBacktestAdapter and (later) IBPaperAdapter.

    Point-in-time, pull-based. The LLM/decision layer reads observations and submits
    orders; it never mutates trading state directly.
    """

    def get_observation(self, t: Date) -> Observation: ...
    def submit_order(self, order: Order) -> None: ...
    def get_positions(self) -> PortfolioState: ...
    def get_account(self) -> AccountState: ...
