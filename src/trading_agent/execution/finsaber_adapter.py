"""FinsaberBacktestAdapter: Approach A (data loaders + execution.py formulas only).

This adapter uses FINSABER strictly as:
  1. a point-in-time DATA layer (an injected ``TradingData``: ``FinsaberDataset`` for tests,
     ``FinsaberParquetDataset`` for real runs), and
  2. a set of pure execution FORMULAS (``backtest.toolkit.execution``).

It implements its OWN shared-portfolio accounting (one account across all tickers) and its
own next_open pending-order queue, exposing the ``ExecutionAdapter`` contract so
the same agent code can later run against ``IBPaperAdapter``.

Fidelity rule: ``_fill`` / ``_resolve_quantity`` replicate FINSABER ``_execute_order`` /
``_resolve_quantity`` step-for-step (quantity resolution incl. the -1 sentinel, prior-only
liquidity filter/cap, slippage, commission, insufficient-cash decrement loop, FINSABER reason
codes), but write to OUR state. The execution *math* is reused verbatim from FINSABER, never
reimplemented.
"""

from __future__ import annotations

from datetime import date as Date, datetime, time
from typing import Any, Mapping, Optional

from trading_agent import _bootstrap  # ensures FINSABER clone on sys.path
from trading_agent._bootstrap import (
    adjusted_bar_price,
    apply_liquidity_cap,
    apply_slippage,
    calculate_commission,
    prior_volume_stats,
)
from trading_agent.contracts import (
    ALL_AVAILABLE,
    AccountState,
    Bar,
    BacktestConfig,
    BacktestResult,
    Fill,
    Observation,
    Order,
    Position,
    PortfolioState,
    RejectedOrder,
)


def _to_date(value) -> Date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        import pandas as pd

        return pd.to_datetime(value).date()
    return value


def _bar_from_raw(raw: Any) -> Bar:
    if isinstance(raw, Mapping):
        close = raw.get("close", raw.get("adjusted_close"))
        adj_close = raw.get("adjusted_close", close)
        return Bar(
            open=raw.get("open", close),
            high=raw.get("high", close),
            low=raw.get("low", close),
            close=close,
            adjusted_close=adj_close,
            volume=raw.get("volume", 0),
            adjusted_open=raw.get("adjusted_open"),
            adjusted_high=raw.get("adjusted_high"),
            adjusted_low=raw.get("adjusted_low"),
        )
    price = float(raw)
    return Bar(open=price, high=price, low=price, close=price, adjusted_close=price, volume=0)


class FinsaberBacktestAdapter:
    """Pull-based, deterministic backtest adapter over a FINSABER data loader."""

    def __init__(self, loader, config: BacktestConfig):
        self._config = config
        window_start = config.preload_date_from or config.date_from
        window = loader.get_subset_by_time_range(window_start, config.date_to)
        if window is None:
            raise ValueError(
                f"No data in window {window_start}..{config.date_to} for the given loader."
            )
        self._window = window
        self._dates = [
            _to_date(d)
            for d in window.get_date_range()
            if config.date_from <= _to_date(d) <= config.date_to
        ]
        if len(self._dates) < 2:
            raise ValueError(
                "Need at least 2 dates in the window for next_open execution; "
                f"got {len(self._dates)}."
            )
        # Own mutable accounting state.
        self._cash: float = float(config.initial_cash)
        self._positions: dict[str, dict] = {}
        self._fills: list[Fill] = []
        self._rejected: list[RejectedOrder] = []
        self._pending: list[Order] = []
        self._equity_curve: list[tuple] = []
        self._cursor: int = 0
        self._finalized: bool = False

    # ---- iteration surface -------------------------------------------------------
    @property
    def dates(self) -> list:
        return list(self._dates)

    def current_date(self) -> Date:
        return self._dates[min(self._cursor, len(self._dates) - 1)]

    @property
    def fills(self) -> list:
        return list(self._fills)

    @property
    def rejected_orders(self) -> list:
        return list(self._rejected)

    @property
    def equity_curve(self) -> list:
        return list(self._equity_curve)

    def open_bar(self, t: Date) -> None:
        """Fill orders queued on the prior date at ``t``'s adjusted_open, then mark equity."""
        t = _to_date(t)
        expected = self._dates[self._cursor]
        if t != expected:
            raise ValueError(f"open_bar({t}) out of order; expected {expected}.")
        pending, self._pending = self._pending, []
        for order in pending:
            self._fill(t, order, price_field="adjusted_open")
        equity = self._account_at(t).equity
        self._equity_curve.append((t, equity))

    def advance(self) -> None:
        if self._cursor < len(self._dates) - 1:
            self._cursor += 1

    # ---- ExecutionAdapter contract ----------------------------------------------
    def get_observation(self, t: Date) -> Observation:
        t = _to_date(t)
        raw_prices = self._window.get_data_by_date(t).get("price", {}) or {}
        bars: dict[str, Bar] = {tk: _bar_from_raw(raw) for tk, raw in raw_prices.items()}
        as_of_dt = datetime.combine(t, time.min)
        return Observation(
            decision_date=t,
            decision_time=as_of_dt,
            availability_time=as_of_dt,  # date-level data is available at the decision date
            bars=bars,
            tickers=tuple(sorted(bars)),
        )

    def submit_order(self, order: Order) -> None:
        """Queue an order for next_open execution (never fills same bar)."""
        self._pending.append(order)

    def get_positions(self) -> PortfolioState:
        positions = [
            Position(ticker=tk, quantity=p["quantity"], avg_price=p["price"])
            for tk, p in sorted(self._positions.items())
        ]
        return PortfolioState(as_of=self.current_date(), positions=positions)

    def get_account(self) -> AccountState:
        return self._account_at(self.current_date())

    # ---- end-of-run --------------------------------------------------------------
    def finalize(self) -> None:
        """Reject orphan pending orders (no future bar) and force-liquidate at last close."""
        if self._finalized:
            return
        last = self._dates[-1]
        for order in self._pending:
            self._rejected.append(
                RejectedOrder(
                    order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    signal_date=order.signal_date,
                    execution_date=None,
                    requested_quantity=order.quantity,
                    reason="no_future_bar",
                )
            )
        self._pending = []
        liquidated = bool(self._positions)
        for ticker in list(self._positions.keys()):
            quantity = self._positions[ticker]["quantity"]
            liquidation = Order(
                order_id=f"liquidate::{ticker}",
                decision_id="finalize",
                signal_date=last,
                ticker=ticker,
                side="sell",
                quantity=quantity,
            )
            self._fill(last, liquidation, price_field="adjusted_close", force=True)
        # Refresh the final equity point so liquidation costs are reflected in the
        # equity curve, keeping curve-based metrics consistent with final_account.
        if liquidated and self._equity_curve and self._equity_curve[-1][0] == last:
            self._equity_curve[-1] = (last, self._account_at(last).equity)
        self._finalized = True

    def result(self, run_id: str) -> BacktestResult:
        final_account = self._account_at(self._dates[-1])
        total_commission = sum(f.commission for f in self._fills)
        total_slippage = sum(f.slippage_cost for f in self._fills)
        equity_values = [e for _, e in self._equity_curve]
        max_dd = self._max_drawdown(equity_values)
        total_return = (final_account.equity / self._config.initial_cash) - 1.0
        summary = {
            "final_equity": final_account.equity,
            "total_return": total_return,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "max_drawdown": max_dd,
            "n_fills": float(len(self._fills)),
            "n_rejected": float(len(self._rejected)),
        }
        return BacktestResult(
            run_id=run_id,
            config_fingerprint=self._config.fingerprint(),
            equity_curve=list(self._equity_curve),
            fills=list(self._fills),
            rejected=list(self._rejected),
            final_account=final_account,
            summary=summary,
        )

    # ---- internal accounting (mirrors FINSABER _execute_order) -------------------
    def _account_at(self, t: Date) -> AccountState:
        positions_value = 0.0
        for ticker, pos in self._positions.items():
            positions_value += pos["quantity"] * self._mark_price(ticker, t, pos["price"])
        return AccountState(
            as_of=t,
            cash=self._cash,
            initial_cash=self._config.initial_cash,
            positions_value=positions_value,
            equity=self._cash + positions_value,
        )

    def _mark_price(self, ticker: str, t: Date, fallback: float) -> float:
        try:
            price = self._window.get_ticker_price_by_date(ticker, t, self._config.price_field)
        except (KeyError, IndexError, TypeError):
            return fallback
        return price if price is not None else fallback

    def _execution_price(self, date: Date, ticker: str, price_field: str, fallback):
        bar = self._window.get_data_by_date(date).get("price", {}).get(ticker)
        if bar is None:
            return fallback
        return adjusted_bar_price(bar, price_field)

    def _resolve_quantity(self, date, side, ticker, requested_quantity, price, force):
        if requested_quantity == ALL_AVAILABLE and side == "buy":
            requested_quantity = int(self._cash / price)
        elif requested_quantity == ALL_AVAILABLE and side == "sell":
            requested_quantity = self._positions.get(ticker, {}).get("quantity", 0)
        else:
            requested_quantity = int(requested_quantity)

        if side == "sell":
            requested_quantity = min(
                requested_quantity, self._positions.get(ticker, {}).get("quantity", 0)
            )

        # Perf: prior-volume stats are only needed when liquidity capping is on, or when slippage
        # has a volume-dependent impact term. When both are off, avg-volume is unused (apply_slippage
        # reduces to slippage_perc; apply_liquidity_cap is a no-op for cap_pct==0), so skip the
        # O(n)-per-order scan. Fill price/commission/quantity are unchanged.
        if self._config.liquidity_cap_pct == 0 and self._config.slippage_impact == 0 and not force:
            average_volume = None
            observations = 0
        else:
            stats = prior_volume_stats(
                self._window, ticker, date, self._config.liquidity_lookback_days
            )
            average_volume = stats["average_volume"]
            observations = stats["observations"]
        if (
            self._config.liquidity_cap_pct > 0
            and not force
            and observations < self._config.liquidity_min_history_days
        ):
            return 0, average_volume, observations, "insufficient_liquidity_history"

        capped = (
            requested_quantity
            if force
            else apply_liquidity_cap(
                requested_quantity,
                average_volume,
                self._config.liquidity_cap_pct,
                require_volume=self._config.liquidity_cap_pct > 0,
            )
        )
        return capped, average_volume, observations, None

    def _fill(self, date: Date, order: Order, price_field: str, force: bool = False) -> bool:
        ticker, side = order.ticker, order.side
        price = self._execution_price(date, ticker, price_field, fallback=order.reference_price)
        if price is None or price <= 0:
            self._reject(order, date, "invalid_price")
            return False

        quantity, average_volume, volume_observations, reject_reason = self._resolve_quantity(
            date, side, ticker, order.quantity, price, force
        )
        if quantity <= 0:
            self._reject(order, date, reject_reason or "zero_quantity")
            return False

        def pricing_for(size):
            fill_price, slippage = apply_slippage(
                price,
                side,
                size,
                average_volume=average_volume,
                slippage_perc=self._config.slippage_perc,
                slippage_impact=self._config.slippage_impact,
            )
            commission = calculate_commission(
                size,
                fill_price,
                commission_per_share=self._config.commission_per_share,
                min_commission=self._config.min_commission,
                max_commission_rate=self._config.max_commission_rate,
            )
            if self._config.execution_cost_bps_per_side > 0:
                commission = size * fill_price * (self._config.execution_cost_bps_per_side / 10_000.0)
            return fill_price, slippage, commission

        fill_price, slippage_cost, commission = pricing_for(quantity)

        if side == "buy":
            while quantity > 0 and quantity * fill_price + commission > self._cash:
                quantity -= 1
                fill_price, slippage_cost, commission = pricing_for(quantity)
            if quantity <= 0:
                self._reject(order, date, "insufficient_cash")
                return False
            self._cash -= quantity * fill_price + commission
            pos = self._positions.setdefault(ticker, {"quantity": 0, "price": fill_price})
            total_cost = pos["quantity"] * pos["price"] + quantity * fill_price
            pos["quantity"] += quantity
            pos["price"] = total_cost / pos["quantity"]
        else:  # sell
            held = self._positions.get(ticker, {}).get("quantity", 0)
            if ticker not in self._positions or held < quantity:
                self._reject(order, date, "insufficient_holdings")
                return False
            self._cash += quantity * fill_price - commission
            self._positions[ticker]["quantity"] -= quantity
            if self._positions[ticker]["quantity"] == 0:
                del self._positions[ticker]

        self._fills.append(
            Fill(
                order_id=order.order_id,
                ticker=ticker,
                side=side,
                signal_date=order.signal_date,
                execution_date=date,
                reference_price=price,
                fill_price=fill_price,
                quantity=quantity,
                commission=commission,
                slippage_cost=slippage_cost,
                average_volume=average_volume,
                volume_observations=volume_observations,
                participation_rate=(
                    abs(quantity) / average_volume if average_volume else None
                ),
            )
        )
        return True

    def _reject(self, order: Order, date: Optional[Date], reason: str) -> None:
        self._rejected.append(
            RejectedOrder(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                signal_date=order.signal_date,
                execution_date=date,
                requested_quantity=order.quantity,
                reason=reason,
            )
        )

    @staticmethod
    def _max_drawdown(equity_values) -> float:
        peak = None
        max_dd = 0.0
        for value in equity_values:
            if peak is None or value > peak:
                peak = value
            if peak and peak > 0:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd
