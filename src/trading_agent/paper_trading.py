"""Interactive Brokers paper-trading bridge utilities.

The module keeps order conversion and safety checks deterministic and testable.
Actual Gateway I/O is injected so CLI dry-runs never need a live connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from data_collection.symbols import canonical_symbol, ib_contract_symbol
from trading_agent.contracts import ALL_AVAILABLE, OrderIntent
from trading_agent.reporting import REPORT_SCHEMA_VERSION, SYSTEM_IDENTITY


COMMON_IB_PAPER_PORTS = (4002, 7497)


@dataclass(frozen=True)
class IBGatewayConfig:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1


def parse_order_spec(spec: str) -> OrderIntent:
    """Parse ``SYMBOL:side:quantity`` into an ``OrderIntent``."""
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid order spec {spec!r}; expected SYMBOL:side:quantity.")
    symbol, side, quantity_text = parts
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"Invalid order side {side!r}; expected buy or sell.")
    quantity = ALL_AVAILABLE if quantity_text.upper() == "ALL" else int(quantity_text)
    return OrderIntent(ticker=canonical_symbol(symbol), side=side, quantity=quantity)


def build_order_previews(
    intents: Iterable[OrderIntent],
    *,
    allow_all_available_sell: bool = False,
) -> list[dict]:
    """Convert project order intents to auditable IB market-order previews."""
    previews: list[dict] = []
    for intent in intents:
        if intent.quantity == ALL_AVAILABLE and intent.side == "buy":
            raise ValueError("ALL_AVAILABLE buy is not valid for IB paper orders.")
        if intent.quantity == ALL_AVAILABLE and not allow_all_available_sell:
            raise ValueError("ALL_AVAILABLE sell requires allow_all_available_sell=True.")
        if intent.quantity <= 0 and intent.quantity != ALL_AVAILABLE:
            raise ValueError(f"Quantity must be positive for {intent.ticker}.")
        previews.append({
            "symbol": canonical_symbol(intent.ticker),
            "ib_symbol": ib_contract_symbol(intent.ticker),
            "action": intent.side.upper(),
            "quantity": "ALL" if intent.quantity == ALL_AVAILABLE else int(intent.quantity),
            "order_type": "MKT",
            "time_in_force": "DAY",
            "transmit": True,
        })
    return previews


def build_paper_manifest(
    *,
    config: IBGatewayConfig,
    symbols: Sequence[str],
    order_intents: Sequence[OrderIntent],
    dry_run: bool,
    execute_orders: bool,
    gateway_connected: bool = False,
    latest_bars: dict | None = None,
    orders_submitted: int = 0,
    require_memory: bool = False,
    connection_smoke: bool = False,
    non_paper_port_override: bool = False,
    launcher: str = "trading-agent paper",
) -> dict:
    """Create the paper-mode audit payload shared by CLI and future adapters."""
    previews = build_order_previews(order_intents, allow_all_available_sell=True)
    preview_count = len(previews)
    submitted_count = int(orders_submitted)
    clean_symbols = [canonical_symbol(symbol) for symbol in symbols]
    pending_submission = max(0, preview_count - submitted_count)
    submission_coverage = (
        submitted_count / preview_count
        if preview_count
        else 0.0
    )
    if connection_smoke:
        run_intent = "connection_smoke"
    elif dry_run:
        run_intent = "dry_run"
    elif execute_orders:
        run_intent = "paper_order_submission"
    else:
        run_intent = "gateway_data_smoke"
    if dry_run:
        validation_status = "dry_run_only"
    elif connection_smoke:
        validation_status = "connection_smoke_verified" if gateway_connected else "connection_smoke_failed"
    elif execute_orders and submitted_count > 0:
        validation_status = "paper_order_submitted"
    elif gateway_connected:
        validation_status = "gateway_connected_no_submission"
    else:
        validation_status = "not_validated"
    ib_paper_validation = {
        "requires_external_gateway": True,
        "status": validation_status,
        "connection_verified": bool(gateway_connected),
        "actual_order_submission": bool(execute_orders and submitted_count > 0),
    }
    return {
        "mode": "paper",
        "system": SYSTEM_IDENTITY,
        "schema_version": REPORT_SCHEMA_VERSION,
        "dry_run": bool(dry_run),
        "host": config.host,
        "port": config.port,
        "client_id": config.client_id,
        "symbols": clean_symbols,
        "execute_orders": bool(execute_orders),
        "memory_required": bool(require_memory),
        "gateway_connected": bool(gateway_connected),
        "manifest": {
            "mode": "paper",
            "system": SYSTEM_IDENTITY,
            "schema_version": REPORT_SCHEMA_VERSION,
            "launcher": launcher,
            "host": config.host,
            "port": config.port,
            "client_id": config.client_id,
            "symbols": clean_symbols,
            "dry_run": bool(dry_run),
            "execute_orders": bool(execute_orders),
            "memory_required": bool(require_memory),
            "gateway_connected": bool(gateway_connected),
            "connection_smoke": bool(connection_smoke),
            "run_intent": run_intent,
            "common_paper_port": config.port in COMMON_IB_PAPER_PORTS,
            "allowed_paper_ports": list(COMMON_IB_PAPER_PORTS),
            "non_paper_port_override": bool(non_paper_port_override),
            "order_preview_count": preview_count,
            "orders_submitted": submitted_count,
            "orders_pending_submission": pending_submission,
            "submission_coverage": submission_coverage,
            "ib_connector_source": "data_collection.ib_connector",
            "external_validation_steps": [
                "connection_smoke",
                "memory_dry_run",
                "paper_order_submission",
            ],
            "ib_paper_validation": ib_paper_validation,
        },
        "latest_bars": latest_bars or {},
        "order_previews": previews,
        "order_preview_count": preview_count,
        "orders_submitted": submitted_count,
        "orders_pending_submission": pending_submission,
        "submission_coverage": submission_coverage,
    }


def latest_close_from_frame(frame) -> float:
    """Extract the latest close/adjusted close from an IB historical bar frame."""
    if getattr(frame, "empty", False):
        raise ValueError("Cannot build paper decision from an empty bar frame.")
    for column in ("close", "adjusted_close"):
        try:
            series = frame[column]
        except (KeyError, TypeError):
            continue
        value = series.iloc[-1]
        if value is not None and float(value) > 0:
            return float(value)
    raise ValueError("Bar frame has no positive close price.")


def auto_decide_order_intents(
    symbols: Sequence[str],
    latest_bar_frames: dict[str, object],
    *,
    paper_cash: float,
    deployment_fraction: float = 1.0,
    max_order_value: float | None = None,
    held: Mapping[str, float] | None = None,
) -> list[OrderIntent]:
    """Turn the committee's long list into orders against what the account already holds.

    ``symbols`` is the set the committee calls long. ``held`` maps symbol to the
    quantity currently owned; pass it whenever the account has been read.

    Exposure in this system is binary and the maintain rule holds a position instead of
    re-opening it, so the four cases are: long and unheld buys, long and held does
    nothing, held and no longer long sells to flat, and neither does nothing. Without
    ``held`` the account was not read, and the old buy-only behaviour is kept rather
    than guessing at a position (before 2026-08-03 that was the only behaviour, so a
    ten-day run bought the same stock ten times and never exited).
    """
    long_symbols = [canonical_symbol(symbol) for symbol in symbols]
    positions = {canonical_symbol(k): float(v) for k, v in (held or {}).items()}
    intents: list[OrderIntent] = []

    # Exits first: a stock the committee no longer calls long is closed at its own size,
    # which costs no cash and must not depend on the buy budget.
    if held is not None:
        for symbol, quantity in sorted(positions.items()):
            if symbol in long_symbols or quantity <= 0:
                continue  # a short was not opened here and is left alone
            if int(quantity) > 0:
                intents.append(
                    OrderIntent(ticker=symbol, side="sell", quantity=int(quantity))
                )

    to_buy = [s for s in long_symbols if positions.get(s, 0.0) <= 0]
    if paper_cash <= 0 or not to_buy:
        return intents
    deploy = max(0.0, min(1.0, float(deployment_fraction)))
    budget_each = (float(paper_cash) * deploy) / len(to_buy)
    if max_order_value is not None and max_order_value > 0:
        budget_each = min(budget_each, float(max_order_value))
    for symbol in to_buy:
        frame = latest_bar_frames.get(symbol)
        if frame is None:
            continue
        close = latest_close_from_frame(frame)
        quantity = int(budget_each / close)
        if quantity > 0:
            intents.append(OrderIntent(ticker=symbol, side="buy", quantity=quantity))
    return intents


def account_cash_from_summary(rows, *, currency: str = "USD") -> float:
    """Extract deployable paper cash from IB account summary rows."""
    values: dict[str, float] = {}
    for row in rows:
        row_currency = getattr(row, "currency", "")
        if row_currency and row_currency != currency:
            continue
        tag = getattr(row, "tag", "")
        value = getattr(row, "value", None)
        if tag in {"AvailableFunds", "NetLiquidation"} and value not in (None, ""):
            values[tag] = float(value)
    if "AvailableFunds" in values:
        return values["AvailableFunds"]
    if "NetLiquidation" in values:
        return values["NetLiquidation"]
    raise ValueError(f"No {currency} AvailableFunds or NetLiquidation in IB account summary.")


def _import_ib_order_types():
    try:
        from ib_async import MarketOrder, Stock
    except ImportError as exc:
        raise RuntimeError("ib_async is not installed. Run: pip install ib_async") from exc
    return Stock, MarketOrder


def _trade_status_report(trade) -> tuple[int | None, str | None]:
    status = getattr(trade, "orderStatus", None)
    order_id = getattr(status, "orderId", None)
    status_text = getattr(status, "status", None)
    return order_id, status_text


def submit_order_previews(
    ib,
    previews: Sequence[dict],
    *,
    stock_factory=None,
    market_order_factory=None,
) -> list[dict]:
    """Submit audited market-order previews through an injected IB client."""
    if stock_factory is None or market_order_factory is None:
        stock_factory, market_order_factory = _import_ib_order_types()

    reports: list[dict] = []
    for preview in previews:
        quantity = preview["quantity"]
        if quantity == "ALL":
            raise ValueError("Cannot submit ALL quantity to IB without live position resolution.")
        contract = stock_factory(preview["ib_symbol"], "SMART", "USD")
        # MarketOrder already sets orderType="MKT"; passing it again raises
        # "got multiple values for keyword argument 'orderType'". The preview's
        # order_type is asserted below instead, so a non-market preview cannot be
        # submitted through this path by accident (found 2026-08-03, on the first
        # real submission: every order-placing process failed here).
        wanted = preview.get("order_type", "MKT")
        if wanted != "MKT":
            raise ValueError(
                f"{preview['symbol']}: this path submits market orders only, "
                f"and the preview asks for {wanted!r}."
            )
        order = market_order_factory(
            preview["action"],
            int(quantity),
            tif=preview.get("time_in_force", "DAY"),
        )
        trade = ib.placeOrder(contract, order)
        order_id, status = _trade_status_report(trade)
        reports.append({
            "symbol": preview["symbol"],
            "ib_symbol": preview["ib_symbol"],
            "action": preview["action"],
            "quantity": int(quantity),
            "order_id": order_id,
            "status": status,
        })
    return reports
