from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from data_collection.config import Settings, ensure_project_dirs
from data_collection.ib_connector import _import_ib_async, connect_ib, request_daily_bars
from data_collection.symbols import canonical_symbol, ib_contract_symbol


DEFAULT_ACCOUNT_TAGS = (
    "NetLiquidation",
    "AvailableFunds",
    "BuyingPower",
    "TotalCashValue",
    "ExcessLiquidity",
    "GrossPositionValue",
)
DELAYED_MARKET_DATA_TYPE = 3


def _obj_attr(obj, *names: str):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    return Settings(
        ib_host=args.host or settings.ib_host,
        ib_port=int(args.port) if args.port is not None else settings.ib_port,
        ib_client_id=(
            int(args.client_id)
            if args.client_id is not None
            else settings.ib_client_id
        ),
        data_dir=settings.data_dir,
        reports_dir=settings.reports_dir,
        rapidapi_key=settings.rapidapi_key,
        rapidapi_host=settings.rapidapi_host,
    )


def _gateway_payload(settings: Settings) -> dict:
    return {
        "host": settings.ib_host,
        "port": settings.ib_port,
        "client_id": settings.ib_client_id,
    }


def _write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def account_row_to_dict(row) -> dict:
    return {
        "account": _obj_attr(row, "account", "Account"),
        "tag": _obj_attr(row, "tag", "Tag"),
        "value": _obj_attr(row, "value", "Value"),
        "currency": _obj_attr(row, "currency", "Currency"),
    }


def order_to_dict(order) -> dict:
    return {
        "orderId": _obj_attr(order, "orderId", "order_id"),
        "permId": _obj_attr(order, "permId", "perm_id"),
        "clientId": _obj_attr(order, "clientId", "client_id"),
        "action": _obj_attr(order, "action"),
        "orderType": _obj_attr(order, "orderType"),
        "totalQuantity": str(_obj_attr(order, "totalQuantity", "quantity")),
        "lmtPrice": _obj_attr(order, "lmtPrice"),
        "auxPrice": _obj_attr(order, "auxPrice"),
        "tif": _obj_attr(order, "tif"),
    }


def order_status_to_dict(status) -> dict:
    return {
        "status": _obj_attr(status, "status"),
        "filled": str(_obj_attr(status, "filled")),
        "remaining": str(_obj_attr(status, "remaining")),
        "avgFillPrice": _obj_attr(status, "avgFillPrice"),
        "lastFillPrice": _obj_attr(status, "lastFillPrice"),
        "whyHeld": _obj_attr(status, "whyHeld"),
    }


def contract_to_dict(contract) -> dict:
    return {
        "symbol": _obj_attr(contract, "symbol"),
        "secType": _obj_attr(contract, "secType"),
        "exchange": _obj_attr(contract, "exchange"),
        "primaryExchange": _obj_attr(contract, "primaryExchange"),
        "currency": _obj_attr(contract, "currency"),
        "conId": _obj_attr(contract, "conId"),
    }


def ticker_to_dict(ticker) -> dict:
    return {
        "bid": _obj_attr(ticker, "bid"),
        "bidSize": _obj_attr(ticker, "bidSize"),
        "ask": _obj_attr(ticker, "ask"),
        "askSize": _obj_attr(ticker, "askSize"),
        "last": _obj_attr(ticker, "last"),
        "lastSize": _obj_attr(ticker, "lastSize"),
        "close": _obj_attr(ticker, "close"),
        "marketPrice": (
            ticker.marketPrice()
            if hasattr(ticker, "marketPrice")
            else _obj_attr(ticker, "marketPrice")
        ),
    }


def position_to_dict(position) -> dict:
    """One row of ib.positions(): what is held and what it cost."""
    return {
        "account": _obj_attr(position, "account"),
        "contract": contract_to_dict(_obj_attr(position, "contract")),
        "position": _obj_attr(position, "position"),
        "avgCost": _obj_attr(position, "avgCost"),
    }


def portfolio_item_to_dict(item) -> dict:
    """One row of ib.portfolio(): the same holding valued at the current market."""
    return {
        "account": _obj_attr(item, "account"),
        "contract": contract_to_dict(_obj_attr(item, "contract")),
        "position": _obj_attr(item, "position"),
        "marketPrice": _obj_attr(item, "marketPrice"),
        "marketValue": _obj_attr(item, "marketValue"),
        "averageCost": _obj_attr(item, "averageCost"),
        "unrealizedPNL": _obj_attr(item, "unrealizedPNL"),
        "realizedPNL": _obj_attr(item, "realizedPNL"),
    }


def positions_payload(ib) -> dict:
    """Current holdings, from both position and portfolio views.

    ``positions()`` gives size and average cost, ``portfolio()`` adds the current market
    value and unrealised P&L. Both are read-only. An empty list is the normal answer on a
    fresh paper account and is reported as a count rather than as an error.
    """
    positions = list(ib.positions()) if hasattr(ib, "positions") else []
    portfolio = list(ib.portfolio()) if hasattr(ib, "portfolio") else []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability": "paper_positions",
        "status": "ok",
        "position_count": len(positions),
        "portfolio_count": len(portfolio),
        "positions": [position_to_dict(p) for p in positions],
        "portfolio": [portfolio_item_to_dict(i) for i in portfolio],
        "notes": [
            "Read-only: this requests current holdings and does not place or amend orders.",
            "positions() carries size and average cost; portfolio() adds market value and "
            "unrealised P&L, which needs a market-data line for the contract.",
            "Zero rows is the expected result on an account that has not traded.",
        ],
    }


def account_summary_payload(
    ib,
    *,
    tags: Sequence[str] = DEFAULT_ACCOUNT_TAGS,
) -> dict:
    requested_tags = {tag.strip() for tag in tags if tag.strip()}
    rows = [
        account_row_to_dict(row)
        for row in ib.accountSummary()
        if not requested_tags or getattr(row, "tag", None) in requested_tags
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability": "paper_account_summary",
        "status": "ok",
        "tags_requested": list(tags),
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "Paper account summary is available through the same TWS/Gateway API session.",
            "IB documents account-summary updates as an initial snapshot followed by changed values every three minutes.",
        ],
    }


def open_orders_payload(ib, *, all_clients: bool = True) -> dict:
    if all_clients and hasattr(ib, "reqAllOpenOrders"):
        ib.reqAllOpenOrders()
    elif hasattr(ib, "reqOpenOrders"):
        ib.reqOpenOrders()
    if hasattr(ib, "sleep"):
        ib.sleep(1)

    trades = list(ib.openTrades()) if hasattr(ib, "openTrades") else []
    orders = list(ib.openOrders()) if hasattr(ib, "openOrders") else []
    serialized_trades = []
    for trade in trades:
        serialized_trades.append({
            "contract": contract_to_dict(_obj_attr(trade, "contract")),
            "order": order_to_dict(_obj_attr(trade, "order")),
            "orderStatus": order_status_to_dict(_obj_attr(trade, "orderStatus")),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability": "paper_open_orders_and_status",
        "status": "ok",
        "all_clients_requested": bool(all_clients),
        "open_trade_count": len(serialized_trades),
        "open_order_count": len(orders),
        "open_trades": serialized_trades,
        "open_orders": [order_to_dict(order) for order in orders],
        "notes": [
            "This is read-only: it requests currently active orders and status.",
            "IB documents that cancelled or fully filled orders are not available through open-order retrieval.",
        ],
    }


def _qualify_stock(ib, symbol: str):
    _, Stock, _ = _import_ib_async()
    clean_symbol = canonical_symbol(symbol)
    contract = Stock(ib_contract_symbol(clean_symbol), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Could not qualify IB stock contract for {clean_symbol}.")
    return clean_symbol, qualified[0]


def delayed_snapshot_payload(
    ib,
    *,
    symbol: str,
    wait_seconds: float = 2.0,
) -> dict:
    clean_symbol, contract = _qualify_stock(ib, symbol)
    ib.reqMarketDataType(DELAYED_MARKET_DATA_TYPE)
    ticker = ib.reqMktData(contract, "", True, False)
    if hasattr(ib, "sleep"):
        ib.sleep(float(wait_seconds))
    snapshot = ticker_to_dict(ticker)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability": "delayed_market_data_snapshot",
        "status": "ok",
        "symbol": clean_symbol,
        "contract": contract_to_dict(contract),
        "requested_market_data_type": DELAYED_MARKET_DATA_TYPE,
        "requested_market_data_type_label": "delayed",
        "delay_expectation": "usually 15-20 minutes for free delayed market data, depending on exchange/product",
        "snapshot": snapshot,
        "notes": [
            "If live permissions exist, TWS may return live data even after a delayed-data request.",
            "Without live subscriptions, delayed tick fields are normally returned for eligible products.",
        ],
    }


def historical_daily_payload(
    ib,
    *,
    symbol: str,
    duration: str = "7 D",
) -> dict:
    clean_symbol = canonical_symbol(symbol)
    frame = request_daily_bars(ib, clean_symbol, duration=duration)
    records = frame.tail(10).to_dict(orient="records") if not frame.empty else []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability": "historical_daily_bars",
        "status": "ok",
        "symbol": clean_symbol,
        "duration": duration,
        "row_count": int(len(frame)),
        "sample_tail": records,
        "notes": [
            "Historical bars are available from the IB system for supported contracts.",
            "Coverage depends on instrument, exchange, pacing limits, and account permissions.",
        ],
    }


def capabilities_probe_payload(
    ib,
    *,
    symbol: str,
    tags: Sequence[str] = DEFAULT_ACCOUNT_TAGS,
    wait_seconds: float = 2.0,
) -> dict:
    checks = {}
    for name, func in (
        ("account_summary", lambda: account_summary_payload(ib, tags=tags)),
        ("positions", lambda: positions_payload(ib)),
        ("open_orders", lambda: open_orders_payload(ib)),
        (
            "delayed_snapshot",
            lambda: delayed_snapshot_payload(
                ib,
                symbol=symbol,
                wait_seconds=wait_seconds,
            ),
        ),
        ("historical_daily", lambda: historical_daily_payload(ib, symbol=symbol)),
    ):
        try:
            checks[name] = func()
        except Exception as exc:
            checks[name] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": canonical_symbol(symbol),
        "scope": "IBKR system only; excludes external SEC EDGAR/RapidAPI/Finsaber sources",
        "checks": checks,
    }


def _with_ib(func, args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    ensure_project_dirs(settings)
    ib = connect_ib(settings)
    try:
        payload = func(ib, args)
        payload["gateway"] = _gateway_payload(settings)
    finally:
        ib.disconnect()
    _write_json(args.output, payload)
    print(f"WROTE {args.output}", flush=True)
    return 0


def summary_command(args: argparse.Namespace) -> int:
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    return _with_ib(lambda ib, _args: account_summary_payload(ib, tags=tags), args)


def orders_command(args: argparse.Namespace) -> int:
    return _with_ib(
        lambda ib, _args: open_orders_payload(ib, all_clients=args.all_clients),
        args,
    )


def positions_command(args: argparse.Namespace) -> int:
    return _with_ib(lambda ib, _args: positions_payload(ib), args)


def delayed_command(args: argparse.Namespace) -> int:
    return _with_ib(
        lambda ib, _args: delayed_snapshot_payload(
            ib,
            symbol=args.symbol,
            wait_seconds=args.wait_seconds,
        ),
        args,
    )


def history_command(args: argparse.Namespace) -> int:
    return _with_ib(
        lambda ib, _args: historical_daily_payload(
            ib,
            symbol=args.symbol,
            duration=args.duration,
        ),
        args,
    )


def probe_command(args: argparse.Namespace) -> int:
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    return _with_ib(
        lambda ib, _args: capabilities_probe_payload(
            ib,
            symbol=args.symbol,
            tags=tags,
            wait_seconds=args.wait_seconds,
        ),
        args,
    )


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only IBKR paper data probes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Export paper account summary rows")
    _add_connection_args(summary)
    summary.add_argument("--tags", default=",".join(DEFAULT_ACCOUNT_TAGS))
    summary.set_defaults(func=summary_command)

    orders = subparsers.add_parser("orders", help="Export open orders and order status")
    _add_connection_args(orders)
    orders.add_argument("--all-clients", action=argparse.BooleanOptionalAction, default=True)
    orders.set_defaults(func=orders_command)

    positions = subparsers.add_parser("positions", help="Export current holdings and P&L")
    _add_connection_args(positions)
    positions.set_defaults(func=positions_command)

    delayed = subparsers.add_parser("delayed", help="Export a delayed market data snapshot")
    _add_connection_args(delayed)
    delayed.add_argument("--symbol", default="AAPL")
    delayed.add_argument("--wait-seconds", type=float, default=2.0)
    delayed.set_defaults(func=delayed_command)

    history = subparsers.add_parser("history", help="Export recent historical daily bars")
    _add_connection_args(history)
    history.add_argument("--symbol", default="AAPL")
    history.add_argument("--duration", default="7 D")
    history.set_defaults(func=history_command)

    probe = subparsers.add_parser("probe", help="Run all read-only paper data probes")
    _add_connection_args(probe)
    probe.add_argument("--symbol", default="AAPL")
    probe.add_argument("--tags", default=",".join(DEFAULT_ACCOUNT_TAGS))
    probe.add_argument("--wait-seconds", type=float, default=2.0)
    probe.set_defaults(func=probe_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
