from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import connect_sqlite, get_last_price_date, upsert_price_frame
from data_collection.symbols import (
    canonical_symbol,
    ib_contract_symbol,
)
from data_collection.universe import load_universe_snapshot, refresh_sp500_snapshot


def _import_ib_async():
    try:
        from ib_async import IB, Stock, util
    except ImportError as exc:
        raise RuntimeError(
            "ib_async is not installed. Run: pip install ib_async"
        ) from exc
    return IB, Stock, util


def bars_to_frame(bars, util_module) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    df = util_module.df(bars)
    if df is None:
        return pd.DataFrame()
    return df


def connect_ib(settings: Settings):
    IB, _, _ = _import_ib_async()
    ib = IB()
    ib.connect(settings.ib_host, settings.ib_port, clientId=settings.ib_client_id)
    if not ib.isConnected():
        raise RuntimeError("IB connection failed.")
    return ib


def request_daily_bars(
    ib,
    symbol: str,
    *,
    duration: str,
    end_datetime: str = "",
    use_rth: bool = True,
) -> pd.DataFrame:
    _, Stock, util = _import_ib_async()
    contract = Stock(ib_contract_symbol(symbol), "SMART", "USD")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_datetime,
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
    )
    return bars_to_frame(bars, util)


def duration_for_symbol(conn, symbol: str, years: int) -> str:
    last_date = get_last_price_date(conn, symbol)
    if not last_date:
        return f"{years} Y"

    last = datetime.strptime(last_date, "%Y-%m-%d").date()
    days_missing = max((date.today() - last).days + 3, 1)
    return f"{days_missing} D"


def smoke_test(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    ensure_project_dirs(settings)
    symbol = canonical_symbol(args.symbol)
    duration = f"{args.days} D"

    if args.dry_run:
        print(
            f"DRY RUN: would connect to {settings.ib_host}:{settings.ib_port} "
            f"clientId={settings.ib_client_id} and request {duration} for {symbol}."
        )
        return 0

    ib = connect_ib(settings)
    try:
        df = request_daily_bars(ib, symbol, duration=duration)
        if df.empty:
            print(f"No bars returned for {symbol}.")
            return 1
        df["symbol"] = symbol
        print(df.to_string(index=False))
        return 0
    finally:
        ib.disconnect()


def fetch_sp500(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    ensure_project_dirs(settings)

    if args.refresh_universe:
        _, _, universe_path = refresh_sp500_snapshot(settings)
        print(f"Refreshed universe snapshot: {universe_path}")
    elif args.source_csv:
        universe_path = Path(args.source_csv)
    else:
        universe_path = settings.sp500_universe_snapshot
        if not universe_path.exists():
            _, _, universe_path = refresh_sp500_snapshot(settings)
            print(f"Bootstrapped universe snapshot: {universe_path}")

    constituents = load_universe_snapshot(universe_path)
    if args.limit:
        constituents = constituents[: args.limit]

    if args.dry_run:
        print(f"DRY RUN: prepared {len(constituents)} S&P 500 symbols.")
        print(f"Universe snapshot: {universe_path}")
        for constituent in constituents[:10]:
            print(f"{constituent.symbol} -> IB '{constituent.ib_symbol}'")
        return 0

    conn = connect_sqlite(Path(args.output))
    ib = connect_ib(settings)
    errors: list[dict[str, str]] = []
    try:
        for index, constituent in enumerate(constituents, start=1):
            duration = duration_for_symbol(conn, constituent.symbol, args.years)
            print(
                f"[{index}/{len(constituents)}] Fetching {constituent.symbol} "
                f"({duration})"
            )
            try:
                df = request_daily_bars(ib, constituent.symbol, duration=duration)
                written = upsert_price_frame(conn, constituent.symbol, df)
                print(f"  wrote {written} rows")
            except Exception as exc:
                error = {
                    "symbol": constituent.symbol,
                    "ib_symbol": constituent.ib_symbol,
                    "error": repr(exc),
                }
                errors.append(error)
                print(f"  ERROR {constituent.symbol}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
            time.sleep(args.sleep)
    finally:
        ib.disconnect()
        conn.close()

    if errors:
        error_path = settings.reports_dir / "ib_collection_errors.json"
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Wrote {len(errors)} collection errors to {error_path}")
    return 0 if args.continue_on_error or not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Brokers data tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-test", help="Fetch a short daily-bar sample")
    smoke.add_argument("--symbol", default="AAPL")
    smoke.add_argument("--days", type=int, default=7)
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(func=smoke_test)

    fetch = subparsers.add_parser("fetch-sp500", help="Fetch S&P 500 daily OHLCV")
    fetch.add_argument("--years", type=int, default=5)
    fetch.add_argument("--output", default=str(Settings.from_env().prices_db))
    fetch.add_argument("--sleep", type=float, default=1.0)
    fetch.add_argument("--limit", type=int)
    fetch.add_argument("--source-csv")
    fetch.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Refresh the local S&P 500 universe snapshot from SPY holdings first.",
    )
    fetch.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue collecting remaining symbols when one symbol fails.",
    )
    fetch.add_argument("--dry-run", action="store_true")
    fetch.set_defaults(func=fetch_sp500)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
