from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import (
    connect_sqlite,
    count_rows,
    list_price_symbols,
    load_price_frame,
    quote_identifier,
    table_exists,
)
from data_collection.symbols import canonical_symbol, sqlite_table_name


QLIB_MARKET = "sp500"
QLIB_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "factor")


def qlib_table_name(symbol: str) -> str:
    return sqlite_table_name("qlib_direct", symbol)


def write_bin(path: Path, start_index: int, values: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.hstack([[float(start_index)], values.astype("float32").to_numpy()])
    data.astype("<f").tofile(path)


def load_all_prices(prices_db: Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    conn = connect_sqlite(prices_db)
    try:
        selected = symbols or list_price_symbols(conn)
        out = {}
        for symbol in selected:
            frame = load_price_frame(conn, symbol)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["symbol"] = frame["symbol"].map(canonical_symbol)
            frame["date"] = pd.to_datetime(frame["date"])
            for column in ("open", "high", "low", "close", "volume", "average"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["vwap"] = frame.get("average", pd.Series(index=frame.index)).fillna(
                frame[["open", "high", "low", "close"]].mean(axis=1)
            )
            frame["factor"] = 1.0
            out[canonical_symbol(symbol)] = frame.sort_values("date")
        return out
    finally:
        conn.close()


def export_qlib_provider(
    prices_db: Path,
    provider_dir: Path,
    *,
    market: str = QLIB_MARKET,
    symbols: list[str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    prices = load_all_prices(prices_db, symbols)
    if not prices:
        raise ValueError(f"No price data found in {prices_db}")

    if force and provider_dir.exists():
        shutil.rmtree(provider_dir)

    calendars_dir = provider_dir / "calendars"
    instruments_dir = provider_dir / "instruments"
    features_dir = provider_dir / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    calendar = pd.Index(
        sorted(set().union(*(set(frame["date"]) for frame in prices.values())))
    )
    calendar_strings = [day.strftime("%Y-%m-%d") for day in calendar]
    (calendars_dir / "day.txt").write_text("\n".join(calendar_strings) + "\n", encoding="utf-8")
    calendar_index = {day: idx for idx, day in enumerate(calendar)}

    instrument_lines = []
    for symbol, frame in sorted(prices.items()):
        start_date = frame["date"].min()
        end_date = frame["date"].max()
        instrument_lines.append(
            f"{symbol}\t{start_date.strftime('%Y-%m-%d')}\t{end_date.strftime('%Y-%m-%d')}"
        )

        start_index = calendar_index[start_date]
        symbol_calendar = calendar[
            (calendar >= start_date)
            & (calendar <= end_date)
        ]
        indexed = frame.set_index("date").reindex(symbol_calendar)
        symbol_dir = features_dir / symbol.lower()
        for field in QLIB_FIELDS:
            write_bin(symbol_dir / f"{field}.day.bin", start_index, indexed[field])

    text = "\n".join(instrument_lines) + "\n"
    (instruments_dir / f"{market}.txt").write_text(text, encoding="utf-8")
    (instruments_dir / "all.txt").write_text(text, encoding="utf-8")
    return len(prices), len(calendar)


def init_qlib(provider_dir: Path):
    import qlib
    from qlib.config import REG_US

    qlib.init(
        provider_uri=str(provider_dir),
        region=REG_US,
        expression_cache=None,
        dataset_cache=None,
    )


def smoke_test_provider(provider_dir: Path, *, market: str = QLIB_MARKET) -> dict[str, object]:
    init_qlib(provider_dir)
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
    from qlib.data import D

    instruments = D.list_instruments(D.instruments(market), as_list=True)
    if not instruments:
        raise RuntimeError(f"No instruments found for market={market}")
    symbol = "AAPL" if "AAPL" in instruments else instruments[0]
    fields_158, names_158 = Alpha158DL.get_feature_config()
    fields_360, names_360 = Alpha360DL.get_feature_config()
    sample = D.features(
        [symbol],
        fields_158[:5] + fields_360[:5],
        freq="day",
        disk_cache=0,
    )
    return {
        "market": market,
        "provider_dir": str(provider_dir),
        "instruments": len(instruments),
        "sample_symbol": symbol,
        "alpha158_features": len(names_158),
        "alpha360_features": len(names_360),
        "sample_shape": tuple(sample.shape),
    }


def compute_direct_qlib_features(
    provider_dir: Path,
    output_db: Path,
    *,
    market: str = QLIB_MARKET,
    symbols: list[str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    init_qlib(provider_dir)
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
    from qlib.data import D

    fields_158, names_158 = Alpha158DL.get_feature_config()
    fields_360, names_360 = Alpha360DL.get_feature_config()
    fields = list(fields_158) + list(fields_360)
    names = [f"QLIB158_{name}" for name in names_158] + [
        f"QLIB360_{name}" for name in names_360
    ]

    all_symbols = D.list_instruments(D.instruments(market), as_list=True)
    selected = [canonical_symbol(symbol) for symbol in (symbols or all_symbols)]
    if force and output_db.exists():
        output_db.unlink()

    conn = connect_sqlite(output_db)
    total_rows = 0
    built_symbols = 0
    try:
        for index, symbol in enumerate(selected, start=1):
            table = qlib_table_name(symbol)
            if not force and table_exists(conn, table) and count_rows(conn, table) > 0:
                continue

            frame = D.features([symbol], fields, freq="day", disk_cache=0)
            if frame.empty:
                print(f"[{index}/{len(selected)}] {symbol}: no Qlib features")
                continue
            frame = frame.copy()
            frame.columns = names
            frame = frame.reset_index()
            frame = frame.rename(columns={"instrument": "symbol", "datetime": "date"})
            frame["symbol"] = frame["symbol"].map(canonical_symbol)
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frame.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {quote_identifier('idx_' + table + '_date')} "
                f"ON {quote_identifier(table)} (date)"
            )
            conn.commit()
            built_symbols += 1
            total_rows += len(frame)
            print(f"[{index}/{len(selected)}] {symbol}: wrote {len(frame)} official Qlib rows")
    finally:
        conn.close()
    return built_symbols, total_rows


def export_command(args: argparse.Namespace) -> int:
    symbols = [canonical_symbol(symbol) for symbol in args.symbols] if args.symbols else None
    count, calendar_len = export_qlib_provider(
        Path(args.prices),
        Path(args.output),
        market=args.market,
        symbols=symbols,
        force=args.force,
    )
    print(f"Exported {count} instruments to Qlib provider: {args.output}")
    print(f"Calendar rows: {calendar_len}")
    return 0


def smoke_command(args: argparse.Namespace) -> int:
    result = smoke_test_provider(Path(args.provider), market=args.market)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def build_factors_command(args: argparse.Namespace) -> int:
    symbols = [canonical_symbol(symbol) for symbol in args.symbols] if args.symbols else None
    built, rows = compute_direct_qlib_features(
        Path(args.provider),
        Path(args.output),
        market=args.market,
        symbols=symbols,
        force=args.force,
    )
    print(f"Wrote official Qlib factors for {built} symbols and {rows} rows.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Qlib provider and official factor tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-provider")
    export.add_argument("--prices", default=str(settings.prices_db))
    export.add_argument("--output", default=str(settings.qlib_provider_dir))
    export.add_argument("--market", default=QLIB_MARKET)
    export.add_argument("--symbols", nargs="*")
    export.add_argument("--force", action="store_true")
    export.set_defaults(func=export_command)

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--provider", default=str(settings.qlib_provider_dir))
    smoke.add_argument("--market", default=QLIB_MARKET)
    smoke.set_defaults(func=smoke_command)

    build = subparsers.add_parser("build-factors")
    build.add_argument("--provider", default=str(settings.qlib_provider_dir))
    build.add_argument("--output", default=str(settings.qlib_direct_factors_db))
    build.add_argument("--market", default=QLIB_MARKET)
    build.add_argument("--symbols", nargs="*")
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=build_factors_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_project_dirs(Settings.from_env())
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
