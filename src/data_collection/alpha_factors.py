from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import (
    connect_sqlite,
    list_price_symbols,
    load_price_frame,
    upsert_long_values,
)
from data_collection.symbols import canonical_symbol


def require_python_313() -> None:
    if sys.version_info < (3, 13):
        raise RuntimeError(
            "Alpha101 requires Python >= 3.13. Create and activate the "
            "Python 3.13 environment before running this command."
        )


def prepare_alpha101_input(
    prices_db: Path,
    *,
    symbols: list[str] | None = None,
    industry_csv: Path | None = None,
) -> pd.DataFrame:
    if not prices_db.exists():
        raise FileNotFoundError(
            f"Price database not found: {prices_db}. Run IB collection first."
        )
    conn = connect_sqlite(prices_db)
    try:
        selected = symbols or list_price_symbols(conn)
        frames = []
        for symbol in selected:
            frame = load_price_frame(conn, symbol)
            if frame.empty:
                continue
            frames.append(frame)
    finally:
        conn.close()

    if not frames:
        raise ValueError("No price data found for Alpha101 input.")

    data = pd.concat(frames, ignore_index=True)
    data["symbol"] = data["symbol"].map(canonical_symbol)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["symbol", "date"])
    for column in ("open", "high", "low", "close", "volume", "average"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["return"] = data.groupby("symbol")["close"].pct_change()
    data["vwap"] = data["average"]
    missing_vwap = data["vwap"].isna()
    data.loc[missing_vwap, "vwap"] = (
        data.loc[missing_vwap, ["open", "high", "low", "close"]].mean(axis=1)
    )
    data["market_value"] = pd.NA
    data["industry"] = "UNKNOWN"

    if industry_csv is not None and industry_csv.exists():
        industries = pd.read_csv(industry_csv)
        symbol_col = next(
            (col for col in industries.columns if col.lower() == "symbol"), None
        )
        industry_col = next(
            (
                col
                for col in industries.columns
                if col.lower() in {"industry", "sector", "gics sector"}
            ),
            None,
        )
        if symbol_col and industry_col:
            mapping = {
                canonical_symbol(row[symbol_col]): row[industry_col]
                for _, row in industries.iterrows()
            }
            data["industry"] = data["symbol"].map(mapping).fillna("UNKNOWN")

    required = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_value",
        "return",
        "vwap",
        "industry",
    ]
    return data[required].set_index(["symbol", "date"]).sort_index()


def build_alpha101(
    prices_db: Path,
    output_db: Path,
    *,
    symbols: list[str] | None = None,
    industry_csv: Path | None = None,
    skip_version_check: bool = False,
) -> int:
    if not prices_db.exists():
        raise FileNotFoundError(
            f"Price database not found: {prices_db}. Run IB collection first."
        )
    if not skip_version_check:
        require_python_313()
    try:
        from alpha101 import Alphas
    except ImportError as exc:
        raise RuntimeError(
            "alpha101 is not installed. Run: pip install -r "
            "requirements-alpha101-py313.txt"
        ) from exc

    data = prepare_alpha101_input(prices_db, symbols=symbols, industry_csv=industry_csv)
    alphas = Alphas(data)
    rows: list[tuple[str, str, str, float | None]] = []

    for index in range(1, 102):
        method = getattr(alphas, f"alpha_{index}", None)
        if method is None:
            continue
        try:
            values = method()
        except NotImplementedError:
            continue
        series = values.stack() if isinstance(values, pd.DataFrame) else values
        if not isinstance(series.index, pd.MultiIndex):
            continue
        for (symbol, row_date), value in series.items():
            rows.append(
                (
                    canonical_symbol(symbol),
                    pd.to_datetime(row_date).strftime("%Y-%m-%d"),
                    f"alpha_{index}",
                    None if pd.isna(value) else float(value),
                )
            )

    conn = connect_sqlite(output_db)
    try:
        return upsert_long_values(conn, "alphas", "alpha_id", rows)
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha101 factor builder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build Alpha101 DB from price DB")
    build.add_argument("--prices", default=str(Settings.from_env().prices_db))
    build.add_argument("--output", default=str(Settings.from_env().alphas_db))
    build.add_argument("--symbols", nargs="*")
    build.add_argument("--industry-csv")
    build.add_argument(
        "--skip-version-check",
        action="store_true",
        help="For tests only. The official pipeline requires Python 3.13.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_project_dirs(Settings.from_env())
    if args.command == "build":
        try:
            total = build_alpha101(
                Path(args.prices),
                Path(args.output),
                symbols=args.symbols,
                industry_csv=Path(args.industry_csv) if args.industry_csv else None,
                skip_version_check=args.skip_version_check,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {total} Alpha101 rows total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
