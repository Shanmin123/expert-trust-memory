from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import (
    connect_sqlite,
    list_price_symbols,
    load_price_frame,
    upsert_long_values,
)
from data_collection.symbols import canonical_symbol


def _rolling_mean_deviation(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute past-only technical indicators for one symbol.

    All rolling windows use min_periods equal to the window length. Initial
    values stay NaN rather than being forward-filled.
    """
    if price_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "indicator_name", "value"])

    df = price_df.copy().sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)
    symbol = canonical_symbol(df["symbol"].dropna().iloc[0])

    indicators: dict[str, pd.Series] = {}

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    indicators["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    indicators["MACD_12_26_9"] = macd
    indicators["MACD_SIGNAL_12_26_9"] = macd.ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    indicators["MACD_HIST_12_26_9"] = (
        indicators["MACD_12_26_9"] - indicators["MACD_SIGNAL_12_26_9"]
    )

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    indicators["BB_MIDDLE_20_2"] = sma20
    indicators["BB_UPPER_20_2"] = sma20 + 2 * std20
    indicators["BB_LOWER_20_2"] = sma20 - 2 * std20

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    indicators["ATR_14"] = true_range.rolling(14, min_periods=14).mean()

    direction = np.sign(close.diff()).fillna(0)
    indicators["OBV"] = (direction * volume).cumsum()

    for window in (20, 50, 200):
        indicators[f"SMA_{window}"] = close.rolling(
            window, min_periods=window
        ).mean()
    indicators["EMA_12"] = ema12
    indicators["EMA_26"] = ema26
    indicators["ROC_10"] = close.pct_change(10) * 100

    lowest14 = low.rolling(14, min_periods=14).min()
    highest14 = high.rolling(14, min_periods=14).max()
    range14 = (highest14 - lowest14).replace(0, np.nan)
    stoch_k = ((close - lowest14) / range14) * 100
    indicators["STOCH_K_14_3"] = stoch_k
    indicators["STOCH_D_14_3"] = stoch_k.rolling(3, min_periods=3).mean()
    indicators["WILLIAMS_R_14"] = -100 * ((highest14 - close) / range14)

    typical = (high + low + close) / 3
    typical_sma = typical.rolling(20, min_periods=20).mean()
    mean_dev = _rolling_mean_deviation(typical, 20)
    indicators["CCI_20"] = (typical - typical_sma) / (0.015 * mean_dev)

    long_rows = []
    for name, values in indicators.items():
        for row_date, value in zip(df["date"], values):
            if pd.isna(value):
                stored_value = None
            else:
                stored_value = float(value)
                if not math.isfinite(stored_value):
                    stored_value = None
            long_rows.append(
                {
                    "date": row_date,
                    "symbol": symbol,
                    "indicator_name": name,
                    "value": stored_value,
                }
            )

    return pd.DataFrame(long_rows)


def build_indicators(
    prices_db: Path,
    output_db: Path,
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> int:
    if not prices_db.exists():
        raise FileNotFoundError(
            f"Price database not found: {prices_db}. Run IB collection first."
        )
    price_conn = connect_sqlite(prices_db)
    out_conn = connect_sqlite(output_db)
    try:
        selected = symbols or list_price_symbols(price_conn)
        selected = [canonical_symbol(symbol) for symbol in selected]
        if limit is not None:
            selected = selected[:limit]

        total = 0
        for symbol in selected:
            prices = load_price_frame(price_conn, symbol)
            indicators = compute_indicators(prices)
            rows = [
                (row.symbol, row.date, row.indicator_name, row.value)
                for row in indicators.itertuples(index=False)
            ]
            total += upsert_long_values(out_conn, "indicators", "indicator_name", rows)
            print(f"{symbol}: wrote {len(rows)} indicator rows")
        return total
    finally:
        price_conn.close()
        out_conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Technical indicator builder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build indicator DB from price DB")
    build.add_argument("--prices", default=str(Settings.from_env().prices_db))
    build.add_argument("--output", default=str(Settings.from_env().indicators_db))
    build.add_argument("--symbols", nargs="*")
    build.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_project_dirs(Settings.from_env())
    if args.command == "build":
        try:
            total = build_indicators(
                Path(args.prices),
                Path(args.output),
                symbols=args.symbols,
                limit=args.limit,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {total} indicator rows total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
