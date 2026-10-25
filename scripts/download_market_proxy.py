"""Download the SPY market proxy (dividend-adjusted daily close) for single-stock runs.

FINSABER ships only S&P 500 constituents, no index/ETF, so for single-name testing
the market-state label and the market-adjusted attribution baseline need an external
market series. SPY (dividend-adjusted, so it is on the same footing as the stocks'
adjusted_close) is the standard S&P 500 market proxy.

Usage:
    python scripts/download_market_proxy.py [--start 2010-01-01] [--end 2025-07-01]
Writes data/market_proxy/spy_daily.parquet with columns (date, adjusted_close).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-07-01")
    parser.add_argument("--output", default="data/market_proxy/spy_daily.parquet")
    args = parser.parse_args(argv)

    import pandas as pd
    import yfinance as yf

    df = yf.download(args.symbol, start=args.start, end=args.end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"no data returned for {args.symbol}")
    close = df["Close"]
    if hasattr(close, "columns"):  # yfinance can return a single-column frame
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": df.index.date, "adjusted_close": close.values})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"WROTE {path} ({len(out)} rows, {out['date'].iloc[0]} -> {out['date'].iloc[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
