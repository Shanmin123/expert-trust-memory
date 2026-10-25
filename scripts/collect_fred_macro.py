"""Collect the FRED macro series used by the design-D news expert (TA-style macro block).

TradingAgents' news analyst grounds macro commentary in FRED indicators; this collector
downloads the same style of series through FRED's free public CSV endpoint (no API key):

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

Series (daily unless noted): DFF (fed funds, rate-hike cycle), T10Y2Y (10y-2y curve),
CPIAUCSL (CPI, monthly), ICSA (initial claims, weekly), UNRATE (unemployment, monthly).

Output: long-format parquet (series, date, value) at data/macro/fred_macro.parquet.

POINT-IN-TIME NOTE (documented limitation): these are latest-revision FRED series, not
ALFRED vintages. Rates/curve are never revised; CPI/UNRATE/ICSA carry small revisions.
The runtime loader additionally applies a PUBLICATION LAG per series so a monthly value
is not readable on its own observation date.

    PYTHONPATH=src python scripts/collect_fred_macro.py [--out data/macro/fred_macro.parquet]
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

SERIES = ("DFF", "T10Y2Y", "CPIAUCSL", "ICSA", "UNRATE")
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_series(sid: str) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(sid=sid), timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    frame = pd.read_csv(io.StringIO(raw))
    date_col = frame.columns[0]          # DATE (older) or observation_date (newer export)
    frame = frame.rename(columns={date_col: "date", sid: "value"})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["series"] = sid
    return frame[["series", "date", "value"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/macro/fred_macro.parquet")
    args = parser.parse_args(argv)
    frames = []
    for sid in SERIES:
        frame = fetch_series(sid)
        print(f"{sid}: {len(frame)} rows {frame['date'].min()} .. {frame['date'].max()}")
        frames.append(frame)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
