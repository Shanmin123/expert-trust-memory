"""Reconstruct FinMem's daily exposure from its equity curve, for description only.

``finmem_baseline.py`` derives returns from the equity curve backtrader produced and
writes a constant ``1.0`` in the exposure slot, because the curve does not carry
positions. That placeholder is wrong: on TSLA, the first completed stock, FinMem was in
cash on 42% of the days the stock moved. Reading the placeholder as participation or
turnover would misdescribe the baseline.

For a single-name strategy the exposure on day t satisfies ``r_strategy = e * r_stock``,
so ``e = r_strategy / r_stock`` wherever the stock actually moved. The ratio is unusable
when ``r_stock`` is near zero, so days below ``MOVE`` are filled by carrying the previous
estimate forward.

The reconstruction is DESCRIPTIVE. Returns stay as the equity curve reported them, which
means costs stay as FinMem's own engine charged them and are counted exactly once. Do not
re-price from the reconstructed exposure — that would charge this project's ten basis
points on top of commissions FinMem already paid.

    PYTHONPATH=src python scripts/finmem_implied_exposure.py
    PYTHONPATH=src python scripts/finmem_implied_exposure.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from trading_agent.paths import slim_prices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baselines"
PANEL = Path(str(slim_prices()))

MOVE = 0.01      # only days the stock moved more than this give a usable ratio
CLIP = (0.0, 1.0)  # the wrapper runs long-or-cash; values outside are estimator noise


def stock_returns(ticker: str, dates: list[str]) -> pd.Series:
    """Adjusted-close returns for one name, on the series' own dates."""
    import pyarrow.dataset as ds

    table = ds.dataset(str(PANEL), format="parquet", partitioning="hive").to_table(
        filter=(ds.field("symbol") == ticker)
    )
    df = table.to_pandas()
    df["d"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("d").set_index("d")
    col = next(c for c in ("adjusted_close", "adj_close", "close") if c in df.columns)
    return df[col].pct_change().reindex(dates)


def implied_exposure(strategy: pd.Series, stock: pd.Series) -> tuple[pd.Series, dict]:
    """e = r_strategy / r_stock on days the stock moved; carried forward elsewhere."""
    usable = stock.abs() > MOVE
    raw = (strategy / stock).where(usable)
    # Shares are OF THE USABLE DAYS. Taking them over the whole series would silently
    # divide by 351 while only 272 days can contribute, understating every bucket.
    seen = raw.dropna()
    n = max(len(seen), 1)
    diagnostics = {
        "days_usable": int(usable.sum()),
        "days_total": int(len(stock)),
        "share_long": float((seen > 0.1).sum() / n),
        "share_cash": float((seen.abs() <= 0.1).sum() / n),
        "share_negative": float((seen < -0.1).sum() / n),
    }
    filled = raw.ffill().bfill().fillna(0.0)
    return filled.clip(*CLIP), diagnostics


def process(path: Path, dry: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ticker = payload["manifest"]["ticker"]
    (arm,) = payload["daily_arm_series"].keys()
    rows = payload["daily_arm_series"][arm]

    dates = [r[0] for r in rows]
    strategy = pd.Series([r[1] for r in rows], index=dates)
    exposure, diag = implied_exposure(strategy, stock_returns(ticker, dates))

    payload["daily_arm_series"][arm] = [
        [d, r, float(exposure[d])] for d, r, _ in rows
    ]
    # Once the exposure varies, a scorer that charges on exposure changes would add this
    # project's ten basis points on top of the commissions already inside the curve. The
    # declaration is what stops it.
    payload["manifest"]["returns"] = "net"
    payload["manifest"]["exposure"] = {
        "source": "reconstructed from the equity curve, descriptive only",
        "estimator": "r_strategy / r_stock where |r_stock| > %.3f, carried forward" % MOVE,
        "costs": "FinMem's own commission model, already inside the returns",
        **diag,
    }
    if not dry:
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {"ticker": ticker, **diag}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(OUT.glob("*__FinMem.json"))
    if not files:
        raise SystemExit(f"no FinMem results in {OUT}")

    print(f"{'stock':6} {'usable':>7} {'long':>7} {'cash':>7} {'neg':>7}")
    for path in files:
        d = process(path, args.dry_run)
        print(f"{d['ticker']:6} {d['days_usable']:7d} {100*d['share_long']:6.0f}% "
              f"{100*d['share_cash']:6.0f}% {100*d['share_negative']:6.0f}%")
    if args.dry_run:
        print("\ndry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
