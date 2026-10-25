"""Reproducibly sample a random test universe with full expert-input coverage.

The four reference names (TSLA/NFLX/AMZN/MSFT) are the FinMem/TradingAgents demo
set: hand-picked mega-cap tech, correlated, and exactly the names a reviewer
would call cherry-picked. This script draws additional names at random from the
pool that actually has the inputs every expert needs, so the evaluation gains a
genuine external-validity set.

Eligibility (a name must clear all three to enter the pool):
- PRICE coverage: present for >= --min-coverage of trading days over the window
  (the survivorship-limited screen from select_fixed_universe; labeled honestly).
- NEWS coverage: >= --min-news-dates dated news days, so the news expert is live
  on a meaningful fraction of days (threshold defaults near the weakest
  reference name, AMZN ~950).
- FACTORS are built later (build_qlib_factors --symbols); FUNDAMENTAL filings are
  broad S&P 500 coverage. Neither gates the pool here.

Selection is a fixed-seed random sample, so the set is reproducible and
reportable. The reference names are excluded from the draw.

    PYTHONPATH=src python scripts/select_random_universe.py --n 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from trading_agent.data.finsaber_news import FinsaberNewsPanel
from trading_agent.data.finsaber_panel import load_panel

REFERENCE_NAMES = ("TSLA", "NFLX", "AMZN", "MSFT")


def _price_covered_names(root: str, start: date, end: date, min_coverage: float) -> dict[str, float]:
    """Symbol -> price coverage fraction over [start, end], for names >= min_coverage."""
    panel = load_panel(root, start, end, tickers=None, modalities=("price",))
    df = panel.get_price_dataframe(tickers="all", date_from=start, date_to=end, adjust=True)
    if df.empty:
        raise ValueError(f"No price data in window {start}..{end}.")
    total_days = df["date"].nunique()
    coverage = df.groupby("symbol")["date"].nunique() / total_days
    return {str(sym): float(cov) for sym, cov in coverage.items() if cov >= min_coverage}


def _news_covered_names(root: str, start: date, end: date, min_news_dates: int) -> dict[str, int]:
    """Symbol -> news-date count over [start, end], for names >= min_news_dates."""
    panel = FinsaberNewsPanel(root, start, end, tickers=None)
    return {
        str(sym): len(dates)
        for sym, dates in panel.dates_by_ticker.items()
        if len(dates) >= min_news_dates
    }


def select_random_universe(
    root: str,
    *,
    start: date,
    end: date,
    n: int,
    seed: int,
    min_coverage: float,
    min_news_dates: int,
    exclude: tuple[str, ...] = REFERENCE_NAMES,
) -> dict:
    price = _price_covered_names(root, start, end, min_coverage)
    news = _news_covered_names(root, start, end, min_news_dates)
    pool = sorted((set(price) & set(news)) - {e.upper() for e in exclude})
    if len(pool) < n:
        raise ValueError(f"eligible pool has only {len(pool)} names; need >= {n}")
    chosen = sorted(random.Random(seed).sample(pool, n))
    return {
        "selected": chosen,
        "reference_excluded": list(exclude),
        "pool_size": len(pool),
        "seed": seed,
        "window": [start.isoformat(), end.isoformat()],
        "thresholds": {"min_coverage": min_coverage, "min_news_dates": min_news_dates},
        "coverage": {
            name: {"price_coverage": round(price[name], 4), "news_dates": news[name]}
            for name in chosen
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data/finsaber/sp500_2000_2025_parquet")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2011, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-news-dates", type=int, default=900)
    parser.add_argument("--output", type=Path, default=Path("reports/universe/random_universe.json"))
    args = parser.parse_args(argv)

    result = select_random_universe(
        args.data_root, start=args.start, end=args.end, n=args.n, seed=args.seed,
        min_coverage=args.min_coverage, min_news_dates=args.min_news_dates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"eligible pool: {result['pool_size']} names "
          f"(price>={args.min_coverage:.0%}, news>={args.min_news_dates} dates)")
    print(f"random sample (seed={args.seed}, n={args.n}):")
    for name in result["selected"]:
        c = result["coverage"][name]
        print(f"  {name:6s} price_coverage={c['price_coverage']:.1%} news_dates={c['news_dates']}")
    print(f"manifest -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
