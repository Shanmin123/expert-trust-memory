"""Per-stock comparison against the two published agents, plus the under-exposure check.

``baseline_table.py`` gives the portfolio-level comparison across every baseline. This is
the per-stock view the write-up needs for the two agents specifically, because their
interesting property is behavioural rather than terminal: how often they hold anything.

Cost handling, the date basis and the net/gross declaration all come from
``baseline_table`` so there is exactly one cost path. FinMem's returns arrive already net
of its own engine's commissions and its exposure column is reconstructed for description
only (``finmem_implied_exposure.py``); TradingAgents is priced here from the shared panel
and is charged this project's ten basis points.

The diagnostic at the end tests the reading that a cash-heavy agent loses to a rising
stock by construction: it regresses each agent's shortfall against buy and hold on the
stock's own return. A slope near -1 means the shortfall is exposure, not judgement. The
project's rule applies -- any cross-stock claim gets a leave-one-out check before it is
believed.

    PYTHONPATH=src python scripts/agent_baseline_table.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import fmean, stdev

import pandas as pd
from trading_agent.paths import slim_prices

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "baseline_table", Path(__file__).resolve().parent / "baseline_table.py")
bt = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bt
_SPEC.loader.exec_module(bt)

PANEL = Path(str(slim_prices()))
AGENTS = ("FinMem", "TradingAgents")
TICKERS = list(bt.SEALED)


def load_agent(strategy: str, ticker: str):
    """(net daily returns by date, exposure by date, manifest) or None if absent."""
    path = bt.BASE / f"{ticker}__{strategy}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    basis = payload.get("manifest", {}).get("returns")
    if basis not in ("net", "gross"):
        raise SystemExit(f"{path.name}: manifest must declare returns as net or gross")
    rows = payload["daily_arm_series"]["equal_weight"]
    return (bt.net(rows, charge=(basis == "gross")),
            {r[0]: float(r[2]) for r in rows},
            payload.get("manifest", {}))


def buy_and_hold(ticker: str, dates: list[str]) -> float:
    import pyarrow.dataset as ds
    table = ds.dataset(str(PANEL), format="parquet", partitioning="hive").to_table(
        filter=(ds.field("symbol") == ticker))
    df = table.to_pandas()
    df["d"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["d"].isin(dates)].sort_values("d")
    col = next(c for c in ("adjusted_close", "adj_close", "close") if c in df.columns)
    return df[col].iloc[-1] / df[col].iloc[0] - 1.0


def arm_sharpes(ticker: str) -> dict:
    payload = json.loads((bt.FINAL / f"{bt.SEALED[ticker]}.json").read_text(encoding="utf-8"))
    arms = payload["arms"]
    return {
        "equal": arms["equal_weight"]["metrics"]["sharpe"],
        "best": max(arms[k]["metrics"]["sharpe"]
                    for k in ("equal_weight", "global_memory", "market_state_memory")),
    }


def slope_through(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of y on x. Nothing fancy; the point is the sign and scale.

    Exact-zero is the wrong guard on the denominator. Stocks whose returns differ only in
    floating-point noise give a denominator near 1e-34 and an arbitrary slope rather than
    a refusal, so the spread is required to be real relative to the scale of x.
    """
    if len(xs) < 2:
        return float("nan")
    mx, my = fmean(xs), fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    scale = max((abs(x) for x in xs), default=0.0) or 1.0
    if den <= (1e-9 * scale) ** 2 * len(xs):
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=bt.MIN_DAYS)
    args = ap.parse_args(argv)

    for strategy in AGENTS:
        print(f"\n=== {strategy} ===")
        rows, gaps, bh_returns = [], [], []
        for ticker in TICKERS:
            got = load_agent(strategy, ticker)
            if got is None:
                print(f"  {ticker:6} not run yet")
                continue
            nets, exposure, manifest = got
            if len(nets) < args.min_days:
                print(f"  {ticker:6} INCOMPLETE, {len(nets)} days")
                continue
            rets = list(nets.values())
            cr = bt.total_return(rets)
            bh = buy_and_hold(ticker, list(nets))
            arms = arm_sharpes(ticker)
            held = fmean(1.0 if e > 0.5 else 0.0 for e in exposure.values())
            rows.append((ticker, cr, bh, cr - bh, bt.sharpe(rets), bt.max_drawdown(rets),
                         held, arms["equal"], arms["best"]))
            gaps.append(cr - bh)
            bh_returns.append(bh)

        if not rows:
            continue
        print(f"  {'stock':6}{'CR%':>8}{'B&H%':>8}{'gap pp':>8}{'SR':>8}{'MDD%':>7}"
              f"{'held':>7}{'eq SR':>8}{'best SR':>9}  beats")
        for t, cr, bh, gap, sr, mdd, held, eq, best in rows:
            print(f"  {t:6}{100*cr:8.2f}{100*bh:8.2f}{100*gap:8.1f}{sr:8.3f}{100*mdd:7.2f}"
                  f"{100*held:6.0f}%{eq:8.3f}{best:9.3f}  "
                  f"{'agent' if sr > best else 'this work'}")
        beats = sum(1 for r in rows if r[4] > r[8])
        print(f"  agent beats this system's best arm on {beats} of {len(rows)}; "
              f"mean held {100*fmean(r[6] for r in rows):.0f}%, "
              f"range {100*min(r[6] for r in rows):.0f}-{100*max(r[6] for r in rows):.0f}%")

        if len(rows) >= 4:
            s = slope_through(bh_returns, gaps)
            print(f"\n  shortfall against buy and hold, regressed on the stock's own return:")
            print(f"    slope {s:+.2f} over {len(rows)} stocks "
                  f"(-1 would mean the shortfall is entirely exposure)")
            loo = [slope_through([b for j, b in enumerate(bh_returns) if j != i],
                                 [g for j, g in enumerate(gaps) if j != i])
                   for i in range(len(rows))]
            print(f"    leave-one-out slope range {min(loo):+.2f} to {max(loo):+.2f}"
                  f"{'  -- survives every removal' if max(loo) < 0 else '  -- NOT robust'}")
            if len(rows) > 2:
                print(f"    sd of the gaps {stdev(gaps):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
