"""Rule-based baselines on the sealed eight stocks, scored the way the arms are scored.

Runs FINSABER's own timing strategies over the sealed test window and writes one
report-shaped JSON per strategy, so `scripts/portfolio_arm_metrics.py` consumes them
unchanged and the comparison is like-for-like: same window, same 10 bps per side charged
on exposure changes, same paired block bootstrap.

    PYTHONPATH=src python scripts/baseline_arms.py
    PYTHONPATH=src python scripts/baseline_arms.py --strategies BuyAndHoldStrategy

Four things about FINSABER's runner would silently corrupt the numbers if left alone,
and each is handled here rather than in a comment:

1. ``date_to`` is EXCLUSIVE, so the window end is passed one day late.
2. Bars are calendar days: the loader reindexes onto a daily frequency and fills
   forward, turning ~373 sessions into ~541 rows. Sharpe and annualised return computed
   on that are understated by roughly 15%. This module reads the equity curve and keeps
   only real trading dates before computing anything.
3. Commission is per-share, not basis points. The cap is saturated so the effective
   charge is exactly 10 bps per side.
4. ``max_drawdown`` comes back as a percent while this project uses a fraction.

Only the equity curve is taken from FINSABER. Every reported statistic is recomputed
here on trading-day returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_agent._bootstrap import ensure_finsaber_on_path
from trading_agent.paths import slim_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baselines"
DATA_ROOT = str(slim_panel())

TICKERS = ["TSLA", "ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT"]
DATE_FROM = "2024-07-09"
DATE_TO_EXCLUSIVE = "2026-01-01"   # trap 1: the runner drops rows on the boundary
SEALED_LAST_DAY = "2025-12-01"     # the arms are scored to here; match them exactly
BPS_PER_SIDE = 10.0

#: FINSABER's own schedule, which is Interactive Brokers' fixed tier and what the
#: benchmark charges by default. On these position sizes it comes to about 0.4 basis
#: points a side, so the flat rate below was roughly twenty-five times it and fell
#: hardest on whichever strategy traded most.
COMMISSION_MODEL = {"commission_per_share": 0.0049, "min_commission": 0.99,
                    "max_commission_rate": 0.01}


def flat_bps_model(bps: float) -> dict:
    """The same three parameters, saturated so the cap binds on every order."""
    return {"commission_per_share": 1e9, "min_commission": 0.0,
            "max_commission_rate": bps / 1e4}


COST_MODELS = {"commission": COMMISSION_MODEL, "bps": flat_bps_model(BPS_PER_SIDE)}

STRATEGIES = [
    "BuyAndHoldStrategy",
    "SMACrossStrategy",
    "WMAStrategy",
    "BollingerBandsStrategy",
    "ATRBandStrategy",
    "TrendFollowingStrategy",
    "TurnOfTheMonthStrategy",
]


def build_config(loader, tickers, cost_model="bps"):
    return {
        "data_loader": loader,
        "tickers": list(tickers),
        "date_from": DATE_FROM,
        "date_to": DATE_TO_EXCLUSIVE,
        "setup_name": "rule_baselines_8",
        "save_results": False,
        "silence": True,          # trap: otherwise plt.show() blocks once per ticker
        # trap 3: the engine only has a per-share schedule, so the flat rate is
        # expressed by saturating it until the cap binds on every order
        **COST_MODELS[cost_model],
    }


def session_only_price_frame():
    """Feed the engine exchange sessions instead of a forward-filled calendar.

    ``operation_utils._prepare_price_frame`` reindexes the pivot onto ``freq="D"`` and
    ``add_tickers_data`` then forward- and back-fills it, so backtrader receives about
    541 bars where the window holds 373 sessions. The strategies act on those filled
    bars: the 2026-08-02 audit counted weekend executions in every rule family (347
    SMA, 375 WMA, 254 turn-of-month, 81 trend, 39 ATR, 16 Bollinger). Dropping the
    filled rows from the OUTPUT, which the earlier version did, cannot undo a trade
    that already happened.

    The fix belongs on our side of the boundary. FINSABER's strategy classes and their
    default parameters are untouched; only the calendar handed to them is corrected,
    by dropping the rows the reindex invented. A row is invented when every price in
    it was absent before the fill, which is exactly the reindex's own footprint.
    """
    import pandas as pd

    from backtest.toolkit import operation_utils as ou

    original = ou._prepare_price_frame
    if getattr(original, "_session_only", False):
        return

    def patched(df, date_from, date_to, return_original=False):
        out = original(df, date_from, date_to, return_original=return_original)
        # return_original hands back the long-form frame on an integer index, which
        # carries no invented rows and must pass through untouched.
        if out is None or return_original:
            return out
        real = pd.DatetimeIndex(pd.to_datetime(df["date"])).normalize()
        return out.loc[out.index.normalize().isin(real)]

    patched._session_only = True
    ou._prepare_price_frame = patched


def equity_to_series(equity_df, ordered_dates, terminal_date=None):
    """Decision-day (date, return, exposure) rows from FINSABER's equity curve.

    Three corrections make this comparable to the sealed arms. The curve is indexed on
    calendar days with weekends filled forward, so only real sessions are kept. A
    strategy with a warm-up produces no equity for its first sessions; it holds nothing
    then, so those days are carried at zero return and zero exposure. And each row is
    labelled by its decision date and carries the move to the NEXT session, the arms'
    own convention; ``terminal_date`` is one session past the last decision, used only
    to price that last decision's move. The 2026-07-31 fairness audit found the old
    realisation-day labelling discarded the first session's move and the last
    decision's, both of which the arms score.
    """
    wanted = set(ordered_dates)
    if terminal_date:
        wanted.add(terminal_date)
    by_date = {}
    for _, r in equity_df.iterrows():
        d = str(r.get("datetime", r.get("date", "")))[:10]
        if d in wanted:
            by_date[d] = float(r["equity"])

    seq = list(ordered_dates) + ([terminal_date] if terminal_date else [])
    out = []
    for i, d in enumerate(ordered_dates):
        eq0 = by_date.get(d)
        eq1 = by_date.get(seq[i + 1]) if i + 1 < len(seq) else None
        if eq0 is None or eq1 is None or eq0 <= 0:
            out.append([d, 0.0, 0.0])          # not yet trading: flat, no exposure
        else:
            out.append([d, eq1 / eq0 - 1.0, 1.0])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=None, help="comma list; default is all seven")
    ap.add_argument("--tickers", default=None,
                    help="comma list; default is the eight sealed stocks")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cost-model", choices=sorted(COST_MODELS), default="bps",
                    help="commission = FINSABER's own per-share schedule; "
                         "bps = the flat rate saturated into it")
    args = ap.parse_args(argv)

    tickers = ([s.strip().upper() for s in args.tickers.split(",")]
               if args.tickers else list(TICKERS))

    ensure_finsaber_on_path()
    from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset
    from backtest.finsaber_bt import FINSABERBt
    import backtest.strategy.timing as timing

    names = ([s.strip() for s in args.strategies.split(",")]
             if args.strategies else list(STRATEGIES))

    session_only_price_frame()

    loader = FinsaberParquetDataset(args.data_root, tickers=tickers, modalities=("price",))
    all_dates = sorted({str(d)[:10] for d in loader.get_date_range()})
    trading_dates = [d for d in all_dates if DATE_FROM <= d <= SEALED_LAST_DAY]
    later = [d for d in all_dates if d > SEALED_LAST_DAY]
    terminal_date = later[0] if later else None
    print(f"{len(trading_dates)} trading sessions in {DATE_FROM}..{SEALED_LAST_DAY}, "
          f"terminal pricing date {terminal_date}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        strategy = getattr(timing, name, None)
        if strategy is None:
            print(f"  {name:26s} NOT FOUND in backtest.strategy.timing")
            continue
        try:
            results = FINSABERBt(build_config(loader, tickers, args.cost_model)
                                 ).run_iterative_tickers(strategy)
        except Exception as exc:
            print(f"  {name:26s} FAILED {type(exc).__name__}: {str(exc)[:60]}")
            continue

        series, kept = {}, []
        for ticker in tickers:
            payload = results.get(ticker) or {}
            eq = payload.get("equity_with_time")
            if eq is None or getattr(eq, "empty", True):
                continue
            rows = equity_to_series(eq, trading_dates, terminal_date)
            if rows:
                series[ticker] = rows
                kept.append(ticker)

        for ticker in kept:
            path = OUT / f"{ticker}__{name}.json"
            path.write_text(json.dumps({
                "mode": "baseline",
                "run_label": f"{ticker}__{name}",
                "daily_arm_series": {"equal_weight": series[ticker]},
                "manifest": {"strategy": name, "ticker": ticker,
                             "window": [DATE_FROM, SEALED_LAST_DAY],
                             "cost_model": args.cost_model,
                             "cost_bps_per_side": (BPS_PER_SIDE
                                                   if args.cost_model == "bps" else None),
                             "commission": COST_MODELS[args.cost_model],
                             # backtrader already charged BPS_PER_SIDE inside this curve,
                             # so the scorer must not charge it a second time
                             "returns": "net",
                             "source": "FINSABER backtest.strategy.timing",
                             "note": "returns recomputed on trading days from equity_with_time"},
            }, indent=1), encoding="utf-8")
        n = len(series.get(kept[0], [])) if kept else 0
        print(f"  {name:26s} {len(kept)}/8 stocks, {n} scored days each")

    print(f"\nwritten to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
