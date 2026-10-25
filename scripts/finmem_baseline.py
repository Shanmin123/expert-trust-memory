"""Run FinMem as a baseline on the sealed stocks, and emit the arms' series shape.

FinMem is the published agent baseline. It cannot run in the project environment:
it pins guardrails-ai 0.3.2, which pins pydantic 2.4.2, against this project's 2.13, so
it lives in its own interpreter. Run it with that one, not with the project's:

    python -m trading_agent.paths        # prints where the FinMem interpreter resolves
    <that interpreter> scripts/finmem_baseline.py --smoke
    <that interpreter> scripts/finmem_baseline.py

--smoke runs one ticker over one month and costs a few cents; it exists to prove the
pipeline before the full run, which is thousands of completions and takes hours.

Output is one report-shaped JSON per ticker with a ``daily_arm_series`` of
``[date, return, exposure]``, so scripts/baseline_table.py scores it through the same
cost model as every other arm. Nothing here places an order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from trading_agent.paths import slim_panel

ROOT = Path(__file__).resolve().parents[1]
FINSABER_DIR = ROOT / "external" / "FINSABER"
OUT = ROOT / "reports" / "baselines"
DATA_ROOT = str(slim_panel())

TICKERS = ["TSLA", "ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT"]
# DATE_TO is one session past the sealed last day (2025-12-01): the arms score the
# 2025-12-01 decision by the move into 2025-12-02, so the engine must hold through it.
DATE_FROM, DATE_TO = "2024-07-09", "2025-12-02"
INITIAL_CASH = 100000.0
ATTEMPTS = 3            # a transient API error otherwise costs the whole ticker
BACKOFF_SECONDS = 60
# Two years rather than three: it is 60-70% of the spend and the shorter window is
# defensible on its own terms, since the memory's own decay half-life is 63 episodes.
TRAINING_YEARS = 2
CONFIG = FINSABER_DIR / "strats_configs" / "finmem_gpt_config.toml"


def _chat_model() -> str:
    """The model the config actually names; a hardcoded string here lied once."""
    import toml

    return toml.load(CONFIG)["chat"]["model"]


def load_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                return
    raise SystemExit("OPENAI_API_KEY not set and not found in .env")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one ticker, one month, a few cents; proves the pipeline")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="re-run tickers that already have a saved result")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(FINSABER_DIR))
    sys.path.insert(0, str(FINSABER_DIR / "llm_traders" / "finmem"))
    os.chdir(FINSABER_DIR)          # every config path in FINSABER is CWD-relative
    load_key()

    from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset
    from backtest.finsaber import FINSABER
    from llm_traders.finsaber_strategies.finmem import FinMemStrategy

    tickers = ([t.strip().upper() for t in args.tickers.split(",")] if args.tickers
               else (["BIIB"] if args.smoke else TICKERS))
    date_from, date_to = (("2024-07-09", "2024-08-09") if args.smoke
                          else (DATE_FROM, DATE_TO))
    training = ("2024-04-09", "2024-07-08") if args.smoke else TRAINING_YEARS

    loader = FinsaberParquetDataset(args.data_root, tickers=tickers,
                                    modalities=("price", "news", "filing_k", "filing_q"))
    config = {
        "data_loader": loader, "tickers": tickers,
        "date_from": date_from, "date_to": date_to,
        "setup_name": "finmem_baseline", "save_results": False, "silence": True,
        "cash": INITIAL_CASH,       # pinned so the exposure replay's seed matches
        # Saturate the per-share schedule to a flat 10 bps per side, the same trick
        # the rule baselines use. Without these the engine charges $0.0049 a share,
        # far below every other row's cost.
        "commission_per_share": 1e9,
        "min_commission": 0.0,
        "max_commission_rate": 10.0 / 1e4,
    }
    # The runner substitutes "$name" placeholders per ticker (auto_resolve_params); it
    # does not inject anything the caller has not asked for, so symbol is declared here.
    strat_params = {
        "symbol": "$symbol",
        "config_path": str(CONFIG),
        "date_from": date_from, "date_to": date_to,
        "data_loader": loader, "training_period": training,
    }

    print(f"FinMem baseline: {len(tickers)} ticker(s), {date_from}..{date_to}, "
          f"training={training}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # One ticker per call, saving as each finishes. The full run is hours long, so a
    # crash on the seventh name must not discard the first six; a finished name is also
    # skipped on a re-run, which makes the whole thing resumable.
    for ticker in tickers:
        target = OUT / f"{ticker}__FinMem.json"
        if target.exists() and not args.force:
            print(f"  {ticker:6s} already done, skipping", flush=True)
            continue
        # A ticker takes hours and a single transient API error discards all of it. CVX
        # and DHI were both lost that way, back to back, to one network blip. Retry
        # before giving the ticker up.
        results = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                results = FINSABER({**config, "tickers": [ticker]}).run_iterative_tickers(
                    FinMemStrategy, strat_params=strat_params, tickers=[ticker])
                break
            except Exception as exc:
                label = f"{type(exc).__name__}: {str(exc)[:70]}"
                if attempt == ATTEMPTS:
                    print(f"  {ticker:6s} FAILED after {ATTEMPTS} attempts, {label}",
                          flush=True)
                else:
                    wait = BACKOFF_SECONDS * attempt
                    print(f"  {ticker:6s} attempt {attempt} hit {label}; "
                          f"retrying in {wait}s", flush=True)
                    time.sleep(wait)
        if results is None:
            continue

        res = ((results or {}).get(ticker) or {})
        eq = res.get("equity_with_time")
        if eq is None or getattr(eq, "empty", True):
            print(f"  {ticker:6s} no equity curve returned", flush=True)
            continue
        eq_by_date = {}
        for _, r in eq.iterrows():
            eq_by_date[str(r.get("datetime", r.get("date", "")))[:10]] = float(r["equity"])

        # The full trade log makes the artifact self-describing: exact exposure now,
        # and re-pricing under any future cost model without another API run.
        trades_df = res.get("trades")
        trades = ([] if trades_df is None or getattr(trades_df, "empty", True) else
                  json.loads(trades_df.assign(
                      execution_date=trades_df["execution_date"].astype(str).str[:10],
                      signal_date=trades_df["signal_date"].astype(str).str[:10],
                  )[["signal_date", "execution_date", "type", "price", "quantity",
                     "commission"]].to_json(orient="records")))

        # Cash after each session, replayed from the trade log; the strategy is
        # all-in/all-out, so exposure(d) = 1 - cash(d)/equity(d).
        cash_moves = {}
        for t in trades:
            delta = (-(t["quantity"] * t["price"] + t["commission"])
                     if t["type"] == "buy"
                     else t["quantity"] * t["price"] - t["commission"])
            cash_moves.setdefault(t["execution_date"], 0.0)
            cash_moves[t["execution_date"]] += delta

        # Decision-date rows on the equity calendar: row d earns the move into the
        # next session, the arms' own convention; the last decision is priced by the
        # extra session DATE_TO adds past the sealed last day.
        dates = sorted(eq_by_date)
        rows, cash = [], INITIAL_CASH
        for i, d in enumerate(dates[:-1]):
            cash += cash_moves.get(d, 0.0)   # executions on d have settled by its close
            exposure = 1.0 - cash / eq_by_date[d] if eq_by_date[d] > 0 else 0.0
            nxt = eq_by_date[dates[i + 1]]
            ret = nxt / eq_by_date[d] - 1.0 if eq_by_date[d] > 0 else 0.0
            rows.append([d, ret, round(max(0.0, min(1.0, exposure)), 6)])

        target.write_text(json.dumps({
            "mode": "baseline", "run_label": f"{ticker}__FinMem",
            "daily_arm_series": {"equal_weight": rows},
            "trades": trades,
            "manifest": {"strategy": "FinMem", "ticker": ticker,
                         "window": [date_from, date_to],
                         "training_period": str(training),
                         "model": _chat_model(),
                         "returns": "net",
                         "commission_model": "saturated to 10 bps per side of notional",
                         "row_convention": "decision date, earns the next session",
                         "source": "external/FINSABER llm_traders/finmem"},
        }, indent=1), encoding="utf-8")
        print(f"  {ticker:6s} {len(rows)} scored days, {len(trades)} trades -> saved",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
