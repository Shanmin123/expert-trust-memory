"""Run TradingAgents as a baseline on the sealed stocks and window.

TradingAgents has no FINSABER adapter, so it is driven through its own published API:
one ``propagate(ticker, date)`` call per decision, with no training phase. It fetches its
own price and news through its yfinance vendors; that difference from this project's
FINSABER panel is a real limit on the comparison and is stated in the write-up rather
than hidden.

Model tier is matched to this project's own experts (gpt-4.1-mini) so the comparison is
not decided by giving one side a stronger model.

Measured cost before writing this: one decision takes about six minutes, so 352 days on
eight stocks is roughly 280 hours sequentially. Tickers therefore run as separate
processes, and each decision is written as it completes so the run is resumable.

    python -m trading_agent.paths        # prints where the TradingAgents interpreter is
    <that interpreter> scripts/tradingagents_baseline.py --ticker BIIB
    <that interpreter> scripts/tradingagents_baseline.py --launch-all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from trading_agent.paths import tradingagents_python

ROOT = Path(__file__).resolve().parents[1]
TA_DIR = ROOT / "external" / "research_refs" / "TradingAgents"
OUT = ROOT / "reports" / "baselines" / "tradingagents"
VENV_PY = Path(str(tradingagents_python()))

TICKERS = ["TSLA", "ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT"]
MODEL = "gpt-4.1-mini"          # matched to this project's own expert model


def load_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("OPENAI_API_KEY not found")


def trading_dates() -> list[str]:
    """The sealed scored dates, taken from a sealed report so the windows match."""
    import json as _json
    p = ROOT / "reports" / "final" / "BIIB__designD3_q_h21cert.json"
    series = _json.loads(p.read_text(encoding="utf-8"))["daily_arm_series"]["equal_weight"]
    return [row[0] for row in series]


def run_ticker(ticker: str, dates: list[str], limit: int | None) -> int:
    sys.path.insert(0, str(TA_DIR))
    os.chdir(TA_DIR)
    load_key()

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["deep_think_llm"] = MODEL
    config["quick_think_llm"] = MODEL
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    # Thirteen concurrent processes sharing ~/.tradingagents/cache produced 48
    # WinError 5/32 collisions, each costing a six-minute decision slot. A cache per
    # ticker keeps the reuse that matters (one ticker's history across its dates).
    config["data_cache_dir"] = os.path.join(
        os.path.expanduser("~"), ".tradingagents", f"cache_{ticker}")

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{ticker}.jsonl"
    done = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                # Only a successful decision is done. The recorded failures are 132
                # transient network errors and 48 cache-lock collisions, all worth a
                # second attempt; the converter takes the later ok line for a date.
                if rec.get("ok"):
                    done.add(rec["date"])
            except (json.JSONDecodeError, KeyError):
                # A power cut mid-append leaves a truncated last line. Re-running that
                # one date is correct; refusing to start because of it is not.
                continue

    graph = TradingAgentsGraph(debug=False, config=config)
    todo = [d for d in dates if d not in done][:limit] if limit else [d for d in dates if d not in done]
    print(f"{ticker}: {len(done)} already done, {len(todo)} to run", flush=True)

    for i, d in enumerate(todo, 1):
        t0 = time.time()
        try:
            _, decision = graph.propagate(ticker, d)
            rec = {"date": d, "decision": str(decision)[:400], "ok": True}
        except Exception as exc:
            rec = {"date": d, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "ok": False}
        rec["seconds"] = round(time.time() - t0, 1)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {ticker} {d} {rec.get('decision', rec.get('error',''))[:40]} "
              f"({rec['seconds']}s, {i}/{len(todo)})", flush=True)
    return 0


def launch_all(limit: int | None) -> int:
    """One process per ticker, so eight run concurrently instead of in series."""
    procs = []
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        cmd = [str(VENV_PY), str(Path(__file__).resolve()), "--ticker", t]
        if limit:
            cmd += ["--limit", str(limit)]
        fh = (logs / f"{t}.log").open("w", encoding="utf-8")
        procs.append((t, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)))
        print(f"launched {t} (pid {procs[-1][1].pid})", flush=True)
    print(f"\n{len(procs)} processes running; logs in {logs}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--launch-all", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of decisions per ticker")
    args = ap.parse_args(argv)

    dates = trading_dates()
    if args.launch_all:
        return launch_all(args.limit)
    if not args.ticker:
        raise SystemExit("pass --ticker or --launch-all")
    return run_ticker(args.ticker, dates, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
