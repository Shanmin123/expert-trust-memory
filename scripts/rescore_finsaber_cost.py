"""Re-score the sealed decisions under FINSABER's own commission schedule.

The thesis charged ten basis points a side on traded notional. The benchmark it takes
its baselines from charges Interactive Brokers' fixed tier instead: $0.0049 a share, at
least $0.99 an order, at most 1% of the trade. On these position sizes that is about
0.4 basis points a side, so the flat rate was roughly twenty-five times the commission,
and it fell hardest on whichever system traded most.

Cost enters at execution and never at a decision: the committee votes on expert views,
the score is compared with a fixed threshold, and a long position is the whole account.
So this is a re-scoring of one recorded run and not a second experiment, and the script
proves that rather than asserting it -- ``--check`` requires every exposure series and
the whole verification report to come back bit-identical to the sealed one.

    PYTHONPATH=src python scripts/rescore_finsaber_cost.py --width 6
    PYTHONPATH=src python scripts/rescore_finsaber_cost.py --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trading_agent.paths import data_root  # noqa: E402
OUT = ROOT / "reports" / "finsaber_cost"
DATA = data_root()

SEC_EXPAND = DATA / "sec_xbrl" / "financials_expand20.parquet"
SEC_SEALED = DATA / "sec_xbrl" / "financials.parquet"
EXPAND20 = {"AFL", "AVY", "AZO", "BBY", "CTAS", "DDS", "INCY", "MTCH", "O", "RL", "TEX"}
#: The macro table feeds the news prompt, and only the runs outside expand20 had it.
#: Omitting it does not fail: the seat's replay lookups all miss and it drops out.
MACRO = DATA / "macro" / "fred_macro.parquet"

#: A zero headline rate is what switches the execution adapter from the flat model to
#: the per-share one, whose defaults are already FINSABER's. The old headline becomes
#: the sensitivity, so the conservative rate is still reported.
HEADLINE_BPS = "0"
SENSITIVITY_BPS = "10"


def argv_for(ticker: str) -> list[str]:
    from portfolio20 import resolve
    sealed, missing = resolve()
    if ticker not in sealed:
        raise SystemExit(f"no sealed report for {ticker} (missing: {missing})")
    m = json.loads(sealed[ticker].read_text(encoding="utf-8"))["manifest"]
    corpus = ROOT / m["view_corpus"]["path"]
    if not corpus.exists():
        raise SystemExit(f"{ticker}: frozen corpus absent at {corpus}")
    stem = OUT / ticker
    argv = [
        sys.executable, str(ROOT / "scripts" / "run_expert_weight_holdout.py"),
        "--tickers", ticker,
        "--data-root", str(DATA / "finsaber" / "sp500_2000_2025_parquet"),
        "--market-proxy", str(DATA / "market_proxy" / "spy_daily.parquet"),
        "--sec-facts-path", str(SEC_EXPAND if ticker in EXPAND20 else SEC_SEALED),
        *(() if ticker in EXPAND20 else ("--fred-macro-path", str(MACRO))),
        "--start", m["start"],
        "--validation-boundary", m["validation_boundary"],
        "--boundary", m["boundary"],
        "--end", m["end"],
        "--embargo-days", str(m["embargo_days"]),
        "--exit-threshold", str(m["exit_threshold"]),
        "--expert-backend", m["expert_backend"],
        "--expert-model", m["expert_model"],
        "--expert-model-revision", m["expert_model_revision"],
        "--reflection-backend", m["reflection_backend"],
        "--attribution-mode", m["attribution_mode"],
        "--attribution-horizon", str(m["attribution_horizon"]),
        "--reflect-every", str(m["reflect_every"]),
        "--reflection-window", str(m["reflection_window"]),
        "--temperature-grid", ",".join(str(x) for x in m["temperature_grid"]),
        "--view-corpus", str(corpus),
        "--view-corpus-mode", "replay_by_date" if m.get("news_sentiment_cache") else "replay",
        "--headline-cost-bps-per-side", HEADLINE_BPS,
        "--sensitivity-cost-bps-per-side", SENSITIVITY_BPS,
        "--run-label", f"{ticker}_FINSABER_COST",
        "--output", str(stem) + ".json",
        "--memory-output", str(stem) + "_memory.json",
    ]
    for flag, key in (("--parallel-experts", "parallel_expert_calls"),
                      ("--terminal-verify", "terminal_verify"),
                      ("--certified-only-memory", "certified_only_memory")):
        if m.get(key):
            argv.append(flag)
    if m.get("news_sentiment_cache"):
        argv += ["--news-sentiment-cache", str(ROOT / m["news_sentiment_cache"]),
                 "--news-sentiment-model", m["news_sentiment_model"]]
    return argv


def run_all(names: list[str], width: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for t in names if not (OUT / f"{t}.json").exists()]
    print(f"{len(todo)} to re-score, {width} at a time")
    live, started, failed = [], time.time(), 0
    while todo or live:
        while todo and len(live) < width:
            t = todo.pop(0)
            log = (OUT / f"{t}.log").open("w", encoding="utf-8")
            live.append((t, subprocess.Popen(argv_for(t), cwd=str(ROOT), stdout=log,
                                             stderr=subprocess.STDOUT), log))
            print(f"  start {t}")
        time.sleep(5)
        for item in list(live):
            t, p, log = item
            if p.poll() is None:
                continue
            live.remove(item)
            log.close()
            failed += 0 if p.returncode == 0 else 1
            print(f"  {'done ' if not p.returncode else 'FAIL '} {t} "
                  f"({(time.time() - started) / 60:.0f} min)")
    print(f"finished, {failed} failures")
    return 1 if failed else 0


def check(names: list[str]) -> int:
    """The re-scored run must differ from the sealed one only in what cost it charged."""
    from portfolio20 import resolve
    sealed, _ = resolve()
    bad = 0
    for t in names:
        p = OUT / f"{t}.json"
        if not p.exists():
            print(f"  {t:5s} NOT RE-SCORED")
            bad += 1
            continue
        a = json.loads(sealed[t].read_text(encoding="utf-8"))
        b = json.loads(p.read_text(encoding="utf-8"))
        arms = sorted(set(a["daily_arm_series"]) & set(b["daily_arm_series"]))
        same_dec = all([(r[0], r[2]) for r in a["daily_arm_series"][k]]
                       == [(r[0], r[2]) for r in b["daily_arm_series"][k]] for k in arms)

        def passed_records(d):
            return {f"{k}|{c}": round(s["p_holm"], 9)
                    for k, v in (d.get("verify_gate_report") or {}).items()
                    for c, s in (v.get("per_cell") or {}).items()
                    if isinstance(s, dict) and s.get("p_holm") is not None}
        same_ver = passed_records(a) == passed_records(b)
        ex = (b["manifest"].get("execution") or {})
        right_cost = (ex.get("commission_per_share") == 0.0049
                      and ex.get("min_commission") == 0.99
                      and ex.get("max_commission_rate") == 0.01
                      and not ex.get("execution_cost_bps_per_side"))
        if same_dec and same_ver and right_cost:
            print(f"  {t:5s} decisions and verification unchanged, commission model active")
        else:
            bad += 1
            print(f"  {t:5s} FAILED: decisions {same_dec}, verification {same_ver}, "
                  f"cost {right_cost} ({ex})")
    print(f"{len(names) - bad}/{len(names)} are clean re-scorings")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    from portfolio20 import resolve
    sealed, _ = resolve()
    names = sorted(sealed)
    return check(names) if a.check else run_all(names, a.width)


if __name__ == "__main__":
    raise SystemExit(main())
