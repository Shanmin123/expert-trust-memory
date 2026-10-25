"""Print (and optionally run) the IB paper forward-demonstration commands.

The forward run is a DEMONSTRATION that the frozen system executes on live data, not
evidence for the sealed result: a run of this length cannot be significant and is reported
as a working-system check.

All twenty drawn names run, each on the memory it deploys in the dissertation's
forward-run table, so the weights fixed in advance can be checked stock by stock. The run
is free: this path never calls a model. This path takes its views from the recorded
committee read: --committee-read replays the three experts that scripts/live_analysis.py
ran over live bars and headlines, so the order follows the decision the read reports.
Every manifest names the read it followed under decision_source.
TSLA is deliberately absent: it was held out of the draw, so including it would
misdescribe a random universe.

Each name needs its own process, because a memory snapshot carries no ticker -- it is a
trust profile trained on one name's own history, and this project's finding is that expert
trust is name-specific. Each process also needs its own IB client id.

Market-state fidelity: the paper defaults (vol window 5, percentile window 20) do NOT match
the frozen system (21 and 504). This script always passes the frozen values, plus a lookback
long enough to fill them.

    PYTHONPATH=src python scripts/paper_forward_run.py               # print the commands
    PYTHONPATH=src python scripts/paper_forward_run.py --preview     # run the no-order previews
    PYTHONPATH=src python scripts/paper_forward_run.py --preview --execute   # and submit them

Submission needs both flags, so the same pass that sends an order has already printed it.
Each run writes its manifests under reports/paper_forward/<decision date>/, one directory
per session, because a flat path meant every day overwrote the one before.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "frozen_memory"
OUT = ROOT / "reports" / "paper_forward"
READS = OUT / "reads"


def session() -> str:
    """The trade date the manifests belong to.

    Manifests used to be written to one flat path per symbol, so each day overwrote the
    day before and a ten-day run would end holding one day's record. The day that
    actually opened this account's positions was lost that way. Filing them under the
    committee read's trade date, which is the stem live_analysis.py writes, keeps the
    manifest with the session whose open it fills.
    """
    reads = sorted(READS.glob("*.json"))
    if not reads:
        raise SystemExit("no committee read yet; run scripts/live_analysis.py "
                         "--save-analysis first")
    return reads[-1].stem

# (symbol, arm, client id, what this name shows)
PLAN = [
    ("BIIB", "global", 11, "learned weights depart from equal (3 verified records)"),
    ("CTAS", "global", 12, "largest departure in the draw: fundamental at 0.635"),
    ("ADSK", "state", 13, "state-conditioned memory, live only in a down-stress state"),
    ("DHI", "global", 14, "fallback: nothing verified, must match equal weights exactly"),
    ("ECL", "global", 15, "2 verified records"),
    ("AZO", "global", 16, "2 verified records"),
    ("DDS", "global", 17, "1 verified news record"),
    ("INCY", "global", 18, "1 verified record that never moved a decision"),
    ("AES", "global", 19, "1 verified news record, the one acting stock that lost"),
    ("AFL", "global", 20, "nothing verified, fallback"),
    ("AVY", "global", 21, "nothing verified, fallback"),
    ("BBY", "global", 22, "nothing verified, fallback"),
    ("C", "global", 23, "nothing verified, fallback"),
    ("CVX", "global", 24, "nothing verified, fallback"),
    ("MTCH", "global", 25, "nothing verified, fallback"),
    ("O", "global", 26, "nothing verified, fallback"),
    ("PNC", "global", 27, "nothing verified, fallback"),
    ("RL", "global", 28, "nothing verified, fallback"),
    ("TEX", "global", 29, "nothing verified, fallback"),
    ("TGT", "global", 30, "nothing verified, fallback, rarely in the market"),
]
VOL_WINDOW, PERCENTILE_WINDOW, LOOKBACK = 21, 504, 760


def command(symbol: str, arm: str, client_id: int, dry: bool = True,
            execute: bool = False) -> list[str]:
    argv = [
        sys.executable, "-m", "trading_agent", "paper",
        "--symbols", symbol,
        "--client-id", str(client_id),
        "--auto-decide",
        "--require-memory",
        "--memory-snapshot", str(MEM / f"{symbol}__{arm}.json"),
        # The order path must follow the same decision the committee read recorded.
        # Without this it builds views from the static backend, which always answers
        # long (found 2026-08-03, when every preview was a BUY).
        "--committee-read", str(READS),
        "--paper-vol-window", str(VOL_WINDOW),
        "--paper-percentile-window", str(PERCENTILE_WINDOW),
        "--paper-trend-window", "200",
        "--paper-trend-confirm-days", "5",
        "--lookback-days", str(LOOKBACK),
        "--market-proxy-symbol", "SPY",
        "--use-account-cash",
        "--max-order-value", "5000",
        "--output", str(OUT / session() / f"{symbol}__{arm}.json"),
    ]
    if execute:
        # --execute-orders and --dry-run are mutually exclusive by design.
        argv.append("--execute-orders")
    elif dry:
        argv.append("--dry-run")
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="Execute the previews. Connects for market data and account cash "
                         "only, then disconnects; submits nothing.")
    ap.add_argument("--execute", action="store_true",
                    help="Submit the orders to the connected IB PAPER account. Requires "
                         "--preview to have been read first; run without it to see what "
                         "would be sent. Nothing is sent for a stock whose target already "
                         "matches its holding.")
    args = ap.parse_args(argv)

    plan = PLAN
    missing = [s for s, arm, _, _ in plan if not (MEM / f"{s}__{arm}.json").exists()]
    if missing:
        print(f"missing snapshots in frozen_memory/ for {missing}")
        return 1
    (OUT / session()).mkdir(parents=True, exist_ok=True)

    for symbol, arm, cid, why in plan:
        print(f"\n# {symbol} ({arm} memory, client id {cid}) -- {why}")
        print(" ".join(f'"{a}"' if " " in a else a for a in command(symbol, arm, cid)))

    if args.execute and not args.preview:
        print("\n--execute also needs --preview: the same pass prints each order "
              "before it is sent. Re-run with both.")
        return 1
    if not args.preview:
        print("\nIB Gateway must already be logged in (that step is manual).")
        print("Re-run with --preview to execute these, or copy a line and run it yourself.")
        return 0

    print("\n" + "=" * 70)
    failures = []
    for symbol, arm, cid, _why in plan:
        print(f"\n--- preview {symbol} ({arm}) ---")
        code = subprocess.call(command(symbol, arm, cid, execute=args.execute),
                               cwd=str(ROOT))
        if code != 0:
            failures.append(symbol)
            continue
        path = OUT / session() / f"{symbol}__{arm}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            info = doc.get("memory_snapshot") or {}
            print(f"    connected={doc.get('gateway_connected')} "
                  f"previews={doc.get('order_preview_count')} "
                  f"submitted={doc.get('orders_submitted')} "
                  f"memory_cards={info.get('entries_loaded')} "
                  f"frozen_at={info.get('frozen_at')}")
    print("\n" + "=" * 70)
    if failures:
        print(f"failed: {failures}")
    elif args.execute:
        print("all symbols completed; see orders_submitted per symbol above")
    else:
        print("all previews completed; nothing was submitted")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
