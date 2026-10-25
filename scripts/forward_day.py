"""One trading day of the forward run: the committee read, then the order previews.

The run is ten trading days, 3 to 14 August 2026, and it is a demonstration that the
frozen system executes on live data, not evidence for the sealed result.

Two paths run in order. ``live_analysis.py`` runs all three experts and the frozen
memory over live bars and headlines and writes the committee read; it builds no order.
``paper_forward_run.py`` replays that read through ``--committee-read`` and submits the
position change the account needs, so the order follows the decision the read reports.

    PYTHONPATH=src python scripts/forward_day.py            # both paths, no orders
    PYTHONPATH=src python scripts/forward_day.py --read-only # committee read only

Each day is written under reports/paper_forward/<date>/ so the two weeks accumulate
without overwriting. IB Gateway must already be logged in; that step is manual.
Orders are never submitted from here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_forward"


def run(argv: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        code = subprocess.call(argv, cwd=str(ROOT), stdout=fh,
                               stderr=subprocess.STDOUT)
    print(log.read_text(encoding="utf-8"))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="Label for today's directory (default: today).")
    ap.add_argument("--read-only", action="store_true",
                    help="Committee read only; skip the order previews.")
    args = ap.parse_args(argv)

    day = OUT / args.date
    day.mkdir(parents=True, exist_ok=True)
    print(f"forward run, day {args.date} -> {day}\n")

    print("=" * 74)
    print("  1/2  COMMITTEE READ (all three experts, frozen memory, no order path)")
    print("=" * 74)
    code = run([sys.executable, "scripts/live_analysis.py",
                "--save-bars", str(day / "bars")], day / "committee_read.log")
    if code != 0:
        print(f"\ncommittee read failed (exit {code}); the gateway may not be logged in")
        return code

    if args.read_only:
        print("\n--read-only: order previews skipped")
        return 0

    print("\n" + "=" * 74)
    print("  2/2  ORDER PREVIEWS (twenty names, nothing submitted)")
    print("=" * 74)
    code = run([sys.executable, "scripts/paper_forward_run.py", "--preview"],
               day / "order_previews.log")

    # A one-line record per day, so the write-up does not have to re-read the logs.
    summary = {"date": args.date, "committee_read": "ok",
               "order_previews": "ok" if code == 0 else f"exit {code}"}
    for path in sorted(OUT.glob("*__*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary.setdefault("runs", {})[path.stem] = {
            "connected": doc.get("gateway_connected"),
            "previews": doc.get("order_preview_count"),
            "submitted": doc.get("orders_submitted"),
            "market_state": doc.get("market_state"),
            "market_state_source": doc.get("market_state_source"),
        }
    (day / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwritten to {day}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
