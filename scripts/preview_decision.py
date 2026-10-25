"""Show one day's committee decision as an operator would read it. Never trades.

The frozen system decides in four steps: the market state is labelled, the experts
speak or abstain, the verified memory supplies a weight for each expert that spoke, and
the boundary rule turns the weighted vote into long or cash. This prints all four for a
single date, so the recommendation can be read back to its inputs.

    PYTHONPATH=src python scripts/preview_decision.py --symbol BIIB --date 2025-11-14
    PYTHONPATH=src python scripts/preview_decision.py --symbol BIIB --latest

Views are replayed from the recorded corpus, so the run costs nothing and calls no
model. Weights come from the deployable snapshot in frozen_memory, read at the
selected retrieval temperature.

This script cannot place an order. It imports no broker connector and has no execution
path; the only output is text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_agent.aggregation.soft_weighting import aggregate
from trading_agent.contracts import ExpertView
from trading_agent.memory.expert_scorecard import ExpertScorecardStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "reports" / "corpus"
MEM = ROOT / "frozen_memory"

# The temperature selected on the validation window and frozen for every run. The
# snapshot's own ``temperature`` field records the TRAINING store and is not a
# retrieval choice, so it is overridden here on purpose.
TAU = 0.5
ORDER = ("fundamental", "news", "technical")
SPEAKS = {"long": +1, "flat": -1}
W = 66


def rule(ch: str = "-") -> str:
    return ch * W


# Each name's SEALED corpus, matching the run that produced its Chapter 5 row. The
# bare views_<SYM>.jsonl files are earlier four-expert recordings and would show a
# committee the dissertation does not report.
SEALED_CORPUS = {
    "BIIB": "BIIB__designD3_q",
    "ECL": "ECL__sealed",
    "DHI": "DHI__sealed",
    "ADSK": "ADSK__designD3_q",
    "TSLA": "TSLA__designD3_q",
    "CVX": "CVX__final",
    "PNC": "PNC__sealed",
    "TGT": "TGT__sealed",
}


def load_views(symbol: str) -> dict[str, dict[str, dict]]:
    """{date: {expert_id: parsed view}} from the sealed corpus for this name."""
    stem = SEALED_CORPUS.get(symbol, symbol)
    path = CORPUS / f"views_{stem}.jsonl"
    if not path.exists():
        known = ", ".join(sorted(SEALED_CORPUS))
        raise SystemExit(f"no sealed corpus at {path}\nnames with one: {known}")
    out: dict[str, dict[str, dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            body = json.loads(row["response"])
        except (json.JSONDecodeError, KeyError):
            continue
        if not isinstance(body, list) or not body:
            continue
        out.setdefault(row["date"], {})[row["expert_id"]] = body[0]
    return out


def bar(value: float, width: int = 22) -> str:
    return "#" * max(0, round(value * width))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print one day's decision. Never trades.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD from the recorded corpus")
    ap.add_argument("--latest", action="store_true", help="use the most recent recorded date")
    ap.add_argument("--arm", default="global", choices=("global", "state"))
    ap.add_argument("--market-state", default="global",
                    help="state label used to retrieve weights (state arm only)")
    args = ap.parse_args(argv)

    by_date = load_views(args.symbol)
    if args.latest or args.date is None:
        date = max(by_date)
    else:
        date = args.date
        if date not in by_date:
            near = sorted(d for d in by_date if d <= date)
            raise SystemExit(
                f"{args.symbol} has no recorded views on {date}."
                + (f" Nearest earlier: {near[-1]}" if near else "")
            )
    raw = by_date[date]

    snap_path = MEM / f"{args.symbol}__{args.arm}.json"
    if not snap_path.exists():
        raise SystemExit(f"no deployable snapshot at {snap_path}")
    payload = json.loads(snap_path.read_text(encoding="utf-8"))
    store = ExpertScorecardStore.from_snapshot(payload, temperature=TAU)
    state = "global" if args.arm == "global" else args.market_state
    weights = store.get_weights(state)
    verified = payload.get("certified_cells", [])

    print()
    print(rule("="))
    print(f"  {args.symbol}  {date}".ljust(W - 22) + "replay, no order".rjust(20))
    print(rule("="))

    print()
    print("  MARKET STATE")
    print(f"    {state}".ljust(24) + "(SPY trend x risk; the traded stock never enters it)")

    print()
    print("  EXPERT VIEWS")
    views: dict[str, ExpertView] = {}
    for eid in ORDER:
        v = raw.get(eid)
        if v is None:
            print(f"    {eid:<14}--      no view recorded")
            continue
        d = str(v.get("direction", "neutral")).lower()
        conf = float(v.get("confidence", 0.0))
        why = str(v.get("reasoning", "")).strip()
        if len(why) > 58:
            why = why[:57] + "…"
        label = {"long": "LONG", "flat": "FLAT"}.get(d, "ABSTAIN")
        print(f"    {eid:<14}{label:<8}conf {conf:.2f}")
        if why:
            print(f"                  {why}")
        if d in SPEAKS:
            views[eid] = ExpertView(
                expert_id=eid, direction=d, confidence=conf, reasoning=why,
            )

    print()
    n_cells = len(verified)
    print("  TRUST WEIGHTS" + f"   frozen {payload.get('frozen_at')}, "
          f"{n_cells} verified cell{'s' if n_cells != 1 else ''}")
    for eid in ORDER:
        w = weights.get(eid, 0.0)
        spoke = "" if eid in views else "   (silent, weight redistributed)"
        print(f"    {eid:<14}{w:.3f}  {bar(w):<24}{spoke}")
    if n_cells == 0:
        print("    nothing passed the test, so these are equal weights by construction")

    print()
    print(rule())
    if not views:
        print("  AGGREGATE       no expert spoke")
        print("  RECOMMENDATION  MAINTAIN  (silence carries the current position)")
    else:
        res = aggregate(views, weights, threshold=0.0)
        spoke = ", ".join(sorted(views))
        print(f"  AGGREGATE       A = {res.score:+.3f}      over: {spoke}")
        print("                  rule: long if A > 0, cash if A < 0, else maintain")
        verdict = {"long": "LONG", "flat": "CASH"}.get(res.direction, "MAINTAIN")
        print(f"  RECOMMENDATION  {verdict}")
    print(rule())
    print()
    print("  No order was placed. This tool has no execution path.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
