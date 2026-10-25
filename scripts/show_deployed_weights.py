"""Print the weights each frozen snapshot deploys, and compare them to a live run.

Two uses. Before the forward run it shows, in three seconds, what the sealed memory
actually does: eight of the twenty stocks depart from equal weights, ADSK only in
down_stress, and the twelve that verified nothing cannot depart at all. These are the
numbers fixed in the dissertation's forward-demonstration table.

After the run, `--compare reports/paper_forward/<date>` reads each live report and
reports whether the weights that were deployed match the ones above, which is what
fills that table's Observed column. A state-scope stock is compared against the state
the live report was actually in, not the state it verified: ADSK verified in
down_stress, and a run that never entered down_stress is expected to show equal
weights there.

    PYTHONPATH=src python scripts/show_deployed_weights.py
    PYTHONPATH=src python scripts/show_deployed_weights.py --compare reports/paper_forward
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_agent.memory.expert_scorecard import ExpertScorecardStore

ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "frozen_memory"
TAU = 0.5
ORDER = ("fundamental", "news", "technical")
# (symbol, arm, the state whose weights the run should show). All twenty stocks of the
# draw: the twelve that verified nothing are here because equal weights holding live
# is the fallback being checked, not filler.
PLAN = [
    ("BIIB", "global", "global"),
    ("ECL", "global", "global"),
    ("AZO", "global", "global"),
    ("CTAS", "global", "global"),
    ("DDS", "global", "global"),
    ("INCY", "global", "global"),
    ("AES", "global", "global"),
    ("ADSK", "state", "down_stress"),
    ("C", "global", "global"),
    ("CVX", "global", "global"),
    ("DHI", "global", "global"),
    ("PNC", "global", "global"),
    ("TGT", "global", "global"),
    ("AFL", "global", "global"),
    ("AVY", "global", "global"),
    ("BBY", "global", "global"),
    ("MTCH", "global", "global"),
    ("O", "global", "global"),
    ("RL", "global", "global"),
    ("TEX", "global", "global"),
]
STATES = ("up_calm", "up_stress", "down_calm", "down_stress")


def weights(symbol: str, arm: str, state: str) -> tuple[dict[str, float], int]:
    payload = json.loads((MEM / f"{symbol}__{arm}.json").read_text(encoding="utf-8"))
    store = ExpertScorecardStore.from_snapshot(payload, temperature=TAU)
    return store.get_weights(state), len(payload.get("certified_cells", []))


def fmt(w: dict[str, float]) -> str:
    return "  ".join(f"{e[:4]}={w.get(e, 0.0):.3f}" for e in ORDER)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=Path, default=None,
                    help="Directory of completed paper-run reports to check against.")
    args = ap.parse_args(argv)

    print(f"Frozen snapshots in {MEM.relative_to(ROOT)}, retrieval at tau={TAU}")
    print()
    for symbol, arm, state in PLAN:
        w, cells = weights(symbol, arm, state)
        equal = all(abs(v - 1.0 / 3.0) < 5e-4 for v in w.values())
        note = "equal weights" if equal else "departs from equal"
        print(f"  {symbol:5s} {arm:6s} {state:12s} {fmt(w)}   {cells} verified, {note}")
        if arm == "state":
            for s in STATES:
                if s == state:
                    continue
                print(f"  {'':5s} {'':6s} {s:12s} {fmt(weights(symbol, arm, s)[0])}")
    print()

    if args.compare is None:
        print("No live run compared. Add --compare <dir> once the forward run has produced")
        print("reports, to check the deployed weights against these.")
        return 0

    print(f"Comparing against {args.compare}")
    missing = 0
    for symbol, arm, state in PLAN:
        path = args.compare / f"{symbol}__{arm}.json"
        if not path.exists():
            print(f"  {symbol:5s} NO REPORT at {path}")
            missing += 1
            continue
        live = json.loads(path.read_text(encoding="utf-8"))
        # The forward run writes the deployed weights under memory_applied; the
        # older manifest field is kept for reports from before that change.
        got = (live.get("memory_applied", {}).get("weights")
               or live.get("aggregation", {}).get("weights")
               or live.get("manifest", {}).get("expert_weights")
               or live.get("expert_weights") or {})
        if not got:
            print(f"  {symbol:5s} report carries no deployed weights; keys={sorted(live)[:8]}")
            missing += 1
            continue
        # A state-scope stock deploys the cell for the state the run was in. It
        # verified one state; comparing every session against that state would
        # report a mismatch for the sessions the market spent elsewhere, which
        # is the fallback working rather than a disagreement.
        compare_state = state
        if arm == "state":
            compare_state = live.get("market_state") or state
        want, _ = weights(symbol, arm, compare_state)
        worst = max(abs(float(got.get(e, 0.0)) - want.get(e, 0.0)) for e in ORDER)
        verdict = "MATCH" if worst < 1e-3 else f"DIFFERS by {worst:.4f}"
        print(f"  {symbol:5s} live {fmt({k: float(v) for k, v in got.items()})}   {verdict}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
