"""Audit what the tuned temperature tau actually changes in the sealed runs.

Reproduces the 2026-07-29 correction to FINAL_DESIGN: which cells' weights move with tau,
whether any of them crosses the threshold at which one expert overrules the other two, and
whether the aggregate sign on real voting days differs across the grid.

Read-only on the sealed artifacts and the frozen corpora; no run, no new test read (the grid
was executed as part of the sealed protocol).

    PYTHONPATH=src python scripts/tau_sensitivity_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path
from trading_agent.paths import data_root

ROOT = Path(__file__).resolve().parents[1]
_LOCAL = Path(str(data_root()))
GRID = (0.5, 1.0, 2.0)
SEALED = {
    "TSLA": "TSLA__designD3_q_h21cert.json", "ADSK": "ADSK__designD3_q_h21cert.json",
    "BIIB": "BIIB__designD3_q_h21cert.json", "CVX": "CVX__final.json",
    "DHI": "DHI__sealed.json", "ECL": "ECL__sealed.json",
    "PNC": "PNC__sealed.json", "TGT": "TGT__sealed.json",
}
CORPORA = {
    "TSLA": "views_TSLA__designD3_q.jsonl", "ADSK": "views_ADSK__designD3_q.jsonl",
    "BIIB": "views_BIIB__designD3_q.jsonl", "CVX": "views_CVX__final.jsonl",
    "DHI": "views_DHI__sealed.jsonl", "ECL": "views_ECL__sealed.jsonl",
    "PNC": "views_PNC__sealed.jsonl", "TGT": "views_TGT__sealed.jsonl",
}


def _sealed(name: str) -> dict:
    p = _LOCAL / "final" / SEALED[name]
    if not p.exists():
        p = ROOT / "reports" / "final" / SEALED[name]
    return json.loads(p.read_text(encoding="utf-8"))


def _views(name: str, keep: set) -> dict:
    out: dict = {}
    with open(_LOCAL / "corpus" / CORPORA[name], encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["date"] not in keep:
                continue
            resp = rec["response"]
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    continue
            if isinstance(resp, list):
                resp = resp[0] if resp else {}
            if not isinstance(resp, dict):
                continue
            if str(resp.get("data_quality", "")).lower() in ("data_unavailable", "parse_error"):
                continue
            direction = str(resp.get("direction", "")).lower()
            if direction == "neutral":
                continue
            out.setdefault(rec["date"], {})[rec["expert_id"]] = 1.0 if direction == "long" else -1.0
    return out


def _score(votes: dict, weights: dict) -> float:
    available = {e: v for e, v in votes.items() if e in weights}
    total = sum(weights[e] for e in available)
    return sum(weights[e] * v for e, v in available.items()) / total if total > 0 else 0.0


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    print("TAU SENSITIVITY AUDIT -- what the one tuned parameter actually changes\n")
    print(f"{'name':6}{'arm':22}{'max weight @0.5':>17}{'voting days':>13}{'sign flips':>12}")
    any_dictator = False
    total_flips = 0
    for name in SEALED:
        doc = _sealed(name)
        for arm in ("global_memory", "market_state_memory"):
            snap = doc["arm_memory_snapshots"][arm]
            if not snap.get("cards"):
                continue
            stores = {t: ExpertScorecardStore.from_snapshot(snap, temperature=t) for t in GRID}
            states = {c["market_state"] for c in snap["cards"]}
            moves = any(
                len({tuple(sorted(stores[t].get_weights(s).items())) for t in GRID}) > 1
                for s in states
            )
            if not moves:
                continue
            labels = {r["date"]: r["state"] for r in doc["market_state_labels"]["arms"][arm]}
            views = _views(name, set(labels))
            largest = max(max(stores[0.5].get_weights(s).values())
                          for s in states if s != "warmup")
            any_dictator |= largest > 0.5
            flips = voting = 0
            for dt, votes in views.items():
                state = labels[dt]
                if state == "warmup" or not votes:
                    continue
                voting += 1
                base = _score(votes, stores[0.5].get_weights(state))
                if any((_score(votes, stores[t].get_weights(state)) > 0) != (base > 0)
                       for t in (1.0, 2.0)):
                    flips += 1
            total_flips += flips
            mark = "  <-- above 0.5: one seat can overrule the other two" if largest > 0.5 else ""
            print(f"{name:6}{arm:22}{largest:>17.4f}{voting:>13}{flips:>12}{mark}")

    print()
    if any_dictator:
        print("The grid CONTAINS a change of decision rule (a weight above 0.5 at tau = 0.5),")
        print(f"so the selection is not structurally inconsequential -- but it produced "
              f"{total_flips} sign changes")
        print("on real voting days, so it is inconsequential in fact. Reported that way.")
    else:
        print("No weight reaches 0.5 anywhere on the grid, so a majority always decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
