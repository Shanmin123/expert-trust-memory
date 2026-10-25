"""Score the model-written findings against the verified cells, across the draw.

For each stock the producer saw the same per-cell evidence the statistical test
consumed, with the verdicts stripped. This compares what it wrote with what the test
verified, under a mechanical rule stated here rather than judged by eye:

  a verified cell (scope, expert, sign) is RECOVERED when some skill for that stock
  names that expert, carries the matching direction word (positive/adds vs
  negative/subtracts), and matches the scope (a global cell needs a skill with no
  state scope; a state cell needs a skill scoped to that state).

Also counted per stock: how many skills it wrote, whether it abstained (an empty list
was explicitly allowed), and how many skills claim statistical significance on stocks
where the corrected test passed nothing in either memory.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/producer_recovery.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio20 import resolve  # noqa: E402

DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")
LESSONS = ROOT / "reports" / "lessons"
OUT = ROOT / "reports" / "final" / "producer_recovery20.json"

POS = re.compile(r"\b(positive|adds?|added|helpful|beneficial)\b", re.I)
NEG = re.compile(r"\b(negative|subtracts?|subtracted|harmful|detracts?)\b", re.I)
SIG = re.compile(r"\bsignifican", re.I)


def verified_cells(report: dict) -> list[tuple[str, str, int]]:
    """(scope, expert, sign) for every verified cell in either memory."""
    cells = []
    for arm in ("global_memory", "market_state_memory"):
        per = (report.get("verify_gate_report") or {}).get(arm, {}).get("per_cell") or {}
        for key, cell in per.items():
            if not cell.get("verified"):
                continue
            scope, expert = key.split("|")
            cells.append((scope, expert, 1 if cell["mean_m"] > 0 else -1))
    return cells


def skill_matches(skill_scope, text: str, cell) -> bool:
    scope, expert, sign = cell
    if expert not in text.lower():
        return False
    if scope == "global":
        if skill_scope is not None:
            return False
    else:
        if skill_scope != scope:
            return False
    has_pos, has_neg = bool(POS.search(text)), bool(NEG.search(text))
    return (sign > 0 and has_pos and not has_neg) or \
           (sign < 0 and has_neg and not has_pos)


def main() -> int:
    sealed, missing = resolve()
    assert not missing, missing
    rows, recovered, total = [], 0, 0
    for t in DRAW:
        report = json.loads(sealed[t].read_text(encoding="utf-8"))
        cells = verified_cells(report)
        prod = json.loads((LESSONS / f"{t}_llm_producer.json").read_text(encoding="utf-8"))
        skills = prod["skills"]
        hits = []
        for cell in cells:
            hit = any(skill_matches(s[0], s[1], cell) for s in skills)
            hits.append({"cell": list(cell), "recovered": hit})
            recovered += hit
            total += 1
        rows.append({
            "stock": t,
            "n_skills": len(skills),
            "abstained": len(skills) == 0,
            "verified_cells": len(cells),
            "cells": hits,
            "claims_significance": sum(1 for s in skills if SIG.search(s[1])),
        })
    nothing = [r for r in rows if r["verified_cells"] == 0]
    payload = {
        "rule": "a cell is recovered when a skill names its expert with the matching "
                "direction word and scope; see this script",
        "stocks": len(rows),
        "verified_cells_total": total,
        "recovered": recovered,
        "abstentions": sum(r["abstained"] for r in rows),
        "stocks_where_nothing_passed": len(nothing),
        "significance_claims_where_nothing_passed":
            sum(r["claims_significance"] for r in nothing),
        "per_stock": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"recovered {recovered} of {total} verified cells across {len(rows)} stocks")
    print(f"abstentions: {payload['abstentions']}")
    print(f"stocks where nothing passed: {len(nothing)}; significance claims there: "
          f"{payload['significance_claims_where_nothing_passed']}")
    for r in rows:
        mark = "" if not r["verified_cells"] else \
            f"  recovered {sum(c['recovered'] for c in r['cells'])}/{r['verified_cells']}"
        print(f"  {r['stock']:<5} skills={r['n_skills']} sig_claims={r['claims_significance']}{mark}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
