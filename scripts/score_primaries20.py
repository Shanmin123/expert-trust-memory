"""Re-score the two Section 6.6 primaries on the unified populations.

Certified sentences: paired over the eight stocks where verification passed anything
(on the other twelve there is no sentence to add and the treated arm would be the
control). Producer findings: paired over the whole draw. Both through the existing
`llm_decision_paired_analysis.py`, which enforces the cache/date/skills guards.

The prior three-stock artifacts are preserved under reports/final/superseded_3name/.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/score_primaries20.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"

DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")
VERIFIED = ("ADSK", "AES", "AZO", "BIIB", "CTAS", "DDS", "ECL", "INCY")


N_BOOT = 40_000   # the sealed headline's resample count; 5,000 carries ~±.01 noise


def loso(treated_suffix: str, names, out_key: str, out: Path) -> None:
    """Leave-one-stock-out on the pooled paired difference, same discipline as the
    main result: a pooled effect that rests on one stock is reported as that."""
    from trading_agent.experiments.llm_decision_layer import net_returns, paired_mbb

    diffs = {}
    for t in names:
        treated = json.loads((FINAL / f"{t}__llm_decision_{treated_suffix}.json")
                             .read_text(encoding="utf-8"))
        control = json.loads((FINAL / f"{t}__llm_decision_none.json")
                             .read_text(encoding="utf-8"))
        tr = {r[0]: r for r in treated["daily_arm_series"][treated["arm"]]}
        co = {r[0]: r for r in control["daily_arm_series"][control["arm"]]}
        assert sorted(tr) == sorted(co), t
        nt = net_returns([tr[d] for d in sorted(tr)], treated["manifest"]["cost_bps_per_side"])
        nc = net_returns([co[d] for d in sorted(co)], control["manifest"]["cost_bps_per_side"])
        diffs[t] = [a - b for a, b in zip(nt, nc)]
    rows = {}
    for drop in [None] + list(names):
        kept = [t for t in names if t != drop]
        pooled = [sum(diffs[t][i] for t in kept) / len(kept)
                  for i in range(len(next(iter(diffs.values()))))]
        stat = paired_mbb(pooled, block_len=21, n_boot=5000)
        rows["all" if drop is None else f"without_{drop}"] = {
            "mean_daily_diff": sum(pooled) / len(pooled),
            "p_one_sided": stat["p_one_sided"] if isinstance(stat, dict) else stat[1],
        }
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc[out_key] = rows

    # Cost sensitivity is free: the decisions are fixed, the charge is applied at
    # scoring, so 25 bps is a re-read of the same series.
    from trading_agent.experiments.llm_decision_layer import net_returns as _nr
    diffs25 = {}
    for t in names:
        treated = json.loads((FINAL / f"{t}__llm_decision_{treated_suffix}.json")
                             .read_text(encoding="utf-8"))
        control = json.loads((FINAL / f"{t}__llm_decision_none.json")
                             .read_text(encoding="utf-8"))
        tr = sorted(treated["daily_arm_series"][treated["arm"]])
        co = sorted(control["daily_arm_series"][control["arm"]])
        diffs25[t] = [a - b for a, b in zip(_nr(tr, 25.0), _nr(co, 25.0))]
    pooled25 = [sum(diffs25[t][i] for t in names) / len(names)
                for i in range(len(next(iter(diffs25.values()))))]
    stat25 = paired_mbb(pooled25, block_len=21, n_boot=5000)
    doc["cost_sensitivity_25bps"] = {
        "mean_daily_diff": sum(pooled25) / len(pooled25),
        "p_one_sided": stat25["p_one_sided"],
    }
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    worst = max((v["p_one_sided"], k) for k, v in rows.items() if k != "all")
    neg = [k for k, v in rows.items() if k != "all" and v["mean_daily_diff"] <= 0]
    print(f"   leave-one-out: worst removal {worst[1]} p={worst[0]:.3f}; "
          f"mean stays positive on {len(names) - len(neg)}/{len(names)} removals"
          + (f"; NEGATIVE after {neg}" if neg else ""))
    print(f"   at 25 bps: mean {doc['cost_sensitivity_25bps']['mean_daily_diff']:+.6f}, "
          f"p={doc['cost_sensitivity_25bps']['p_one_sided']:.3f}")


def run(treated_suffix: str, names, out: Path) -> None:
    cmd = [sys.executable, "scripts/llm_decision_paired_analysis.py",
           "--n-boot", str(N_BOOT), "--out", str(out)]
    missing = []
    for t in names:
        treated = FINAL / f"{t}__llm_decision_{treated_suffix}.json"
        control = FINAL / f"{t}__llm_decision_none.json"
        if not treated.exists() or not control.exists():
            missing.append(t)
            continue
        cmd += ["--pair", t, str(treated), str(control)]
    if missing:
        raise SystemExit(f"{out.name}: arms missing for {missing}; run the arms first")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "src"})
    if r.returncode != 0:
        raise SystemExit(f"{out.name} failed:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    d = json.loads(out.read_text(encoding="utf-8"))
    p = d["pooled_equal_capital"]["primary_block"]
    print(f"{out.name}: {len(d['per_name'])} pairs, pooled mean "
          f"{p['mean_daily_diff']:+.6f}, one-sided p={p['p_one_sided']:.4f}")
    for name, v in sorted(d["per_name"].items()):
        b = v["primary_block"]
        print(f"   {name:<5} {b['mean_daily_diff']:+.6f}  p={b['p_one_sided']:.3f}  "
              f"skills={v['n_skills']}")


def main() -> int:
    run("certified", VERIFIED, FINAL / "llm_decision_primary_certified.json")
    loso("certified", VERIFIED, "leave_one_out",
         FINAL / "llm_decision_primary_certified.json")
    print()
    run("producer", DRAW, FINAL / "llm_decision_primary_producer.json")
    loso("producer", DRAW, "leave_one_out",
         FINAL / "llm_decision_primary_producer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
