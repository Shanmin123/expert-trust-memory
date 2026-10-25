"""PRE-REGISTERED PRIMARY for the LLM decision layer comparison: within-LLM-decision-layer skill effect.

Computes, per name and pooled, the paired daily net-return difference between the
certified-skill LLM decision layer arm and the no-skill LLM decision layer arm (same views, same dates,
same cached temperature-0 draws wherever prompts are identical), with a centered,
plus-one-corrected one-sided moving-block bootstrap (primary block 21; sensitivity 7/42).

Why this is the primary (external review 2026-07-20, finding 9): the LLM decision layer reads
reasoning/state/date while the numeric decision layer reads directions only, so
LLM decision layer-vs-numeric is an architecture comparison with an information-set confound;
the within-LLM-decision-layer pair isolates exactly the skill text. Comparisons against the
numeric arms remain descriptive.

Enforced preconditions (review new findings 1-2):
- both reports are the SAME name, same date basis (identical date vectors);
- both manifests point at the SAME prompt cache file (shared temperature-0 draws:
  identical prompts must replay identical decisions across the two arms);
- treated arm used skills (mode certified or a producer file), control used none;
- both used the same cost model and views export.

Inference statement (fixed): conditional on the frozen model responses and the named
stocks; NOT inference over LLM sampling randomness or a wider stock population.

Usage:
    PYTHONPATH=src python scripts/llm_decision_paired_analysis.py \
        --pair BIIB reports/final/BIIB__llm_decision_certified.json reports/final/BIIB__llm_decision_none.json \
        --pair ECL  reports/final/ECL__llm_decision_certified.json  reports/final/ECL__llm_decision_none.json \
        --out reports/final/llm_decision_primary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_agent.experiments.llm_decision_layer import net_returns, paired_mbb

PRIMARY_BLOCK = 21
SENSITIVITY_BLOCKS = (7, 42)


def load_arm_series(report: dict) -> tuple[str, list[list]]:
    """(arm name, series) for the LLM decision layer arm inside a run_llm_decision_layer report."""
    arm = report["arm"]
    return arm, report["daily_arm_series"][arm]


def check_pair(name: str, treated: dict, control: dict) -> None:
    if treated["name"] != name or control["name"] != name:
        raise AssertionError(f"pair {name}: reports are for {treated['name']}/{control['name']}")
    t_man, c_man = treated["manifest"], control["manifest"]
    if t_man.get("skills_mode") == "none" or t_man.get("n_skills", 0) == 0:
        raise AssertionError(f"pair {name}: treated report has no skills")
    if c_man.get("skills_mode") != "none" or c_man.get("n_skills", 0) != 0:
        raise AssertionError(f"pair {name}: control report is not the no-skill arm")
    if t_man.get("cache") != c_man.get("cache"):
        raise AssertionError(
            f"pair {name}: arms used different prompt caches "
            f"({t_man.get('cache')!r} vs {c_man.get('cache')!r}); identical prompts must "
            "share one temperature-0 draw for the paired difference to isolate the skill"
        )
    for key in ("views", "canonical_report", "cost_bps_per_side", "model"):
        if t_man.get(key) != c_man.get(key):
            raise AssertionError(f"pair {name}: manifest {key} differs across arms")
    _, t_series = load_arm_series(treated)
    _, c_series = load_arm_series(control)
    if [r[0] for r in t_series] != [r[0] for r in c_series]:
        raise AssertionError(f"pair {name}: date vectors differ across arms")


def paired_diffs(treated: dict, control: dict) -> tuple[list[str], list[float]]:
    cost = float(treated["manifest"]["cost_bps_per_side"])
    _, t_series = load_arm_series(treated)
    _, c_series = load_arm_series(control)
    t_net = net_returns(t_series, cost)
    c_net = net_returns(c_series, cost)
    dates = [r[0] for r in t_series]
    return dates, [a - b for a, b in zip(t_net, c_net)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("NAME", "TREATED_REPORT", "CONTROL_REPORT"),
                    help="one name's certified-skill report and no-skill report")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    per_name: dict[str, dict] = {}
    diff_by_date_lists: list[dict[str, float]] = []
    for name, treated_path, control_path in args.pair:
        treated = json.loads(Path(treated_path).read_text(encoding="utf-8"))
        control = json.loads(Path(control_path).read_text(encoding="utf-8"))
        check_pair(name, treated, control)
        dates, diffs = paired_diffs(treated, control)
        blocks = {PRIMARY_BLOCK: paired_mbb(diffs, PRIMARY_BLOCK, args.n_boot)}
        for b in SENSITIVITY_BLOCKS:
            blocks[b] = paired_mbb(diffs, b, args.n_boot)
        per_name[name] = {
            "treated_report": treated_path,
            "control_report": control_path,
            "treated_arm": treated["arm"],
            "n_skills": treated["manifest"]["n_skills"],
            "skills_sha": treated["manifest"].get("skills_sha"),
            "primary_block": blocks[PRIMARY_BLOCK],
            "sensitivity_blocks": {str(b): blocks[b] for b in SENSITIVITY_BLOCKS},
        }
        diff_by_date_lists.append(dict(zip(dates, diffs)))

    pooled = None
    if len(diff_by_date_lists) > 1:
        common = set(diff_by_date_lists[0])
        for d in diff_by_date_lists[1:]:
            common &= set(d)
        for i, d in enumerate(diff_by_date_lists):
            if len(common) != len(d):
                raise AssertionError(
                    f"pooled series: pair #{i} has {len(d)} dates but only {len(common)} "
                    "are common across names; refusing a silently unbalanced pool"
                )
        dates_sorted = sorted(common)
        pooled_diffs = [sum(d[dt] for d in diff_by_date_lists) / len(diff_by_date_lists)
                        for dt in dates_sorted]
        pooled = {PRIMARY_BLOCK: paired_mbb(pooled_diffs, PRIMARY_BLOCK, args.n_boot)}
        for b in SENSITIVITY_BLOCKS:
            pooled[b] = paired_mbb(pooled_diffs, b, args.n_boot)
        pooled = {"primary_block": pooled[PRIMARY_BLOCK],
                  "sensitivity_blocks": {str(b): pooled[b] for b in SENSITIVITY_BLOCKS}}

    payload = {
        "estimand": "PRE-REGISTERED PRIMARY: within-LLM-decision-layer paired daily net-return "
                    "difference (skill arm minus no-skill arm)",
        "inference_scope": "conditional on the frozen model responses and the named "
                           "stocks; not over LLM sampling randomness or a stock population",
        "numeric_arm_note": "comparisons against numeric committee arms are DESCRIPTIVE "
                            "(information-set confound: the LLM decision layer reads reasoning/"
                            "state/date, the numeric layer reads directions only)",
        "per_name": per_name,
        "pooled_equal_capital": pooled,
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"WROTE {args.out}")
    for name, row in per_name.items():
        p = row["primary_block"]
        print(f"{name}: mean_diff={p['mean_daily_diff']:+.2e}/day "
              f"p={p['p_one_sided']:.4f} (block {p['block_len']}, "
              f"nonzero {p['n_nonzero_diffs']}/{p['n_dates']})")
    if pooled:
        p = pooled["primary_block"]
        print(f"POOLED: mean_diff={p['mean_daily_diff']:+.2e}/day p={p['p_one_sided']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
