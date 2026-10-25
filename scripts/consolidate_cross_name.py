"""Consolidate per-name holdout reports into the dissertation's cross-name tables.

Reads reports/final/<name>.json for the reference set (TSLA/NFLX/AMZN/MSFT) and
the random set, and emits two tables with per-name and cross-name-mean metrics
for the four arms, plus the two headline comparisons the evaluation is built
around:

- self-improving vs not: online-state vs frozen-state vs equal-weight
- learned trust vs equal: frozen-state vs equal-weight (the core claim)

reports/ is gitignored, so the consolidated markdown is written to the tracked
docs/analysis/ tree.

    PYTHONPATH=src python scripts/consolidate_cross_name.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REFERENCE = ("TSLA", "NFLX", "AMZN", "MSFT")
RANDOM = ("ADSK", "BIIB", "C", "CVX", "DHI", "ECL", "PNC", "TGT")
ARMS = (
    ("equal_weight", "equal"),
    ("global_memory", "global"),
    ("market_state_memory", "state(frozen)"),
    ("market_state_memory_online", "state(online)"),
)
METRICS = ("total_return", "sharpe", "deflated_sharpe_ratio", "max_drawdown",
           "turnover", "participation_rate")


def _load(name: str, report_dir: Path) -> dict | None:
    path = report_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_metrics(report: dict, arm_key: str) -> dict | None:
    arm = report.get("arms", {}).get(arm_key)
    return arm["metrics"] if arm else None


def _fmt(v, pct=False):
    if v is None:
        return "   -  "
    return f"{v:+.1%}" if pct else f"{v:+.2f}"


def _table(names: tuple[str, ...], report_dir: Path) -> tuple[list[str], dict]:
    """Render one set's table; return (lines, {arm_label: [sharpes]} for means)."""
    lines = [
        f"| {'name':6s} | {'arm':13s} | {'ret':>7s} | {'Sharpe':>6s} | {'DSR':>5s} "
        f"| {'maxDD':>6s} | {'turn':>6s} | {'part':>6s} |",
        "|--------|---------------|--------:|------:|-----:|------:|------:|------:|",
    ]
    sharpe_by_arm: dict[str, list[float]] = {label: [] for _, label in ARMS}
    ret_by_arm: dict[str, list[float]] = {label: [] for _, label in ARMS}
    for name in names:
        report = _load(name, report_dir)
        if report is None:
            lines.append(f"| {name:6s} | (no report yet) | | | | | | |")
            continue
        for arm_key, label in ARMS:
            m = _arm_metrics(report, arm_key)
            if m is None:
                lines.append(f"| {name:6s} | {label:13s} | {'(absent)':>7s} | | | | | |")
                continue
            if m.get("sharpe") is not None:
                sharpe_by_arm[label].append(m["sharpe"])
            if m.get("total_return") is not None:
                ret_by_arm[label].append(m["total_return"])
            lines.append(
                f"| {name:6s} | {label:13s} | {_fmt(m.get('total_return'), pct=True)} "
                f"| {_fmt(m.get('sharpe'))} | {_fmt(m.get('deflated_sharpe_ratio'))} "
                f"| {_fmt(m.get('max_drawdown'), pct=True)} | {m.get('turnover', 0):6.1f} "
                f"| {_fmt(m.get('participation_rate'), pct=True)} |"
            )
    lines.append("")
    lines.append("Cross-name means (arms present on all reported names):")
    for _, label in ARMS:
        s = sharpe_by_arm[label]
        r = ret_by_arm[label]
        if s:
            lines.append(f"- {label:13s}: mean Sharpe {st.mean(s):+.3f} "
                         f"(n={len(s)}), mean return {st.mean(r):+.1%}")
    return lines, sharpe_by_arm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/final"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/analysis/cross_name_evaluation.md"))
    args = parser.parse_args(argv)

    out = ["# Cross-name evaluation (standardized memory + HOLD + LLM reflection)",
           "",
           "Test window 2024-07..2025-12 (post model cutoff 2024-06). Arms: equal-weight,",
           "global memory, frozen state memory, online (self-improving) state memory.",
           ""]

    out.append("## Reference set (FinMem/TradingAgents lineage)")
    ref_lines, ref_sharpe = _table(REFERENCE, args.report_dir)
    out += ref_lines

    out.append("")
    out.append("## Random set (external validity, seed=42)")
    rnd_lines, rnd_sharpe = _table(RANDOM, args.report_dir)
    out += rnd_lines

    # Headline comparisons over all reported names (both sets pooled).
    out.append("")
    out.append("## Headline comparisons (all reported names pooled)")
    pooled: dict[str, list[float]] = {label: [] for _, label in ARMS}
    for names in (REFERENCE, RANDOM):
        for name in names:
            report = _load(name, args.report_dir)
            if report is None:
                continue
            for arm_key, label in ARMS:
                m = _arm_metrics(report, arm_key)
                if m and m.get("sharpe") is not None:
                    pooled[label].append(m["sharpe"])

    def _mean(label):
        v = pooled[label]
        return f"{st.mean(v):+.3f} (n={len(v)})" if v else "n/a"

    out.append(f"- **learned trust vs equal**: state(frozen) {_mean('state(frozen)')} "
               f"vs equal {_mean('equal')}")
    out.append(f"- **self-improving vs frozen**: state(online) {_mean('state(online)')} "
               f"vs state(frozen) {_mean('state(frozen)')}")
    out.append(f"- global memory {_mean('global')} (pooled control)")
    out.append("")
    out.append("Down-state episodes are underpowered in this single post-cutoff test "
               "window (the test regime is predominantly up-state); per-state "
               "participation is reported per name in the JSON, not averaged here.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nwritten -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
