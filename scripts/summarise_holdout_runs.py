"""Cross-name summary over expert-weight holdout reports.

Reads one or more holdout JSON reports (the launcher's --output files) and
prints per-name arm metrics plus the cross-name average per arm: total return,
Sharpe, deflated Sharpe, max drawdown, turnover, participation, and the
market_state_memory-vs-equal_weight deltas. Also prints per-state coverage so
underpowered states are visible rather than hidden.

Usage:
    PYTHONPATH=src python scripts/summarise_holdout_runs.py \
        reports/holdout/TSLA.json reports/holdout/NFLX.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

ARM_ORDER = ("equal_weight", "global_memory", "market_state_memory")
METRICS = (
    "total_return", "sharpe", "deflated_sharpe_ratio", "certainty_equivalent",
    "max_drawdown", "turnover", "participation_rate",
)


def _fmt(value) -> str:
    if value is None:
        return "   n/a"
    return f"{float(value):+.4f}"


def _name(path: Path, payload: dict) -> str:
    tickers = payload.get("manifest", {}).get("tickers") or []
    return "+".join(tickers) if tickers else path.stem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)

    runs = []
    for path in args.reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append((_name(path, payload), payload))

    header = f"{'name':10s} {'arm':22s} " + " ".join(f"{m:>18s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    per_arm_values: dict[str, dict[str, list[float]]] = {
        arm: {m: [] for m in METRICS} for arm in ARM_ORDER
    }
    for name, payload in runs:
        for arm in ARM_ORDER:
            metrics = payload["arms"][arm]["metrics"]
            row = " ".join(f"{_fmt(metrics.get(m)):>18s}" for m in METRICS)
            print(f"{name:10s} {arm:22s} {row}")
            for m in METRICS:
                value = metrics.get(m)
                if value is not None:
                    per_arm_values[arm][m].append(float(value))
        print()

    print("CROSS-NAME AVERAGE")
    for arm in ARM_ORDER:
        row = " ".join(
            f"{_fmt(fmean(vals) if (vals := per_arm_values[arm][m]) else None):>18s}"
            for m in METRICS
        )
        print(f"{'avg':10s} {arm:22s} {row}")

    print("\nSTATE COVERAGE (test window, market_state_memory arm)")
    for name, payload in runs:
        coverage = payload.get("market_state_coverage", {})
        arm_cov = coverage.get("arms", {}).get("market_state_memory", coverage)
        print(f"  {name:10s} {arm_cov}")

    print("\nSELECTED CONFIGURATION / AUDIT")
    for name, payload in runs:
        split = payload.get("split_protocol", {})
        manifest = payload.get("manifest", {})
        print(
            f"  {name:10s} tau={split.get('selected_temperature')} "
            f"thr={split.get('selected_decision_threshold')} "
            f"trials={payload.get('trial_count')} "
            f"freeze_ok={payload.get('freeze_ok')} "
            f"reflection={manifest.get('reflection_backend')} "
            f"model={manifest.get('expert_model_revision')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
