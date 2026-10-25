"""Run one LLM-decision-layer comparison arm for one name (see experiments/llm_decision_layer.py).

The LLM decision layer replaces the numeric weight layer with a FinMem/TradingAgents-form trader
that reads the three experts' stated views (frozen corpus, via the equal arm's
--export-daily-views file with its binding header) and optionally the certified skills
DERIVED from the canonical report's verification receipts (never free text), then issues
buy/sell/hold per test date. Stateless; test-window evaluation of a fixed policy.

Usage:
    PYTHONPATH=src python scripts/run_llm_decision_layer.py \
        --views reports/views_daily/ECL.jsonl \
        --canonical-report reports/final/ECL__sealed.json \
        --skills-mode certified \
        --cache reports/corpus/llm_decision_ECL_certified.jsonl \
        --out reports/final/ECL__llm_decision_certified.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_agent.experiments.llm_decision_layer import (
    PromptCache,
    check_export_binding,
    derive_certified_skills,
    load_daily_views,
    load_producer_skills,
    metrics_from_series,
    parity_check,
    run_llm_decision_arm,
    skills_digest,
)

DEFAULT_DATA_ROOT = "data/finsaber/sp500_2000_2025_parquet"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def price_series(data_root: str, ticker: str, year_min: int) -> tuple[list[str], dict[str, float]]:
    """(sorted price dates, next-bar adjusted-close return keyed by decision date)."""
    import pandas as pd

    root = Path(data_root) / "price_daily"
    frames = []
    for part in sorted(root.glob("year=*")):
        year = int(part.name.split("=", 1)[1])
        if year < year_min:
            continue
        df = pd.read_parquet(part, columns=["date", "symbol", "adjusted_close"])
        frames.append(df[df["symbol"] == ticker])
    px = pd.concat(frames).sort_values("date").reset_index(drop=True)
    if px.empty:
        raise SystemExit(f"no prices for {ticker} under {root}")
    dates = px["date"].astype(str).tolist()
    if len(set(dates)) != len(dates):
        raise SystemExit(f"duplicate price dates for {ticker}")
    closes = px["adjusted_close"].tolist()
    nxt: dict[str, float] = {}
    for i in range(len(dates) - 1):
        nxt[dates[i]] = closes[i + 1] / closes[i] - 1.0
    return dates, nxt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="JSONL from --export-daily-views (with header)")
    ap.add_argument("--canonical-report", required=True,
                    help="canonical holdout report JSON (binding + parity + skills source)")
    ap.add_argument("--skills-mode", choices=("none", "certified"), default="none",
                    help="certified = derive skills from the canonical report's verified "
                         "cells (state-scoped); none = no-skill baseline arm")
    ap.add_argument("--skills-file", default=None,
                    help="OPTIONAL alternative producer file (JSON object with non-empty "
                         "'producer' provenance and 'skills' [[scope, text], ...]; bare "
                         "lists refused), e.g. LLM-produced skills; mutually exclusive "
                         "with --skills-mode certified; provenance goes into the output")
    ap.add_argument("--cache", required=True, help="LLM decision layer prompt-cache JSONL")
    ap.add_argument("--arm", default=None, help="arm label; default derived from skills mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cost-bps-per-side", type=float, default=10.0)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--no-api", action="store_true", help="replay only; cache miss raises")
    args = ap.parse_args(argv)

    if args.skills_file and args.skills_mode == "certified":
        raise SystemExit("--skills-file and --skills-mode certified are mutually exclusive")

    # Secrets from the gitignored .env, same pattern as the holdout launcher.
    try:
        from data_collection.config import load_env_file
        load_env_file()
    except ImportError:
        pass

    header, days_all = load_daily_views(args.views)
    ticker = days_all[0]["ticker"]
    report = json.loads(Path(args.canonical_report).read_text(encoding="utf-8"))
    check_export_binding(header, report["manifest"])

    canonical_series = report["daily_arm_series"]
    equal_series = canonical_series["equal_weight"]
    canonical_dates = [row[0] for row in equal_series]

    # Evaluate on exactly the canonical equal-series dates (the attributed subset every
    # canonical arm reports) so every series in the table shares one date basis; the
    # parity check separately enforces that these dates are a PREFIX of the export.
    canon_set = set(canonical_dates)
    days = [d for d in days_all if d["date"] in canon_set]
    missing = canon_set - {d["date"] for d in days}
    if missing:
        raise SystemExit(f"views export missing {len(missing)} canonical dates, e.g. {sorted(missing)[:3]}")

    year_min = int(min(canonical_dates)[:4]) - 1
    price_dates, next_returns = price_series(args.data_root, ticker, year_min)
    export_dates = [d["date"] for d in days_all]
    window_price_dates = [d for d in price_dates if export_dates[0] <= d <= export_dates[-1]]

    parity = parity_check(days_all, canonical_series, next_returns,
                          price_dates=window_price_dates)
    print(f"parity check PASSED: {parity}")

    if args.skills_mode == "certified":
        skills = derive_certified_skills(report)
        skills_source = f"derived:{args.canonical_report}"
        skills_mode_effective = "certified"
        arm = args.arm or "llm_decision_certified_skills"
        if not skills:
            print("NOTE: canonical report has zero verified cells -> certified skills "
                  "are empty; this arm is byte-identical to the no-skill arm by design.")
    elif args.skills_file:
        producer, skills = load_producer_skills(args.skills_file)
        skills_source = {"file": args.skills_file, "producer": producer}
        skills_mode_effective = "producer_file"
        arm = args.arm or "llm_decision_producer_skills"
    else:
        skills = ()
        skills_source = "none"
        skills_mode_effective = "none"
        arm = args.arm or "llm_decision_no_skills"

    cache = PromptCache(args.cache, args.model)
    if cache.n_stale_rows:
        print(f"NOTE: ignored {cache.n_stale_rows} stale cache rows (different policy)")
    if args.no_api:
        call_fn = None
    else:
        from trading_agent.experts.backends import OpenAIChatBackend
        backend = OpenAIChatBackend(model=args.model, temperature=0.0)
        call_fn = backend.call

    run = run_llm_decision_arm(
        days, skills, cache, call_fn, next_returns,
        arm=arm, cost_bps_per_side=args.cost_bps_per_side,
    )

    comparison = {
        name: metrics_from_series(rows, args.cost_bps_per_side)
        for name, rows in canonical_series.items()
    }
    payload = {
        "name": ticker,
        "arm": run.arm,
        "manifest": {
            "model": args.model,
            "views": args.views,
            "views_header": dict(header),
            "canonical_report": args.canonical_report,
            "skills_mode": skills_mode_effective,
            "skills_source": skills_source,
            "n_skills": len(skills),
            "skills": list(skills),
            "skills_sha": skills_digest(skills),
            "cache": args.cache,
            "cost_bps_per_side": args.cost_bps_per_side,
            "parity_check": parity,
            "position_blind": True,
            "basis_note": "all metrics computed on the canonical attributed-day basis "
                          f"({len(canonical_dates)} days); NOT comparable to the "
                          "canonical reports' full-window headline metrics",
            "reference_contracts": "TradingAgents trader role; FinMem decision prompt (FINSABER bundle)",
        },
        "metrics": run.metrics,
        "canonical_arms_same_basis": comparison,
        "n_api_calls": run.n_api_calls,
        "n_cache_hits": run.n_cache_hits,
        "n_parse_errors": run.n_parse_errors,
        "n_stale_cache_rows": run.n_stale_cache_rows,
        "decisions": run.decisions,
        # daily_arm_series so the existing paired-MBB tool (portfolio_arm_metrics.py)
        # can consume LLM decision layer reports directly for the pre-registered comparisons.
        "daily_arm_series": {run.arm: run.series, "equal_weight": list(equal_series)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    m = run.metrics
    eq = comparison["equal_weight"]
    print(
        f"{ticker} {run.arm}: sharpe={m['sharpe']:+.3f} ret={m['total_return']:+.3%} "
        f"switches/yr={m['position_changes_per_year']:.1f} "
        f"(equal same-basis {eq['sharpe']:+.3f}/{eq['total_return']:+.3%}) "
        f"api_calls={run.n_api_calls} cache_hits={run.n_cache_hits} parse_errors={run.n_parse_errors}"
    )
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
