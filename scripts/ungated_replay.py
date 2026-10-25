"""Replay a name's test window from its stored trained memory.

Four of the twenty names lost the early part of their view corpus to a sync failure,
so they cannot be re-run end to end: training the memory would call the model again and
put a second set of expert judgements into a comparison whose whole point is that both
arms read one set. But they do not need re-training. Three facts make the test window
replayable from what survives:

  the trained memory is stored, per arm, in the sealed report's arm_memory_snapshots;
  the four names' TEST-window views are complete, and the technical seat is a
    deterministic rating recomputed from the price panel rather than a cached call;
  the verified and unverified runs train IDENTICAL memory. Verified field by field on the
    sixteen names that have both: same EW mean, variance, count, hit rate, reflection
    delta and certified set across all three arms. `--certified-only-memory` changes
    only how a weight is RETRIEVED from that memory, not what is learned.

So the unverified arm is a numeric re-read of one trained memory, exactly like the
temperature grid. This script performs that re-read: it loads the stored snapshot,
flips `certified_only`, and calls the project's own `run_expert_loop`. Nothing about
aggregation, the hold rule or attribution is reimplemented here.

The claim is only as good as its check, so the script checks itself two ways:

    python scripts/ungated_replay.py --validate-ungated     # reproduce the 16 known unverified arms
    python scripts/ungated_replay.py --validate-verified       # reproduce the 4 names' sealed arms
    python scripts/ungated_replay.py --run MTCH,O,RL,TEX    # produce the missing unverified arms

Exposure is the comparison column: it is the decision path, and it is invariant to the
next-open repricing that the sealed artifacts carry and the launcher's raw output does not.

Measured fidelity, and one place it is not exact. Eighty arms were checked: sixteen names
by four arms against the launcher's unverified run, and four names by four arms against their
sealed run. Seventy-nine reproduce day for day. The exception is AVY's continually
updating arm under unverified retrieval, which differs on five of 352 days and by 0.145
Sharpe; the same arm reproduces exactly under verified retrieval, and its end-state memory
matches the sealed one field for field, so the divergence is transient and reconverges.
The cause was not found. It is the only configuration where the memory both acts and keeps
changing, which is why nothing else shows it. Three things bound the consequence: the
sixteen names' figures come from real launcher runs and never touch this script; the four
replayed names were checked against their own ground truth; and Chapter 6's counts and
means are unchanged whether those four names' updated arms are included, dropped, or the
four names are dropped altogether.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "reports" / "ungated20"
SEALED = ROOT / "reports" / "reprice"
DAMAGED = ("MTCH", "O", "RL", "TEX")


def _setup(ticker: str):
    """Rebuild exactly what the launcher builds, up to the point of running an arm."""
    from ungated_twenty import args_for
    from trading_agent.experiments import expert_weight_holdout_launcher as L
    from trading_agent.experiments.expert_weight_holdout import _copy_config, _first_test_date
    from trading_agent.expert_context import build_expert_signal_sources
    from trading_agent.alpha.market_state_classifier import MarketStateConfig
    from trading_agent.alpha.technical_rating import TradingViewRating
    from trading_agent.market import LoaderOHLCVPanel
    from trading_agent.experts.backends import build_experts
    from trading_agent.experts.view_corpus import ViewCorpus, attach_corpus
    from trading_agent.data.finsaber_panel import load_panel, resolve_data_root
    from trading_agent.memory.expert_reflection import DeterministicExpertReflection

    args = L.build_parser().parse_args(args_for(ticker))
    args.preload_start = L._preload_start(args.start, args.preload_calendar_days)
    data_root = str(resolve_data_root(args.data_root))

    loader = load_panel(data_root, args.preload_start, args.end,
                        tickers=(ticker,), modalities=("price",))
    cfg = L._config(args, (ticker,))
    signal_sources = build_expert_signal_sources(
        data_root=data_root, start=args.preload_start, end=args.end, tickers=(ticker,),
        alpha158_factors_db=None, qlib_provider_uri=None, news_sentiment_backend=None,
        news_sentiment_cache_path=None, include_news=True, strict_news=args.strict_news,
        include_filings=True, sec_facts_path=args.sec_facts_path,
        technical_rating=TradingViewRating(LoaderOHLCVPanel(loader)),
        fred_macro_path=args.fred_macro_path, prior_lessons_path=None,
    )
    market_state_config = MarketStateConfig(
        vol_window=args.vol_window, percentile_window=args.percentile_window,
        trend_window=args.trend_window, trend_confirm_days=args.trend_confirm_days,
        mode="trend_risk",
    )
    experts = build_experts(args.expert_backend, model=args.expert_model,
                            model_revision=args.expert_model_revision,
                            participation_prior=args.participation_prior)
    corpus = ViewCorpus(args.view_corpus, mode=args.view_corpus_mode)
    attach_corpus(experts, corpus)

    from trading_agent.experiments.expert_weight_holdout import _to_date
    dates = [_to_date(d) for d in loader.get_date_range()]
    test_start = _first_test_date(dates, args.boundary, args.embargo_days)
    test_cfg = _copy_config(cfg, test_start, args.end)
    reflection = DeterministicExpertReflection(min_support=args.reflection_min_support,
                                               step_mode=args.reflection_step_mode)
    from trading_agent.data.market_proxy import MarketProxy
    proxy = MarketProxy.from_parquet(args.market_proxy)
    return (args, loader, test_cfg, experts, signal_sources, market_state_config,
            reflection, corpus, proxy)


def replay(ticker: str, *, certified_only: bool) -> dict:
    """Run the three memory arms plus equal weight, off the stored trained memory."""
    from trading_agent.experiments import expert_weight_holdout_launcher as L
    from trading_agent.runner import run_expert_loop
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    (args, loader, test_cfg, experts, signal_sources,
     market_state_config, reflection, corpus, proxy) = _setup(ticker)

    sealed = json.loads((SEALED / f"{ticker}__reprice.json").read_text(encoding="utf-8"))
    tau = float(sealed["selected_temperature"])
    theta = float(sealed["selected_decision_threshold"])
    snaps = sealed["arm_memory_snapshots"]

    def store_for(arm: str, *, writable: bool = False) -> "ExpertScorecardStore":
        payload = json.loads(json.dumps(snaps[arm]))   # deep copy; never mutate the artifact
        payload["certified_only"] = bool(certified_only)
        if writable:
            # The launcher seeds the online arm from the snapshot taken just BEFORE the
            # freeze. `freeze()` sets only these two fields and touches no card, so
            # clearing them restores that pre-freeze payload exactly.
            payload["frozen"], payload["frozen_at"] = False, None
        return ExpertScorecardStore.from_snapshot(payload, temperature=tau)

    common = dict(
        market_state_config=market_state_config, reflection=reflection,
        reflect_every=args.reflect_every, reflection_window=args.reflection_window,
        reflection_min_support=args.reflection_min_support,
        decision_threshold=theta, exit_threshold=args.exit_threshold,
        position_sizing=args.position_sizing, stale_view_abstain=args.stale_view_abstain,
        signal_sources=signal_sources, attribution_mode=args.attribution_mode,
        attribution_horizon=args.attribution_horizon, market_proxy=proxy,
        parallel_expert_calls=args.parallel_experts,
    )

    series: dict[str, list] = {}
    for arm in ("equal_weight", "global_memory", "market_state_memory"):
        result = run_expert_loop(loader, test_cfg, experts, weight_store=store_for(arm),
                                 allow_reflection=False, run_id=f"replay_{arm}", **common)
        series[arm] = L._daily_series(result)
    # The online arm seeds from the frozen STATE memory and keeps updating through the
    # test window; the stored snapshot for it is the end state, so it must be re-seeded.
    result = run_expert_loop(loader, test_cfg, experts,
                             weight_store=store_for("market_state_memory", writable=True),
                             allow_reflection=True, run_id="replay_online", **common)
    series["market_state_memory_online"] = L._daily_series(result)
    return {"ticker": ticker, "certified_only": certified_only,
            "selected_temperature": tau, "selected_decision_threshold": theta,
            "corpus_hits": corpus.hits, "daily_arm_series": series}


def _exposure(rows) -> list:
    return [(r[0], round(float(r[2]), 12)) for r in rows]


def compare(got: dict, truth_path: Path) -> tuple[bool, str]:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["daily_arm_series"]
    lines, ok = [], True
    for arm in sorted(got["daily_arm_series"]):
        a, b = _exposure(got["daily_arm_series"][arm]), _exposure(truth[arm])
        if a == b:
            lines.append(f"    {arm:28} {len(a):>4} days  identical")
            continue
        ok = False
        diff = sum(1 for x, y in zip(a, b) if x != y)
        lines.append(f"    {arm:28} {len(a):>4}/{len(b)} days  DIFFER on {diff}")
    return ok, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-ungated", action="store_true")
    ap.add_argument("--validate-verified", action="store_true")
    ap.add_argument("--run", default=None, help="comma-separated names to produce")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    if a.validate_ungated:
        names = sorted(p.name.split("__")[0] for p in OUT.glob("*__ungated.json"))
        if a.limit:
            names = names[: a.limit]
        print(f"reproducing the launcher's UNGATED arms on {len(names)} names\n")
        bad = []
        for t in names:
            got = replay(t, certified_only=False)
            ok, detail = compare(got, OUT / f"{t}__ungated.json")
            print(f"  {t}  {'OK' if ok else 'MISMATCH'}   (corpus hits {got['corpus_hits']})")
            print(detail)
            if not ok:
                bad.append(t)
        print(f"\n{len(names)-len(bad)}/{len(names)} reproduced exactly"
              + (f"; failed: {' '.join(bad)}" if bad else ""))
        return 1 if bad else 0

    if a.validate_gated:
        names = list(DAMAGED)[: a.limit] if a.limit else list(DAMAGED)
        print(f"reproducing the SEALED arms on the {len(names)} damaged names\n")
        bad = []
        for t in names:
            got = replay(t, certified_only=True)
            ok, detail = compare(got, SEALED / f"{t}__reprice.json")
            print(f"  {t}  {'OK' if ok else 'MISMATCH'}   (corpus hits {got['corpus_hits']})")
            print(detail)
            if not ok:
                bad.append(t)
        print(f"\n{len(names)-len(bad)}/{len(names)} reproduced exactly"
              + (f"; failed: {' '.join(bad)}" if bad else ""))
        return 1 if bad else 0

    if a.run:
        OUT.mkdir(parents=True, exist_ok=True)
        for t in [x.strip() for x in a.run.split(",") if x.strip()]:
            got = replay(t, certified_only=False)
            path = OUT / f"{t}__ungated.json"
            path.write_text(json.dumps(got, indent=1, sort_keys=True), encoding="utf-8")
            print(f"wrote {path.name}  ({len(got['daily_arm_series']['equal_weight'])} days)")
        return 0

    ap.error("pass --validate-ungated, --validate-verified or --run")


if __name__ == "__main__":
    raise SystemExit(main())
