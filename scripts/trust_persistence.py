"""Does an expert's measured trust persist from one period to the next?

Section 6.5 uses this to explain why a verified deviation can be real in training and
absent afterwards. The check splits each expert's attributed series inside the training
window and asks whether the two halves agree.

The series is not recomputed here. The script runs the project's own
`run_expert_loop` over the training window and reads `signal_attribution` off the
episodes it returns, which is the same quantity the memory is built from
(`g = v * r`, `v` from the expert's own direction, `r` the 21-bar forward return,
abstentions and safe defaults excluded). Reimplementing the parse would risk
disagreeing with the memory in a way the numbers could not reveal.

Two guards. The reconstructed episode counts and hit counts must equal the stored
global scorecard's, cell for cell, or the series is not the one the memory saw. And the
run reads the frozen view corpus in `replay_by_date`, where a miss is an error, so a
completed pass called no model.

Coverage is the sixteen stocks whose stored views span the training window. MTCH, O,
RL and TEX lost part of theirs and are absent here for that reason alone.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/trust_persistence.py --collect
    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/trust_persistence.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "reports" / "persistence"
EXPERTS = ("technical", "news", "fundamental")
HALFLIFE = 63


def collect(ticker: str, window: str = "train") -> dict:
    """Run one window and return each expert's dated contribution series."""
    import ungated_replay as UR
    from trading_agent.experiments.expert_weight_holdout import _copy_config, _to_date
    from trading_agent.runner import run_expert_loop
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    (args, loader, _test_cfg, experts, signal_sources,
     market_state_config, reflection, corpus, proxy) = UR._setup(ticker)
    # The memory is trained from the run's start to the validation boundary; the test
    # window is the config the replayer already builds.
    cfg = (_test_cfg if window == "test" else
           _copy_config(_test_cfg, _to_date(args.start), _to_date(args.validation_boundary)))

    result = run_expert_loop(
        loader, cfg, experts,
        market_state_config=market_state_config, reflection=reflection,
        reflect_every=args.reflect_every, reflection_window=args.reflection_window,
        reflection_min_support=args.reflection_min_support,
        decision_threshold=0.0, exit_threshold=args.exit_threshold,
        position_sizing=args.position_sizing, stale_view_abstain=args.stale_view_abstain,
        # Weights cannot change the answer: signal_attribution is counterfactual, scored
        # on each expert's own vote rather than on the ensemble. An equal-weight store
        # keeps that explicit.
        weight_store=ExpertScorecardStore(),
        allow_reflection=False, signal_sources=signal_sources,
        attribution_mode=args.attribution_mode,
        attribution_horizon=args.attribution_horizon,
        market_proxy=proxy, parallel_expert_calls=args.parallel_experts,
        run_id=f"persistence_{window}_{ticker}",
    )
    series: dict[str, list] = {e: [] for e in EXPERTS}
    for ep in result.episodes:
        for expert_id, g in ep.signal_attribution.items():
            if expert_id in series:
                series[expert_id].append([ep.decision_date.isoformat(), float(g)])
    for e in series:
        series[e].sort()
    return {"ticker": ticker, "corpus_hits": corpus.hits, "series": series}


def _ew(values) -> tuple[float, float]:
    alpha = 1.0 - 0.5 ** (1.0 / HALFLIFE)
    mean = var = 0.0
    for k, x in enumerate(values):
        if k == 0:
            mean, var = x, 0.0
            continue
        d = x - mean
        mean += alpha * d
        var = (1.0 - alpha) * (var + alpha * d * d)
    return mean, var


def validate(ticker: str, data: dict) -> list[str]:
    """The reconstructed series must reproduce the stored global scorecard exactly."""
    snap = json.loads((ROOT / "reports" / "reprice" / f"{ticker}__reprice.json")
                      .read_text(encoding="utf-8"))["arm_memory_snapshots"]["global_memory"]
    stored = {c["expert_id"]: c for c in snap["cards"]}
    bad = []
    for e in EXPERTS:
        g = [x[1] for x in data["series"][e]]
        card = stored.get(e)
        if card is None:
            bad.append(f"{ticker}/{e}: no stored card")
            continue
        hits = sum(1 for x in g if x > 0.0)
        mean, var = _ew(g)
        if len(g) != int(card["count"]) or hits != int(card["hits"]):
            bad.append(f"{ticker}/{e}: n {len(g)} vs {card['count']}, "
                       f"hits {hits} vs {card['hits']}")
        elif (abs(mean - float(card["ew_mean"])) > 1e-9
              or abs(var - float(card["ew_var"])) > 1e-9):
            bad.append(f"{ticker}/{e}: ew_mean {mean:.8f} vs {card['ew_mean']:.8f}")
    return bad


def corr(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else 0.0


def stats(g) -> tuple[float, float]:
    """Hit rate and mean contribution over a block."""
    return sum(1 for x in g if x > 0) / len(g), sum(g) / len(g)


def compute(files) -> dict:
    """Every number Section 6.5's persistence subsection quotes, in one place.

    The thesis verifier calls this rather than repeating the construction, so the
    prose and the check cannot drift apart into two different definitions.
    """
    data = {p.stem.split("__")[0]: json.loads(p.read_text(encoding="utf-8")) for p in files}
    names = sorted(data)
    out: dict = {"names": names}

    # --- long windows: split the training episodes in half ------------------------
    cells, hr, ev = 0, ([], []), ([], [])
    for t in names:
        for e in EXPERTS:
            g = [x[1] for x in data[t]["series"][e]]
            h = len(g) // 2
            if min(h, len(g) - h) < 30:
                continue
            cells += 1
            a, b = stats(g[:h]), stats(g[h:])
            hr[0].append(a[0]); hr[1].append(b[0])
            ev[0].append(a[1]); ev[1].append(b[1])
    out["split_half"] = {"cells": cells, "hit": corr(*hr), "contribution": corr(*ev)}

    # --- eighteen-month blocks, the length of the test window ----------------------
    # The question is whether the RANKING of experts carries from one period to the
    # next, so each correlation is taken ACROSS cells within one time offset, and the
    # offsets are then averaged. Pooling every (cell, offset) point into one
    # correlation answers a different question and gives a different number.
    BLOCK = 378                      # about eighteen months of trading days
    by_offset: dict = {}
    for t in names:
        for e in EXPERTS:
            byday: dict = {}
            for d, g in data[t]["series"][e]:
                byday.setdefault(d, []).append(g)
            days = sorted(byday)
            k = 0
            while (k + 2) * BLOCK <= len(days):
                a = [g for d in days[k * BLOCK:(k + 1) * BLOCK] for g in byday[d]]
                b = [g for d in days[(k + 1) * BLOCK:(k + 2) * BLOCK] for g in byday[d]]
                if min(len(a), len(b)) >= 30:
                    by_offset.setdefault(k, []).append((stats(a), stats(b)))
                k += 1
    hr_by_offset, ev_by_offset = [], []
    for k in sorted(by_offset):
        pts = by_offset[k]
        if len(pts) < 5:
            continue
        hr_by_offset.append(corr([p[0][0] for p in pts], [p[1][0] for p in pts]))
        ev_by_offset.append(corr([p[0][1] for p in pts], [p[1][1] for p in pts]))
    if hr_by_offset:
        out["periods"] = {
            "n": len(hr_by_offset),
            "hit_mean": statistics.mean(hr_by_offset),
            "hit_sd": statistics.pstdev(hr_by_offset),
            "hit_range": [min(hr_by_offset), max(hr_by_offset)],
            "contribution_mean": statistics.mean(ev_by_offset),
            "contribution_sd": statistics.pstdev(ev_by_offset),
            "contribution_range": [min(ev_by_offset), max(ev_by_offset)],
            "cells_per_period": [len(by_offset[k]) for k in sorted(by_offset)],
        }

    # --- the draw that actually happened ------------------------------------------
    # The periods above are what an eighteen-month gap looks like on average. This is
    # the one the evaluation drew: training statistic against test-window statistic,
    # across the cells that carry enough episodes on both sides to be measured.
    MIN = 30
    pairs = []
    for t in names:
        test_path = OUT / f"{t}__test.json"
        if not test_path.exists():
            continue
        test = json.loads(test_path.read_text(encoding="utf-8"))
        for e in EXPERTS:
            a = [x[1] for x in data[t]["series"][e]]
            b = [x[1] for x in test["series"][e]]
            if min(len(a), len(b)) < MIN:
                continue
            pairs.append((stats(a), stats(b), t, e))
    if len(pairs) >= 5:
        out["draw"] = {
            "cells": len(pairs),
            "min_episodes": MIN,
            "hit": corr([p[0][0] for p in pairs], [p[1][0] for p in pairs]),
            "contribution": corr([p[0][1] for p in pairs], [p[1][1] for p in pairs]),
            "dropped": [[t, e] for t in names for e in EXPERTS
                        if (OUT / f"{t}__test.json").exists()
                        and len(json.loads((OUT / f"{t}__test.json")
                                           .read_text(encoding="utf-8"))["series"][e]) < MIN],
        }
    return out


def report(files) -> None:
    r = compute(files)
    print(f"stocks: {len(r['names'])}  {' '.join(r['names'])}\n")
    s = r["split_half"]
    print(f"split-half over {s['cells']} cells: hit-rate corr {s['hit']:+.3f}, "
          f"contribution corr {s['contribution']:+.3f}")
    p = r.get("periods")
    if p:
        print(f"adjacent eighteen-month periods, {p['n']} of them, correlated across "
              f"cells and averaged:")
        print(f"  hit rate    mean {p['hit_mean']:+.3f}  sd {p['hit_sd']:.3f}  "
              f"range {p['hit_range'][0]:+.2f} to {p['hit_range'][1]:+.2f}")
        print(f"  E[v.r]      mean {p['contribution_mean']:+.3f}  "
              f"sd {p['contribution_sd']:.3f}  "
              f"range {p['contribution_range'][0]:+.2f} to {p['contribution_range'][1]:+.2f}")
        print(f"  cells per period: {p['cells_per_period']}")
    d = r.get("draw")
    if d:
        print(f"the actual training-to-test draw, {d['cells']} cells "
              f"(at least {d['min_episodes']} episodes each side):")
        print(f"  hit rate    corr {d['hit']:+.3f}")
        print(f"  E[v.r]      corr {d['contribution']:+.3f}")
        if d["dropped"]:
            print(f"  dropped for thin test-window support: {d['dropped']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--window", choices=("train", "test"), default="train")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.collect:
        from ungated_twenty import NAMES
        w = a.window
        todo = [a.only] if a.only else [
            t for t in NAMES if not (OUT / f"{t}__{w}.json").exists()]
        print(f"collecting {w} for {len(todo)}: {' '.join(todo)}")
        for t in todo:
            data = collect(t, window=w)
            # The scorecard guard only applies to the training window: the stored card
            # is what training produced, and the test window never fed it.
            bad = validate(t, data) if w == "train" else []
            if bad:
                print(f"  {t} REJECTED: {bad}")
                continue
            (OUT / f"{t}__{w}.json").write_text(json.dumps(data), encoding="utf-8")
            n = {e: len(data['series'][e]) for e in EXPERTS}
            note = ("reproduces the stored scorecard" if w == "train"
                    else "scorecard guard n/a: the stored card is the training one")
            print(f"  {t} ok, {note}; episodes {n}")
        return 0

    if a.report:
        files = sorted(OUT.glob("*__train.json"))
        if not files:
            print("nothing collected yet")
            return 1
        report(files)
        return 0

    ap.error("pass --collect or --report")


if __name__ == "__main__":
    raise SystemExit(main())
