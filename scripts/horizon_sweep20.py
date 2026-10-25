"""Move the attribution horizon and see which verified records survive it.

The population is the eight stocks that carry a verified record, which is where the
question lives: the system only acts on a record that passed, so whether such a record
holds up under a different horizon is what there is to ask. The other twelve have
nothing to be stable or unstable about.

Each name is launched with its OWN sealed configuration, read from its own report rather
than retyped here, because the twenty come from three separate runs that differ in corpus
path, news-sentiment cache and filing table. The only argument this script changes is
``--attribution-horizon``.

Run ``--horizons 21 --control`` first. A missing input does not fail here: it changes one
seat's prompt, that seat's replay lookups all miss, the seat falls back to its
data-unavailable default and drops out of both the vote and the learning evidence, and
the run still reports success. On BIIB a missing macro table served 4,383 of 8,766 views
and quietly tested two records instead of three. The control catches that by requiring
the sealed verification report back, record for record.

Cost never enters a verification report -- attribution is charged nothing -- so the rate
these runs pass through is immaterial to what they are read for.

    PYTHONPATH=src python scripts/horizon_sweep20.py --horizons 21 --control
    PYTHONPATH=src python scripts/horizon_sweep20.py --horizons 1,5,10,42,63 --width 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trading_agent.paths import data_root  # noqa: E402
OUT = ROOT / "reports" / "horizon20"

#: The panel, macro and proxy live outside the repository checkout.
DATA = data_root()

#: Names the pre-registered sweep already covered at 5/10/42/63 (reports/prereg/x*).
ALREADY = {"ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT"}

#: Filing table by provenance. The eleven names added with the draw were run against a
#: separate extract, and pointing them at the eight-name table silently empties the
#: fundamental seat.
SEC_EXPAND = DATA / "sec_xbrl" / "financials_expand20.parquet"
SEC_SEALED = DATA / "sec_xbrl" / "financials.parquet"
EXPAND20 = {"AFL", "AVY", "AZO", "BBY", "CTAS", "DDS", "INCY", "MTCH", "O", "RL", "TEX"}

#: The macro table goes into the news prompt, and the two populations sealed differently:
#: the runs under reports/final passed it and the ones expand_universe.py recorded did
#: not. Getting this wrong does not fail. It changes the news prompt, every one of that
#: seat's corpus lookups misses, the seat falls back to its data-unavailable default and
#: drops out of both the vote and the learning evidence, and the run still reports
#: success -- on BIIB it served 4,383 of 8,766 views and quietly tested two records
#: instead of three. That is why --control below is not optional.
MACRO = DATA / "macro" / "fred_macro.parquet"
NO_MACRO = EXPAND20


def sealed_manifest(ticker: str) -> dict:
    from portfolio20 import resolve
    sealed, missing = resolve()
    if ticker not in sealed:
        raise SystemExit(f"no sealed report for {ticker} (missing: {missing})")
    return json.loads(sealed[ticker].read_text(encoding="utf-8"))["manifest"]


def argv_for(ticker: str, horizon: int) -> list[str]:
    """The name's own sealed configuration, with one argument changed."""
    m = sealed_manifest(ticker)
    corpus = ROOT / m["view_corpus"]["path"]
    if not corpus.exists():
        raise SystemExit(f"{ticker}: frozen corpus absent at {corpus}")
    stem = OUT / f"{ticker}__h{horizon}"
    argv = [
        sys.executable, str(ROOT / "scripts" / "run_expert_weight_holdout.py"),
        "--tickers", ticker,
        "--data-root", str(DATA / "finsaber" / "sp500_2000_2025_parquet"),
        "--market-proxy", str(DATA / "market_proxy" / "spy_daily.parquet"),
        "--sec-facts-path", str(SEC_EXPAND if ticker in EXPAND20 else SEC_SEALED),
        *(() if ticker in NO_MACRO else ("--fred-macro-path", str(MACRO))),
        "--start", m["start"],
        "--validation-boundary", m["validation_boundary"],
        "--boundary", m["boundary"],
        "--end", m["end"],
        "--embargo-days", str(m["embargo_days"]),
        "--exit-threshold", str(m["exit_threshold"]),
        "--expert-backend", m["expert_backend"],
        "--expert-model", m["expert_model"],
        "--expert-model-revision", m["expert_model_revision"],
        "--reflection-backend", m["reflection_backend"],
        "--attribution-mode", m["attribution_mode"],
        "--reflect-every", str(m["reflect_every"]),
        "--reflection-window", str(m["reflection_window"]),
        "--temperature-grid", ",".join(str(x) for x in m["temperature_grid"]),
        "--attribution-horizon", str(horizon),
        "--view-corpus", str(corpus),
        # Strict identity replay. A miss falls back to the data-unavailable default and
        # quietly shrinks the committee, so the launcher refuses a run that serves none;
        # the per-name hit counts are checked again in --report.
        "--view-corpus-mode", "replay_by_date" if m.get("news_sentiment_cache") else "replay",
        "--headline-cost-bps-per-side", "10",
        "--sensitivity-cost-bps-per-side", "25",
        "--run-label", f"{ticker}_H{horizon}",
        "--output", str(stem) + ".json",
        "--memory-output", str(stem) + "_memory.json",
    ]
    if m.get("parallel_expert_calls"):
        argv.append("--parallel-experts")
    if m.get("terminal_verify"):
        argv.append("--terminal-verify")
    if m.get("certified_only_memory"):
        argv.append("--certified-only-memory")
    if m.get("news_sentiment_cache"):
        argv += ["--news-sentiment-cache", str(ROOT / m["news_sentiment_cache"]),
                 "--news-sentiment-model", m["news_sentiment_model"]]
    return argv


def run_all(names: list[str], horizons: list[int], width: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, h) for h in horizons for t in names
            if not (OUT / f"{t}__h{h}.json").exists()]
    print(f"{len(jobs)} runs to do, {width} at a time")
    live: list[tuple[str, int, subprocess.Popen, object]] = []
    started = time.time()
    failed = 0
    while jobs or live:
        while jobs and len(live) < width:
            t, h = jobs.pop(0)
            log = (OUT / f"{t}__h{h}.log").open("w", encoding="utf-8")
            p = subprocess.Popen(argv_for(t, h), cwd=str(ROOT), stdout=log,
                                 stderr=subprocess.STDOUT)
            live.append((t, h, p, log))
            print(f"  start {t} h={h}")
        time.sleep(5)
        for item in list(live):
            t, h, p, log = item
            if p.poll() is None:
                continue
            live.remove(item)
            log.close()
            ok = p.returncode == 0
            failed += 0 if ok else 1
            print(f"  {'done ' if ok else 'FAIL '} {t} h={h} "
                  f"({(time.time() - started) / 60:.0f} min elapsed)")
    print(f"finished, {failed} failures")
    return 1 if failed else 0


def passed_records(doc: dict) -> dict:
    """Every tested record with its corrected p and sample count."""
    out = {}
    for arm, b in (doc.get("verify_gate_report") or {}).items():
        for c, s in (b.get("per_cell") or {}).items():
            if isinstance(s, dict) and s.get("p_holm") is not None:
                out[f"{arm}|{c}"] = (round(s["p_holm"], 9), s.get("n"))
    return out


def control(names: list[str]) -> int:
    """Re-run each name at the sealed horizon and require the sealed report back.

    A horizon comparison is only worth reading if the replay reproduces the run it is
    compared against, and the failure it protects against is silent: a missing input
    changes one seat's prompt, that seat's lookups all miss, and the run completes with
    a smaller committee than the one it claims to be re-scoring.
    """
    from portfolio20 import resolve
    sealed, _ = resolve()
    bad = 0
    for t in names:
        p = OUT / f"{t}__h21.json"
        if not p.exists():
            print(f"  {t:5s} CONTROL NOT RUN")
            bad += 1
            continue
        got = json.loads(p.read_text(encoding="utf-8"))
        want = json.loads(sealed[t].read_text(encoding="utf-8"))
        gh = (got.get("manifest") or {}).get("view_corpus") or {}
        wh = (want.get("manifest") or {}).get("view_corpus") or {}
        # A sealed run that RECORDED its corpus counts most lookups as misses it then
        # wrote, so its hit count is not what a replay should equal. What both share is
        # the number of lookups: hits plus records written. Comparing hits directly
        # failed six names whose verification reports were in fact identical.
        lookups = (wh.get("hits") or 0) + (wh.get("records_written") or 0)
        ok_served = gh.get("hits") == lookups and not gh.get("records_written")
        gg, gw = passed_records(got), passed_records(want)
        if ok_served and gg == gw:
            print(f"  {t:5s} reproduces the sealed run  ({gh.get('hits')} views served, "
                  f"{len(gg)} records identical)")
        else:
            bad += 1
            why = []
            if not ok_served:
                why.append(f"served {gh.get('hits')} of {lookups} views")
            diff = [k for k in sorted(set(gg) | set(gw)) if gg.get(k) != gw.get(k)]
            if diff:
                why.append(f"{len(diff)} records differ, first "
                           f"{diff[0]}: {gg.get(diff[0])} vs {gw.get(diff[0])}")
            print(f"  {t:5s} DOES NOT REPRODUCE: {'; '.join(why)}")
    print(f"{len(names) - bad}/{len(names)} reproduce")
    return 1 if bad else 0


def report(names: list[str], horizons: list[int]) -> int:
    """Per run: how many views the replay served, and which records verified."""
    bad = 0
    for h in horizons:
        for t in names:
            p = OUT / f"{t}__h{h}.json"
            if not p.exists():
                print(f"  {t:5s} h={h:<3} MISSING")
                bad += 1
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            vc = (d.get("manifest") or {}).get("view_corpus") or {}
            ver = []
            for arm, b in (d.get("verify_gate_report") or {}).items():
                a = b.get("alpha", 0.10)
                ver += [f"{arm[0]}:{c}" for c, s in (b.get("per_cell") or {}).items()
                        if isinstance(s, dict) and s.get("p_holm") is not None
                        and s["p_holm"] <= a]
            print(f"  {t:5s} h={h:<3} entries={vc.get('entries'):>5} "
                  f"hits={vc.get('hits'):>5} verified={sorted(ver)}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="5,10,42,63")
    ap.add_argument("--names", default="")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="check the h=21 runs against the sealed reports")
    a = ap.parse_args(argv)
    hs = [int(x) for x in a.horizons.split(",")]
    from portfolio20 import resolve
    sealed, _ = resolve()
    names = ([x.strip() for x in a.names.split(",") if x.strip()]
             or sorted(set(sealed) - ALREADY))
    if a.control:
        return control(names)
    if a.report:
        return report(names, hs)
    return run_all(names, hs, a.width)


if __name__ == "__main__":
    raise SystemExit(main())
