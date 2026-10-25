"""The learner without the verification step, on the twenty-name draw.

Chapter 6 argues that verification earns its place because the same learner, left
ungated, loses on most configurations. Until now that claim rested on two development
stocks. This runs the ungated arm on the draw itself.

The only difference from the sealed configuration is that `--certified-only-memory` is
dropped, so every learned cell may move a weight instead of only the verified ones.
Everything else is copied from the sealed manifest.

The corpus is opened in `replay_by_date`, not `readthrough`: a cached view is served, a
miss is an error rather than a model call. So this costs nothing and cannot silently
introduce a second set of expert judgements. The experts never see the weights, so the
prompts are the ones the sealed run already recorded.

Four names cannot be re-run here. MTCH, O, RL and TEX hold 1,255 to 2,457 views against
the 3,637 dates a full run reads, which is the same gap Chapter 5 discloses, and the
missing part is the training window. Back-filling it would call the model and put a
second set of judgements into a comparison whose whole point is that both arms read one
set. They do not need it: their trained memory is stored and their test window is
complete, so `ungated_replay.py` replays them off the stored memory instead.

    PYTHONPATH=src python scripts/ungated_twenty.py --dry-run
    PYTHONPATH=src python scripts/ungated_twenty.py --only AZO
    PYTHONPATH=src python scripts/ungated_twenty.py --width 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from trading_agent.paths import data_root, finsaber_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ungated20"
MODEL = "gpt-4.1-mini-2025-04-14"

# The corpus each name's sealed run actually read; the naming differs by era, and the
# plain views_<T>.jsonl file is NOT always it. C and CVX both have an older four-expert
# recording under the plain name whose fundamental views differ completely from the
# sealed ones (0% long against 52% long on C), so picking a corpus by file size or by
# view count silently reads a different set of judgements. The guard is the equal-weight
# arm: it ignores the memory, so a correct replay reproduces the sealed one day for day.
CORPUS = {
    "ADSK": "views_ADSK__designD3_q.jsonl", "AES": "views_AES__designD3_q.jsonl",
    "BIIB": "views_BIIB__designD3_q.jsonl", "C": "views_C__designD3_q.jsonl",
    "CVX": "views_CVX__final.jsonl", "DHI": "views_DHI__sealed.jsonl",
    "ECL": "views_ECL__sealed.jsonl", "PNC": "views_PNC__sealed.jsonl",
    "TGT": "views_TGT__sealed.jsonl",
}
for t in ("AFL", "AVY", "AZO", "BBY", "CTAS", "DDS", "INCY"):
    CORPUS[t] = f"views_{t}.jsonl"
# These four hold only the tail of their corpus, so the launcher cannot re-train them
# here. Their entries are still needed: `ungated_replay.py` imports `args_for` to
# rebuild the same configuration and replays their TEST window, which is complete,
# off the stored trained memory. Each name's manifest records exactly this file.
for t in ("MTCH", "O", "RL", "TEX"):
    CORPUS[t] = f"views_{t}.jsonl"

# Short corpora: 1,255 to 2,457 views against 3,637 dates. Run by ungated_replay.py.
SKIP = ("MTCH", "O", "RL", "TEX")
NAMES = sorted(t for t in CORPUS if t not in SKIP)

SEC = "data/sec_xbrl/financials_expand20.parquet"


def args_for(ticker: str) -> list[str]:
    stem = OUT / f"{ticker}__ungated"
    return [
        "--data-root", str(finsaber_panel()),
        "--tickers", ticker,
        "--market-proxy", str(data_root() / "market_proxy" / "spy_daily.parquet"),
        "--start", "2011-07-01",
        "--validation-boundary", "2023-12-31",
        "--boundary", "2024-06-30",
        "--end", "2025-12-31",
        "--embargo-days", "5",
        "--exit-threshold", "0",
        "--expert-backend", "openai",
        "--expert-model", MODEL,
        "--expert-model-revision", MODEL,
        "--reflection-backend", "deterministic_scaled",
        "--attribution-horizon", "21",
        "--attribution-mode", "raw",
        "--reflect-every", "21",
        "--reflection-window", "63",
        "--parallel-experts",
        "--terminal-verify",
        # --certified-only-memory deliberately absent: this is the ungated arm.
        "--temperature-grid", "0.5,1,2",
        "--sec-facts-path", SEC,
        "--view-corpus", f"reports/corpus/{CORPUS[ticker]}",
        "--view-corpus-mode", "replay_by_date",
        "--headline-cost-bps-per-side", "10",
        "--sensitivity-cost-bps-per-side", "25",
        "--run-label", f"{ticker}_UNGATED20",
        "--output", str(stem) + ".json",
        "--memory-output", str(stem) + "_memory.json",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    todo = [a.only] if a.only else [
        t for t in NAMES if not (OUT / f"{t}__ungated.json").exists()]
    print(f"skipped for a short corpus: {' '.join(SKIP)}")
    print(f"to run ({len(todo)}): {' '.join(todo)}")
    if a.dry_run:
        print("\n" + " ".join(args_for(todo[0])) if todo else "")
        return 0

    running: list[tuple[str, subprocess.Popen, float]] = []
    ok, bad = [], []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < a.width:
            t = queue.pop(0)
            log = open(OUT / "logs" / f"{t}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "-m",
                   "trading_agent.experiments.expert_weight_holdout_launcher"] + args_for(t)
            print(f"start {t}")
            running.append((t, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log,
                                                stderr=subprocess.STDOUT), time.time()))
        time.sleep(5)
        for entry in list(running):
            t, p, t0 = entry
            if p.poll() is None:
                continue
            running.remove(entry)
            mins = (time.time() - t0) / 60
            if p.returncode == 0:
                ok.append(t)
                print(f"  done {t} in {mins:.1f} min")
            else:
                bad.append(t)
                print(f"  FAILED {t} rc={p.returncode} after {mins:.1f} min "
                      f"(see logs/{t}.log)")
    print(f"\nfinished {len(ok)}: {' '.join(ok)}")
    if bad:
        print(f"failed {len(bad)}: {' '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
