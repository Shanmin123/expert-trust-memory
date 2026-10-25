"""Run the decision-layer trader arms the twenty-stock Section 6.6 needs.

Per stock: a control arm (no skills) for every name; a certified arm only where
verification passed something (eight names -- on the other twelve there is no sentence
to add and the treated arm would be byte-identical to the control); a producer arm for
every name. Existing reports are kept, so this is resumable and only fills gaps.

All arms of one stock share one prompt cache, which `check_pair` later requires. A run
whose views file is not there yet is skipped and reported, so this can be started while
the view exports are still running and re-run afterwards.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/run_trader_arms.py --dry-run
    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/run_trader_arms.py --width 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio20 import resolve  # noqa: E402
from trading_agent.paths import finsaber_panel  # noqa: E402

DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")
# The eight where verification passed anything; only they have certified sentences.
VERIFIED = ("ADSK", "AES", "AZO", "BIIB", "CTAS", "DDS", "ECL", "INCY")
VIEWS = ROOT / "reports" / "views_daily"
FINAL = ROOT / "reports" / "final"
LESSONS = ROOT / "reports" / "lessons"
CORPUS = ROOT / "reports" / "corpus"


def jobs() -> list[tuple[str, str, list[str]]]:
    sealed, missing = resolve()
    assert not missing, missing
    out = []
    for t in DRAW:
        views = VIEWS / f"{t}.jsonl"
        cache = CORPUS / f"llm_decision_{t}.jsonl"
        canon = sealed[t]
        base = [sys.executable, "scripts/run_llm_decision_layer.py",
                "--views", str(views), "--canonical-report", str(canon),
                "--cache", str(cache),
                "--data-root", str(finsaber_panel())]
        arms = [("none", ["--skills-mode", "none"],
                 FINAL / f"{t}__llm_decision_none.json")]
        if t in VERIFIED:
            arms.append(("certified", ["--skills-mode", "certified"],
                         FINAL / f"{t}__llm_decision_certified.json"))
        arms.append(("producer",
                     ["--skills-file", str(LESSONS / f"{t}_llm_producer.json")],
                     FINAL / f"{t}__llm_decision_producer.json"))
        for label, extra, outfile in arms:
            if outfile.exists():
                continue
            out.append((t, label, base + extra + ["--out", str(outfile)]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    todo = jobs()
    ready = [j for j in todo if (VIEWS / f"{j[0]}.jsonl").exists()]
    waiting = sorted({j[0] for j in todo} - {j[0] for j in ready})
    print(f"jobs missing: {len(todo)}  runnable now: {len(ready)}"
          + (f"  waiting on views: {' '.join(waiting)}" if waiting else ""))
    if a.dry_run:
        for t, label, cmd in ready[:4]:
            print(f"  {t} {label}: {' '.join(cmd[-6:])}")
        return 0

    logs = FINAL / "trader_logs"
    logs.mkdir(exist_ok=True)
    running, ok, bad = [], [], []
    queue = list(ready)
    env = {**os.environ, "PYTHONPATH": "src"}
    while queue or running:
        # An arm of one stock must not run beside another arm of the SAME stock:
        # they share a cache file and the writer appends without a lock.
        active = {t for t, _, _ in [(r[0], r[1], None) for r in running]}
        k = 0
        while k < len(queue) and len(running) < a.width:
            t, label, cmd = queue[k]
            if t in active:
                k += 1
                continue
            queue.pop(k)
            log = open(logs / f"{t}__{label}.log", "w", encoding="utf-8")
            print(f"start {t} {label}")
            running.append((t, label, subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env),
                time.time()))
            active.add(t)
        time.sleep(5)
        for entry in list(running):
            t, label, p, t0 = entry
            if p.poll() is None:
                continue
            running.remove(entry)
            mins = (time.time() - t0) / 60
            if p.returncode == 0:
                ok.append(f"{t}/{label}")
                print(f"  done {t} {label} in {mins:.1f} min")
            else:
                bad.append(f"{t}/{label}")
                print(f"  FAILED {t} {label} rc={p.returncode} "
                      f"(see trader_logs/{t}__{label}.log)")
    print(f"\ncompleted {len(ok)}; failed {len(bad)}"
          + (f": {' '.join(bad)}" if bad else ""))
    # Surface parse errors and parity results without opening every log by hand.
    for t in DRAW:
        for label in ("none", "certified", "producer"):
            f = FINAL / f"{t}__llm_decision_{label}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            pe = d.get("n_parse_errors", 0)
            if pe:
                print(f"  parse errors {t}/{label}: {pe}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
