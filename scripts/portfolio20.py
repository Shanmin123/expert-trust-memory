"""Assemble the twenty-name random universe and score it beside the sealed eight.

The twenty names are one fixed-seed draw (``reports/universe/random_universe_20.json``,
seed 42, 436 eligible names) that contains the eight already evaluated, so this is the same
sample at a larger n rather than a second search. Their reports live in three places:

  * seven from the sealed runs (``reports/final``) -- TSLA is not in the draw;
  * C and AES from the first registered extension, whose manifests match the sealed config
    on all twenty configuration keys (checked, not assumed);
  * eleven from ``reports/expand20``, recorded by ``expand_universe.py``.

TSLA is reported separately. It was held out of the draw and added because the two agent
baselines demonstrate on it, so folding it into a random universe would misdescribe both.

The sealed eight-name result is not replaced. This is a registered replication at n=20 and
belongs beside it, which is also what the power analysis predicts will happen: more random
names dilute a mean carried by two of them.

    PYTHONPATH=src python scripts/portfolio20.py --list
    PYTHONPATH=src python scripts/portfolio20.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"
EXPAND = ROOT / "reports" / "expand20"
UNIVERSE_20 = ROOT / "reports" / "universe" / "random_universe_20.json"

# The sealed stems, for the names the draw and the sealed eight share.
SEALED = {"ADSK": "ADSK__designD3_q_h21cert", "BIIB": "BIIB__designD3_q_h21cert",
          "CVX": "CVX__final", "DHI": "DHI__sealed", "ECL": "ECL__sealed",
          "PNC": "PNC__sealed", "TGT": "TGT__sealed"}
FIRST_EXTENSION = {"C": "C__designD3_q_rec", "AES": "AES__designD3_q_rec"}

# Configuration keys that must agree for the reports to be poolable at all.
CONFIG_KEYS = ("attribution_horizon", "attribution_mode", "reflection_backend",
               "exit_threshold", "embargo_days", "start", "validation_boundary",
               "boundary", "end", "expert_model", "terminal_verify",
               "certified_only_memory", "bounded_memory", "standardized_memory",
               "reflect_every", "reflection_window", "vote_policy", "regime_mode")


def resolve() -> tuple[dict[str, Path], list[str]]:
    """Ticker -> report path for every name of the draw that has been run."""
    draw = json.loads(UNIVERSE_20.read_text(encoding="utf-8"))["selected"]
    if len(draw) != 20 or len(set(draw)) != 20:
        raise SystemExit(f"the recorded draw must be twenty unique names, got {len(draw)}")
    found, missing = {}, []
    for ticker in draw:
        if ticker in SEALED:
            path = FINAL / f"{SEALED[ticker]}.json"
        elif ticker in FIRST_EXTENSION:
            path = FINAL / f"{FIRST_EXTENSION[ticker]}.json"
        else:
            path = EXPAND / f"{ticker}__expand.json"
        (found.setdefault(ticker, path) if path.exists() else missing.append(ticker))
    return found, missing


def check_config(found: dict[str, Path]) -> None:
    """Refuse to pool reports whose configuration differs; that would not be one arm."""
    baseline_name = "DHI" if "DHI" in found else next(iter(found))
    base = json.loads(found[baseline_name].read_text(encoding="utf-8"))["manifest"]
    problems = []
    for ticker, path in found.items():
        m = json.loads(path.read_text(encoding="utf-8"))["manifest"]
        for key in CONFIG_KEYS:
            if m.get(key) != base.get(key):
                problems.append(f"{ticker}.{key}={m.get(key)!r} vs {base.get(key)!r}")
    if problems:
        raise SystemExit("configuration differs across reports, so they are not one "
                         "experiment:\n  " + "\n  ".join(problems[:12]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show what is present and missing")
    ap.add_argument("--n-boot", type=int, default=40000,
                    help="2000 leaves about a point of Monte Carlo noise on each p")
    ap.add_argument("--bps", type=float, default=10.0,
                    help="cost per side; 10 is the headline, 25 the registered sensitivity")
    ap.add_argument("--out", default="reports/final/portfolio_20name.json")
    args = ap.parse_args(argv)

    found, missing = resolve()
    print(f"{len(found)} of 20 names present")
    if missing:
        print(f"still to record: {' '.join(missing)}")
    if args.list:
        for ticker, path in sorted(found.items()):
            print(f"  {ticker:6} {path.relative_to(ROOT)}")
        return 0
    if missing:
        print("\nrefusing to score a partial universe: a subset of a random draw is not "
              "a random sample of the pool, and the number would not mean what it says.",
              file=sys.stderr)
        return 1

    check_config(found)
    cmd = [sys.executable, "scripts/portfolio_arm_metrics.py"]
    cmd += [str(p) for _, p in sorted(found.items())]
    cmd += ["--cost-bps-per-side", str(args.bps), "--n-boot", str(args.n_boot),
            "--out", args.out]
    print(f"\nscoring {len(found)} names at {args.bps} bps, n_boot={args.n_boot} "
          f"-> {args.out}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
