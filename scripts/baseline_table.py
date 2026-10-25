"""Score the baselines the way the arms are scored, and print the comparison.

Cost model, matching scripts/portfolio_arm_metrics.py: 10 bps per side on every change in
exposure, then annualised Sharpe on the net daily series. The equal-capital portfolio is
the mean of the eight per-stock net series, matching the confirmatory endpoint.

COSTS MUST BE CHARGED EXACTLY ONCE, and the baselines do not agree on where they live.
A series taken from a foreign engine's equity curve is already net of that engine's own
commissions; a series this project priced from the panel is gross. Charging the gross rule
on an already-net series double-counts, and the error is invisible in the output.

So every baseline manifest must declare ``returns``: ``"net"`` (costs inside, charge
nothing more) or ``"gross"`` (charge here). A file that does not declare it is an error
rather than a guess. This mattered as soon as the FinMem exposure stopped being a constant:
with a real long/cash path, the double charge lands on roughly 40% of days.

    PYTHONPATH=src python scripts/baseline_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "baselines"
FINAL = ROOT / "reports" / "final"
COST = 10.0 / 1e4
MIN_DAYS = 300      # the arms score 352; anything far below this is a run still in flight

SEALED = {"TSLA": "TSLA__designD3_q_h21cert", "ADSK": "ADSK__designD3_q_h21cert",
          "BIIB": "BIIB__designD3_q_h21cert", "CVX": "CVX__final", "DHI": "DHI__sealed",
          "ECL": "ECL__sealed", "PNC": "PNC__sealed", "TGT": "TGT__sealed"}
TICKERS = list(SEALED)


def use_universe20():
    """Switch the module to the twenty-stock random draw (TSLA is not in it)."""
    global TICKERS, SEALED
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from portfolio20 import check_config, resolve
    found, missing = resolve()
    if missing:
        raise SystemExit(f"universe20 incomplete: {' '.join(missing)}")
    check_config(found)
    global _RESOLVED
    _RESOLVED = dict(found)
    SEALED = {t: p.stem for t, p in found.items()}
    TICKERS = sorted(SEALED)


def net(series, charge=True):
    """Charge cost on exposure changes, exactly as portfolio_arm_metrics does.

    ``charge=False`` passes the returns through untouched, for series that already carry
    the costs their own engine applied.
    """
    out, prev = {}, 0.0
    for d, gross, exp in series:
        out[d] = float(gross) - (COST * abs(float(exp) - prev) if charge else 0.0)
        prev = float(exp)
    return out


def sharpe(rets):
    if len(rets) < 2:
        return 0.0
    sd = stdev(rets)
    return 0.0 if sd == 0 else fmean(rets) / sd * math.sqrt(252)


def total_return(rets):
    lvl = 1.0
    for r in rets:
        lvl *= 1.0 + r
    return lvl - 1.0


def max_drawdown(rets):
    lvl, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        lvl *= 1.0 + r
        peak = max(peak, lvl)
        mdd = max(mdd, (peak - lvl) / peak)
    return mdd


def portfolio(per_stock):
    """Equal-capital: mean of the eight net series over their shared dates."""
    common = set.intersection(*(set(v) for v in per_stock.values()))
    dates = sorted(common)
    return [fmean(per_stock[t][d] for t in per_stock) for d in dates], len(dates)


def collect_baseline(strategy):
    per = {}
    for t in TICKERS:
        p = BASE / f"{t}__{strategy}.json"
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        basis = payload.get("manifest", {}).get("returns")
        if basis not in ("net", "gross"):
            raise SystemExit(
                f"{p.name}: manifest must declare returns as 'net' or 'gross'; got "
                f"{basis!r}. Without it the cost model cannot know whether this series "
                f"already carries its engine's commissions, and guessing double-charges."
            )
        per[t] = net(payload["daily_arm_series"]["equal_weight"], charge=(basis == "gross"))
    return per


_RESOLVED = None


def collect_arm(arm):
    per = {}
    for t, stem in SEALED.items():
        path = (_RESOLVED[t] if _RESOLVED
                else FINAL / f"{stem}.json")
        s = json.loads(path.read_text(encoding="utf-8"))
        per[t] = net(s["daily_arm_series"][arm])
    return per


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe20", action="store_true",
                    help="score the twenty-stock random draw instead of the eight")
    args = ap.parse_args()
    if args.universe20:
        use_universe20()
    strategies = sorted({p.name.split("__", 1)[1].removesuffix(".json")
                         for p in BASE.glob("*__*.json")
                         if p.parent == BASE})
    rows, partial = [], []
    for s in strategies:
        per = collect_baseline(s)
        if per is None:
            continue
        rets, n = portfolio(per)
        # A run still in progress has files for every stock but only the dates it has
        # reached, and the shared-date intersection can collapse to a handful. Scoring
        # that produces a confident-looking Sharpe from a few days, so it is refused.
        if n < MIN_DAYS:
            partial.append((s.removesuffix("Strategy"), n))
            continue
        per_stock_sr = [sharpe(list(per[t].values())) for t in TICKERS]
        rows.append((s.removesuffix("Strategy"), sharpe(rets), total_return(rets),
                     max_drawdown(rets), fmean(per_stock_sr), n))

    for arm, label in (("equal_weight", "Equal-weight committee"),
                       ("global_memory", "Verified global (this work)"),
                       ("market_state_memory", "Verified state (this work)")):
        per = collect_arm(arm)
        rets, n = portfolio(per)
        per_stock_sr = [sharpe(list(per[t].values())) for t in TICKERS]
        rows.append((label, sharpe(rets), total_return(rets),
                     max_drawdown(rets), fmean(per_stock_sr), n))

    print(f"{'arm / strategy':<30}{'port SR':>9}{'CR %':>9}{'MDD %':>8}"
          f"{'mean stock SR':>15}{'days':>6}")
    print("-" * 77)
    for name, sr, cr, mdd, msr, n in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<30}{sr:>9.3f}{cr*100:>9.2f}{mdd*100:>8.2f}{msr:>15.3f}{n:>6d}")
    for name, n in partial:
        print(f"{name:<30}{'INCOMPLETE - ' + str(n) + ' shared days, not scored':>50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
