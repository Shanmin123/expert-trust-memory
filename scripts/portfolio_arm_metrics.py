"""Portfolio-level PRIMARY metric across names (protocol repair item 5, SIMPLE version).

Mean-of-per-name-Sharpes ignores cross-name covariance; the primary result is instead an
equal-capital, date-synchronized daily portfolio per arm, with a PAIRED moving-block
bootstrap CI on the arm-minus-equal_weight mean daily difference.

Input: holdout report JSONs (launcher output) that carry "daily_arm_series"
       {arm: [[iso_date, gross_return, long_exposure], ...]} — reports produced before that
       export exist will be rejected with a clear message.
Cost:  charged identically per arm from exposure changes: cost_t = bps/1e4 * |exp_t - exp_{t-1}|.

Usage:
    PYTHONPATH=src python scripts/portfolio_arm_metrics.py reports/final/A.json B.json \
        --cost-bps-per-side 10 [--arm market_state_memory] [--n-boot 2000] [--out out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean

BASELINE_ARM = "equal_weight"
SEED = 20260716


def _load_series(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload.get("daily_arm_series")
    if not series:
        raise SystemExit(
            f"{path}: no 'daily_arm_series' — re-run the holdout with the current launcher "
            "(free replay of the cached corpus) to export per-date arm returns."
        )
    return {
        arm: {d: (float(g), float(x)) for d, g, x in rows}
        for arm, rows in series.items()
    }


#: FINSABER's own schedule, charged when the bps rate is zero. The commission is a
#: per-share amount, so it needs the price each trade fills at, which under this
#: execution is the next session's open.
PER_SHARE, MIN_COMMISSION, MAX_RATE, CASH = 0.0049, 0.99, 0.01, 100_000.0
_PRICES: dict = {}


def _opens(ticker: str):
    if ticker not in _PRICES:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from reprice_next_open import adjusted_prices
        o, _c = adjusted_prices([ticker])
        _PRICES[ticker] = (o[ticker], sorted(o[ticker]))
    return _PRICES[ticker]


def _net_returns(per_date: dict[str, tuple[float, float]], dates: list[str], bps: float,
                 ticker: str | None = None) -> list[float]:
    """Net of cost. A positive bps rate is flat; zero charges the commission."""
    if bps > 0 or ticker is None:
        out, prev_x = [], 0.0
        for d in dates:
            gross, x = per_date[d]
            out.append(gross - (bps / 1e4) * abs(x - prev_x))
            prev_x = x
        return out
    opens, cal = _opens(ticker)
    idx = {d: i for i, d in enumerate(cal)}
    out, prev_x, equity = [], 0.0, CASH
    for d in dates:
        gross, x = per_date[d]
        i = idx.get(d)
        nxt = cal[i + 1] if i is not None and i + 1 < len(cal) else None
        px = opens.get(nxt) if nxt else None
        traded = abs(x - prev_x)
        frac = (min(max(PER_SHARE / px, MIN_COMMISSION / equity), MAX_RATE)
                if (traded and px) else 0.0)
        r = gross - frac * traded
        out.append(r)
        equity *= (1.0 + r)
        prev_x = x
    return out


def _ann_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = fmean(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(252.0) * mu / math.sqrt(var) if var > 0 else 0.0


def _total_return(returns: list[float]) -> float:
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return total - 1.0


def _max_drawdown(returns: list[float]) -> float:
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
    return mdd


def _block_bootstrap_ci(diffs: list[float], n_boot: int, seed: int) -> dict:
    n = len(diffs)
    block = max(2, round(n ** (1.0 / 3.0)))
    n_blocks = math.ceil(n / block)
    max_start = max(0, n - block)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        vals: list[float] = []
        for _b in range(n_blocks):
            s = rng.randint(0, max_start)
            vals.extend(diffs[s:s + block])
        means.append(fmean(vals[:n]))
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    p_le_zero = sum(1 for m in means if m <= 0.0) / n_boot
    return {"mean_daily_diff": fmean(diffs), "ci95": [lo, hi],
            "p_diff_le_0": p_le_zero, "block_len": block, "n_boot": n_boot}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="holdout report JSONs (one per name)")
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--arm", action="append", default=None,
                        help="treatment arm(s) vs equal_weight (default: all non-equal arms)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default=None, help="optional output JSON path")
    args = parser.parse_args(argv)

    per_name = {Path(p).stem: _load_series(Path(p)) for p in args.reports}
    # A report that declares returns="net" already carries its engine's costs; charging
    # it again here was the double-charge path the fairness audit flagged. Arms declare
    # nothing and are charged, which is the original behaviour.
    for p in args.reports:
        basis = (json.loads(Path(p).read_text(encoding="utf-8")).get("manifest") or {}).get("returns")
        if basis == "net":
            raise SystemExit(
                f"{Path(p).name} declares returns='net'; this scorer charges every "
                "series and would double-charge it. Score it through baseline_table, "
                "which honours the declaration.")
    arms = sorted({a for s in per_name.values() for a in s})
    if BASELINE_ARM not in arms:
        raise SystemExit(f"baseline arm '{BASELINE_ARM}' missing from reports")
    treatment_arms = args.arm or [a for a in arms if a != BASELINE_ARM]
    missing = [(n, a) for n in per_name for a in ([BASELINE_ARM] + treatment_arms)
               if a not in per_name[n]]
    if missing:
        raise SystemExit(f"missing name-arm combinations, refusing to renormalise "
                         f"over a subset: {missing[:6]}")

    # Synchronized calendar: dates present for EVERY name and EVERY arm involved.
    involved = set(treatment_arms) | {BASELINE_ARM}
    date_sets = [set(series[a]) for series in per_name.values() for a in involved if a in series]
    dates = sorted(set.intersection(*date_sets)) if date_sets else []
    if len(dates) < 30:
        raise SystemExit(f"only {len(dates)} synchronized dates — refusing to report")

    # Equal-capital portfolio per arm: mean net return across names each date.
    portfolio: dict[str, list[float]] = {}
    for arm in involved:
        daily = []
        nets = {name: _net_returns(series[arm], dates, args.cost_bps_per_side, name)
                for name, series in per_name.items() if arm in series}
        for i in range(len(dates)):
            daily.append(fmean([nets[name][i] for name in nets]))
        portfolio[arm] = daily

    result = {
        "names": sorted(per_name), "n_dates": len(dates),
        "window": [dates[0], dates[-1]], "cost_bps_per_side": args.cost_bps_per_side,
        "arms": {}, "paired_vs_equal_weight": {},
    }
    for arm in sorted(involved):
        rets = portfolio[arm]
        result["arms"][arm] = {
            "ann_sharpe": _ann_sharpe(rets), "total_return": _total_return(rets),
            "max_drawdown": _max_drawdown(rets),
        }
    base = portfolio[BASELINE_ARM]
    for arm in treatment_arms:
        diffs = [portfolio[arm][i] - base[i] for i in range(len(dates))]
        result["paired_vs_equal_weight"][arm] = _block_bootstrap_ci(diffs, args.n_boot, SEED)

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
