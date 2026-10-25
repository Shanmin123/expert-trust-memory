"""Is the portfolio p-value an artifact of the block length? Test it at 7, 21 and 42.

``portfolio_arm_metrics.py`` reports a plain percentile moving-block bootstrap whose
block is fixed at ``round(n**(1/3))`` = 7 days. Attribution runs at a 21-day horizon, so
dependence in the daily difference series plausibly outlives a seven-day block, and a
reviewer is entitled to ask whether p = .082 survives a longer one.

The statistic here is the centred, studentised one this project already uses for cell
verification (``memory/expert_reflection.py``): ``t_b = (mean_b - mean_0) / se_b``
compared against ``t_0 = mean_0 / se_0``, with plus-one p-values. Reusing that convention
keeps one definition of "the bootstrap" across the thesis instead of two.

Blocks are 7 (the cube-root default), 21 (the attribution horizon) and 42 (twice it, the
length cell verification uses).

    PYTHONPATH=src python scripts/portfolio_block_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"
REPRICE = ROOT / "reports" / "reprice"
# FINSABER's own schedule, the one every row in the baselines table is charged.
# The commission is a per-share amount, so it needs the price each trade fills at,
# which under this execution is the next session's open.
PER_SHARE, MIN_COMMISSION, MAX_RATE, CASH = 0.0049, 0.99, 0.01, 100_000.0
SEED = 20260716
BLOCKS = (7, 21, 42)
DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")

SEALED = {"TSLA": "TSLA__designD3_q_h21cert", "ADSK": "ADSK__designD3_q_h21cert",
          "BIIB": "BIIB__designD3_q_h21cert", "CVX": "CVX__final", "DHI": "DHI__sealed",
          "ECL": "ECL__sealed", "PNC": "PNC__sealed", "TGT": "TGT__sealed"}


_PRICES: dict = {}


def _opens(ticker: str):
    """The panel's adjusted opens, loaded once per ticker."""
    if ticker not in _PRICES:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from reprice_next_open import adjusted_prices
        o, _c = adjusted_prices([ticker])
        _PRICES[ticker] = (o[ticker], sorted(o[ticker]))
    return _PRICES[ticker]


def net_series(rows, ticker):
    """Charge the commission on exposure changes, exactly as the scorer does."""
    opens, cal = _opens(ticker)
    idx = {d: i for i, d in enumerate(cal)}
    out, prev, equity = {}, 0.0, CASH
    for d, gross, exp in rows:
        i = idx.get(d)
        nxt = cal[i + 1] if i is not None and i + 1 < len(cal) else None
        px = opens.get(nxt) if nxt else None
        traded = abs(float(exp) - prev)
        frac = (min(max(PER_SHARE / px, MIN_COMMISSION / equity), MAX_RATE)
                if (traded and px) else 0.0)
        r = float(gross) - frac * traded
        out[d] = r
        equity *= (1.0 + r)
        prev = float(exp)
    return out


def report_paths(universe20: bool, reprice: bool = False) -> dict[str, Path]:
    """Name -> report path: the evaluated eight, or the twenty-name random draw.

    ``reprice`` reads the next-open series instead of the sealed close-to-close one.
    The primary p = .107 is computed on the repriced series, so a block sensitivity run
    off the sealed reports answers a question about a series the thesis no longer
    reports, and its p-values are not comparable with the primary.
    """
    if reprice:
        return {t: REPRICE / f"{t}__reprice.json" for t in DRAW}
    if not universe20:
        return {t: FINAL / f"{stem}.json" for t, stem in SEALED.items()}
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from portfolio20 import resolve
    found, missing = resolve()
    if missing:
        raise SystemExit(f"universe20 incomplete, missing: {' '.join(missing)}")
    return found


def portfolio_diffs(arm: str, universe20: bool = False,
                    reprice: bool = False) -> list[float]:
    """Equal-capital daily difference, arm minus equal weight, on shared dates."""
    per_arm, per_eq = {}, {}
    for ticker, path in report_paths(universe20, reprice).items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_arm[ticker] = net_series(payload["daily_arm_series"][arm], ticker)
        per_eq[ticker] = net_series(payload["daily_arm_series"]["equal_weight"], ticker)
    dates = sorted(set.intersection(*(set(v) for v in per_arm.values())))
    return [fmean(per_arm[t][d] for t in per_arm) - fmean(per_eq[t][d] for t in per_eq)
            for d in dates]


def t_stat(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    mean = fmean(vals)
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, (mean / se if se > 0 else 0.0)


def studentised_p(diffs: list[float], block: int, n_boot: int, seed: int) -> dict:
    """Centred, studentised moving-block bootstrap, the convention used for cells."""
    n = len(diffs)
    mean0, se0, t0 = t_stat(diffs)
    n_blocks = math.ceil(n / block)
    max_start = max(0, n - block)
    rng = random.Random(seed)

    ge_one_sided = ge_two_sided = 0
    for _ in range(n_boot):
        vals: list[float] = []
        for _b in range(n_blocks):
            s = rng.randint(0, max_start)
            vals.extend(diffs[s:s + block])
        vals = vals[:n]
        mean_b, se_b, _ = t_stat(vals)
        if se_b <= 0:
            ge_one_sided += 1
            ge_two_sided += 1
            continue
        t_b = (mean_b - mean0) / se_b        # centred + studentised
        if t_b >= t0:
            ge_one_sided += 1
        if abs(t_b) >= abs(t0):
            ge_two_sided += 1

    return {
        "block": block,
        "mean_daily_diff": mean0,
        "t": t0,
        "p_one_sided": (ge_one_sided + 1) / (n_boot + 1),
        "p_two_sided": (ge_two_sided + 1) / (n_boot + 1),
        "n_boot": n_boot,
        "n_days": n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="global_memory")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--universe20", action="store_true",
                    help="pool the twenty-name random draw instead of the evaluated eight")
    ap.add_argument("--reprice", action="store_true",
                    help="read the twenty-name next-open series the primary p is "
                         "computed on, rather than the sealed close-to-close reports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    diffs = portfolio_diffs(args.arm, universe20=args.universe20, reprice=args.reprice)
    rows = [studentised_p(diffs, b, args.n_boot, SEED) for b in BLOCKS]

    print(f"{args.arm} minus equal_weight, {rows[0]['n_days']} days, "
          f"mean {rows[0]['mean_daily_diff']:.3e}, t = {rows[0]['t']:.3f}")
    print(f"\n{'block':>6}{'one-sided p':>14}{'two-sided p':>14}")
    for r in rows:
        print(f"{r['block']:>6}{r['p_one_sided']:>14.3f}{r['p_two_sided']:>14.3f}")
    lo1, hi1 = min(r["p_one_sided"] for r in rows), max(r["p_one_sided"] for r in rows)
    lo2, hi2 = min(r["p_two_sided"] for r in rows), max(r["p_two_sided"] for r in rows)
    print(f"\none-sided range {lo1:.3f} to {hi1:.3f};  two-sided {lo2:.3f} to {hi2:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"arm": args.arm, "statistic": "centred+studentised moving-block bootstrap",
             "seed": SEED, "results": rows}, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
