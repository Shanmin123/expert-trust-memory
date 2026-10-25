"""Rebuild tab:baselines with the full metric set, grouped by family.

FINSABER reports return, risk and risk-adjusted metrics side by side (AR, CR, AV, MDD,
SPR, STR) and groups strategies by family. This prints the same shape for this thesis's
rows, computed here from each strategy's daily net series so every row is on one basis:
no risk-free deduction anywhere, including the rows whose returns came from FINSABER's
engine, which applies its own rate internally.

    TRADING_AGENT_DATA_ROOT=... PYTHONPATH=src python scripts/baselines_table_full.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from portfolio20 import resolve  # noqa: E402
from reprice_next_open import adjusted_prices, reprice  # noqa: E402
from verify_from_series import ARMS, net, priced, sharpe  # noqa: E402

FAMILIES = [
    ("Timing rules", [
        ("Turn of the month", "TurnOfTheMonthStrategy"),
        ("WMA crossover", "WMAStrategy"),
        ("SMA crossover", "SMACrossStrategy"),
        ("Trend following", "TrendFollowingStrategy"),
        ("Bollinger bands", "BollingerBandsStrategy"),
        ("ATR band", "ATRBandStrategy"),
    ]),
    ("Factor models", [
        ("Alpha158, ridge", "AlphaMining_ridgepx"),
        ("Alpha158, boosted trees", "AlphaMining_gbtpx"),
    ]),
    ("LLM agent", [("FinMem", "FinMem")]),
    ("Passive", [("Buy and hold", "BuyAndHoldNextOpen")]),
]


def annualised_return(rets: list[float]) -> float:
    total = 1.0
    for r in rets:
        total *= 1.0 + r
    years = len(rets) / 252.0
    return (total ** (1.0 / years) - 1.0) * 100 if years > 0 and total > 0 else float("nan")


def cumulative(rets: list[float]) -> float:
    total = 1.0
    for r in rets:
        total *= 1.0 + r
    return (total - 1.0) * 100


def volatility(rets: list[float]) -> float:
    n = len(rets)
    m = sum(rets) / n
    v = sum((x - m) ** 2 for x in rets) / (n - 1)
    return math.sqrt(v) * math.sqrt(252) * 100


def max_drawdown(rets: list[float]) -> float:
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in rets:
        equity *= 1.0 + r
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return -worst * 100


def sortino(rets: list[float]) -> float:
    n = len(rets)
    m = sum(rets) / n
    downside = [x for x in rets if x < 0.0]
    if not downside:
        return float("inf")
    dd = math.sqrt(sum(x * x for x in downside) / n)
    return m / dd * math.sqrt(252) if dd > 0 else float("nan")


def main() -> int:
    found, missing = resolve()
    if missing:
        print(f"missing runs: {missing}")
        return 1
    draw = sorted(found)
    opens, closes = adjusted_prices(draw)

    series = {}
    for t in draw:
        rep = json.loads(found[t].read_text(encoding="utf-8"))
        cal = sorted(opens[t])
        series[t] = {a: priced(reprice(rows, opens[t], closes[t], cal), opens[t], cal)
                     for a, rows in rep["daily_arm_series"].items() if a in ARMS}
    calendar = [r[0] for r in series[draw[0]]["equal_weight"]]

    def portfolio(per_stock: list[dict]) -> list[float]:
        return [sum(m[d] for m in per_stock) / len(per_stock) for d in calendar]

    rows: list[tuple[str, list[float] | None]] = []
    base = ROOT / "reports" / "baselines"
    for family, members in FAMILIES:
        rows.append((f"@{family}", None))
        for label, stem in members:
            per_stock = []
            for t in draw:
                path = base / f"{t}__{stem}.json"
                if not path.exists():
                    per_stock = None
                    break
                d = json.loads(path.read_text(encoding="utf-8"))
                r = d["daily_arm_series"]["equal_weight"]
                declared = d["manifest"]["returns"]
                vals = ([x[1] for x in r] if declared == "net"
                        else net(priced(r, opens[t], sorted(opens[t]))))
                per_stock.append(dict(zip([x[0] for x in r], vals)))
            rows.append((label, portfolio(per_stock) if per_stock else None))

    rows.append(("@This work", None))
    for label, arm in (("Equal-weight committee", "equal_weight"),
                       ("Verified state memory", "market_state_memory"),
                       ("Verified global memory", "global_memory")):
        per_stock = [dict(zip([r[0] for r in series[t][arm]], net(series[t][arm])))
                     for t in draw]
        rows.append((label, portfolio(per_stock)))

    print(f"{'Strategy':26s} {'SPR':>7} {'STR':>7} {'AR%':>7} {'CR%':>7} "
          f"{'AV%':>7} {'MDD%':>7}")
    for label, vals in rows:
        if label.startswith("@"):
            print(f"\n-- {label[1:]}")
            continue
        if vals is None:
            print(f"{label:26s}   (artifact missing)")
            continue
        print(f"{label:26s} {sharpe(vals):+7.3f} {sortino(vals):+7.3f} "
              f"{annualised_return(vals):+7.2f} {cumulative(vals):+7.2f} "
              f"{volatility(vals):7.2f} {max_drawdown(vals):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
