"""Re-price the FinMem baseline under FINSABER's own commission schedule.

The baseline was run with the per-share schedule saturated into a flat ten basis points
a side, and its artifact records the quantity, fill price and charge of every trade. So
moving it onto the schedule the benchmark actually charges is arithmetic on that log and
needs no API call: refund what was charged, levy min(max(q*0.0049, 0.99), 0.01*q*price),
and carry the difference into the day the trade executed.

    PYTHONPATH=src python scripts/finmem_reprice_commission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baselines"
PER_SHARE, MIN_COMMISSION, MAX_RATE, CASH = 0.0049, 0.99, 0.01, 100_000.0
BPS_CHARGED = 10.0 / 1e4
DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")


def native(qty: float, price: float) -> float:
    return min(max(abs(qty) * PER_SHARE, MIN_COMMISSION), MAX_RATE * abs(qty * price))


def main() -> int:
    refunded, moved = 0.0, 0
    for t in DRAW:
        path = OUT / f"{t}__FinMem.json"
        if not path.exists():
            print(f"  {t:5s} no artifact")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        m = d["manifest"]
        if m.get("cost_model") == "commission":
            print(f"  {t:5s} already on the commission schedule")
            continue
        rows = d["daily_arm_series"]["equal_weight"]
        delta: dict[str, float] = {}
        for tr in d.get("trades", []):
            q, px, charged = tr["quantity"], tr["price"], tr["commission"]
            # The recorded charge must be the flat rate it claims, or the refund is wrong.
            want = BPS_CHARGED * abs(q * px)
            if abs(charged - want) > max(0.02, 1e-6 * abs(q * px)):
                raise SystemExit(
                    f"{t} {tr['execution_date']}: recorded {charged:.2f} is not "
                    f"{BPS_CHARGED * 1e4:.0f} bps of {abs(q * px):.0f}")
            delta[tr["execution_date"]] = (delta.get(tr["execution_date"], 0.0)
                                           + charged - native(q, px))
            refunded += charged - native(q, px)
            moved += 1
        equity, out = CASH, []
        for date, r, e in rows:
            adj = delta.get(date, 0.0) / equity if equity else 0.0
            out.append([date, r + adj, e])
            equity *= (1.0 + r + adj)
        d["daily_arm_series"]["equal_weight"] = out
        m["cost_model"] = "commission"
        m["commission"] = {"commission_per_share": PER_SHARE,
                           "min_commission": MIN_COMMISSION,
                           "max_commission_rate": MAX_RATE}
        m.pop("commission_model", None)
        m["note"] = ((m.get("note", "") + "; ") if m.get("note") else "") + \
            "re-priced from the stored trade log onto FINSABER's own schedule"
        path.write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f"  {t:5s} {len(delta)} sessions adjusted")
    print(f"{moved} trades re-priced, ${refunded:,.0f} of commission refunded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
