"""Buy and hold on the arms' own accounting, written as report-shaped artifacts.

The thesis quotes a buy-and-hold row in tab:baselines, and until now nothing on disk
produced it. `baseline_arms.py` routes buy and hold through FINSABER's engine, which
sizes all-in from the decision close and cannot fill through a gap; that delayed eleven
of the twenty entries and scores the row at +0.700. Commit 86205f9 replaced the row with
a first-open rebuild but never persisted the series, so the quoted +0.800 matched no
artifact and `verify_from_series.py` had no buy-and-hold entry to check.

This closes that. Each stock is bought at the first open of the test window and held to
the last close, on the same adjusted panel the arms are repriced against, charged the
same ten basis points on the one exposure change. Holding to the end never exits, so no
exit charge is applied, which is what the arms' cost model does too.

The portfolio is assembled by the caller (verify_from_series, baseline_table) exactly as
it is for every other row: the mean of the per-stock net daily returns, equal capital
rebalanced daily. That aggregation is what reproduces the equal-weight arm's +0.6244, so
using anything else here would compare buy and hold against the arms on two different
portfolio constructions.

    TRADING_AGENT_DATA_ROOT=... PYTHONPATH=src python scripts/buy_and_hold_nextopen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reprice_next_open import adjusted_prices  # noqa: E402

# FINSABER's own schedule, which is what every other row in the baselines table is
# charged: $0.0049 a share, at least $0.99 an order, at most 1% of the trade. One
# entry fills the whole account, so the charge as a fraction of it is
# min(max(per_share / price, minimum / cash), cap).
PER_SHARE, MIN_COMMISSION, MAX_RATE, CASH = 0.0049, 0.99, 0.01, 100_000.0
DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")
OUT = ROOT / "reports" / "baselines"


def main() -> int:
    calendar = [r[0] for r in json.loads(
        (ROOT / "reports" / "reprice" / "ADSK__reprice.json").read_text(encoding="utf-8")
    )["daily_arm_series"]["equal_weight"]]
    opens, closes = adjusted_prices(list(DRAW))

    for ticker in DRAW:
        rows = []
        for i, date in enumerate(calendar):
            if i == 0:
                # The one decision: fill at the first open, charged once.
                px = opens[ticker][date]
                fee = min(max(PER_SHARE / px, MIN_COMMISSION / CASH), MAX_RATE)
                ret = closes[ticker][date] / px - 1.0 - fee
            else:
                ret = closes[ticker][date] / closes[ticker][calendar[i - 1]] - 1.0
            rows.append([date, ret, 1.0])
        report = {
            "mode": "baseline",
            "run_label": f"{ticker}__BuyAndHoldNextOpen",
            "daily_arm_series": {"equal_weight": rows},
            "manifest": {
                "strategy": "BuyAndHoldNextOpen",
                "ticker": ticker,
                "window": [calendar[0], calendar[-1]],
                "cost_model": "commission",
                "commission": {"commission_per_share": PER_SHARE,
                               "min_commission": MIN_COMMISSION,
                               "max_commission_rate": MAX_RATE},
                "returns": "net",
                "source": "adjusted panel via reprice_next_open.adjusted_prices",
                "note": "bought at the first open of the window and held to the last "
                        "close; one entry charge, no exit",
            },
        }
        path = OUT / f"{ticker}__BuyAndHoldNextOpen.json"
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {path.name}  n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
