"""The four reported metrics per stock, committee against verified global memory.

tab:terminal carries Sharpe alone, which says the memory helped but not how. The
benchmark paper reports SPR, CR, MDD and AV side by side per symbol, and those four
separate a gain that came from return from one that came from risk. Printed for every
stock; the fourteen whose two settings are one series are identical by construction and
are marked as such.

    TRADING_AGENT_DATA_ROOT=... PYTHONPATH=src python scripts/per_stock_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from baselines_table_full import (cumulative, max_drawdown, volatility)  # noqa: E402
from portfolio20 import resolve  # noqa: E402
from reprice_next_open import adjusted_prices, reprice  # noqa: E402
from verify_from_series import ARMS, net, priced, sharpe  # noqa: E402


def main() -> int:
    found, missing = resolve()
    if missing:
        print(f"missing runs: {missing}")
        return 1
    draw = sorted(found)
    opens, closes = adjusted_prices(draw)

    print(f"{'':6s} {'SPR equal':>10} {'SPR mem':>9} {'CR equal':>9} {'CR mem':>8} "
          f"{'AV equal':>9} {'AV mem':>8} {'MDD equal':>10} {'MDD mem':>9}")
    differ = []
    for t in draw:
        rep = json.loads(found[t].read_text(encoding="utf-8"))
        cal = sorted(opens[t])
        s = {a: net(priced(reprice(rep["daily_arm_series"][a], opens[t], closes[t], cal),
                           opens[t], cal))
             for a in ("equal_weight", "global_memory")}
        e, g = s["equal_weight"], s["global_memory"]
        same = all(abs(x - y) < 1e-12 for x, y in zip(e, g))
        row = (f"{t:6s} {sharpe(e):10.3f} {sharpe(g):9.3f} {cumulative(e):8.2f}% "
               f"{cumulative(g):7.2f}% {volatility(e):8.2f}% {volatility(g):7.2f}% "
               f"{max_drawdown(e):9.2f}% {max_drawdown(g):8.2f}%")
        print(row + ("   (one series)" if same else ""))
        if not same:
            differ.append((t, sharpe(e), sharpe(g), cumulative(e), cumulative(g),
                           volatility(e), volatility(g), max_drawdown(e),
                           max_drawdown(g)))

    print(f"\n{len(differ)} stocks where the two settings differ\n")
    print("  where the Sharpe gain came from:")
    for t, se, sg, ce, cg, ve, vg, me, mg in differ:
        parts = []
        parts.append(f"return {cg - ce:+.2f}pp")
        parts.append(f"volatility {vg - ve:+.2f}pp")
        parts.append(f"drawdown {mg - me:+.2f}pp")
        print(f"    {t:5s} SPR {sg - se:+.3f}  ({', '.join(parts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
