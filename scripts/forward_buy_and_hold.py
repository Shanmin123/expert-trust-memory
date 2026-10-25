"""Buy and hold over the forward window, from the bars the committee read already saved.

The forward run needs a comparison that costs nothing and touches nothing. Buy and hold
is the one the sealed window already reports (Table tab:baselines, +0.800 against the
committee's +0.624), so tracking it live keeps the same reference in both places.

It makes NO model call, opens NO IB connection of its own, and places NO order. It reads
the parquet bars that ``live_analysis.py --save-bars`` writes each day, so it adds
nothing to IB's historical-request pacing budget.

Convention, matched to the sealed artifacts (``reports/baselines/*__BuyAndHoldStrategy``):
bought at the first open of the window and held, valued at the close, ten basis points
charged once on entry, exposure 1.0 throughout. The emitted shape is the standard
``daily_arm_series`` of ``[date, return, exposure]`` with ``returns: net``, so
``scripts/baseline_table.py`` and ``scripts/portfolio_arm_metrics.py`` score it through
the same code path as every other row.

    PYTHONPATH=src python scripts/forward_buy_and_hold.py
    PYTHONPATH=src python scripts/forward_buy_and_hold.py --bars reports/paper_forward/bars
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARS = ROOT / "reports" / "paper_forward" / "bars"
OUT = ROOT / "reports" / "paper_forward" / "buy_and_hold"
COST_BPS = 10.0
# The draw, as registered. TSLA and the proxy are not part of it.
DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")


def load_bars(bars_dir: Path, symbol: str):
    """Return [(date, open, close)] for one symbol, oldest first."""
    import pandas as pd

    path = bars_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    rows = []
    for _, r in frame.iterrows():
        date = str(r["date"])[:10]
        close = float(r.get("adjusted_close", r["close"]))
        # The saved bars carry raw open with adjusted close; scale the open by the same
        # factor so both legs of a return sit on one basis.
        raw_close = float(r["close"])
        scale = close / raw_close if raw_close else 1.0
        rows.append((date, float(r["open"]) * scale, close))
    return sorted(rows)


def series_for(rows, start: str) -> list:
    """Entry at the open of the first session on or after ``start``, then hold."""
    window = [r for r in rows if r[0] >= start]
    if len(window) < 1:
        return []
    entry_open = window[0][1]
    if not entry_open:
        return []
    out = []
    prev_close = None
    for i, (date, _open_px, close) in enumerate(window):
        if i == 0:
            gross = close / entry_open - 1.0
            net = gross - COST_BPS / 1e4          # one side, charged on entry
        else:
            net = close / prev_close - 1.0
        out.append([date, net, 1.0])
        prev_close = close
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=str(DEFAULT_BARS),
                    help="directory written by live_analysis.py --save-bars")
    ap.add_argument("--start", default="2026-08-03",
                    help="first session of the forward window")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)

    bars_dir = Path(args.bars)
    if not bars_dir.exists():
        raise SystemExit(
            f"{bars_dir} not found. Run scripts/live_analysis.py --save-bars first; "
            "this script never fetches bars of its own."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, missing, lengths = [], [], set()
    for symbol in DRAW:
        rows = load_bars(bars_dir, symbol)
        if rows is None:
            missing.append(symbol)
            continue
        series = series_for(rows, args.start)
        if not series:
            missing.append(symbol)
            continue
        lengths.add(len(series))
        payload = {
            "mode": "baseline",
            "run_label": f"{symbol}__ForwardBuyAndHold",
            "daily_arm_series": {"equal_weight": series},
            "manifest": {
                "strategy": "BuyAndHoldStrategy",
                "ticker": symbol,
                "window": [series[0][0], series[-1][0]],
                "cost_bps_per_side": COST_BPS,
                "returns": "net",
                "source": "forward run, bars saved by live_analysis.py",
                "note": "bought at the first open of the forward window and held; "
                        "one side of cost charged on entry",
            },
        }
        path = out_dir / f"{symbol}__ForwardBuyAndHold.json"
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        written.append(symbol)

    print(f"wrote {len(written)} of {len(DRAW)} to {out_dir}")
    if missing:
        print(f"no bars yet for {missing}")
    if len(lengths) > 1:
        print(f"WARNING: series lengths differ {sorted(lengths)}; a stock is missing a "
              "session and the equal-capital portfolio would misalign")
    elif lengths:
        n = lengths.pop()
        print(f"{n} session(s) so far, {series[0][0]} to {series[-1][0]}")
        print("\nscore it beside the arms with:")
        print(f"  PYTHONPATH=src python scripts/portfolio_arm_metrics.py "
              f"{out_dir}/*__ForwardBuyAndHold.json")
    return 0 if written and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
