"""Reprice the recorded decisions at the execution the system actually uses.

The contract executes at the next open (``contracts.py``: execution_timing =
"next_open"), but the reported daily series was built from the reporting attribution,
whose realised leg is adjusted_close(t) -> adjusted_close(t+1) (``runner.py``). A
position entered at the open of t+1 never earns the overnight gap from the close of t,
so the published series credits a move the executed system could not capture. The
2026-08-02 cross-model audit found this; it applies to every arm equally, so the
paired comparison stays internally consistent, but the levels and the p-value do not.

This script does not re-run anything. The decisions are already recorded as the
exposure column of ``daily_arm_series``, so each date's return is recomputed as the
account's own profit and loss: a session earns close/open when the position is
entered, close/close while it is held, open/close when it is exited, and nothing
while flat. That is the convention FINSABER's engine produces for the rule
baselines, so the arms and the baselines end on one accounting.

KNOWN LIMIT: ``daily_arm_series`` holds 352 rows because it is built from matured
attribution episodes, while each arm actually decided on 373 test sessions. The last
twenty-one decisions are therefore outside this repricing. Both arms are truncated
identically, so the paired comparison stays valid, but the window is shorter than the
test period. Closing it needs the applied exposure serialised during the decision
loop, or a free replay of the frozen memory over the frozen corpus.

The learning signal is untouched. g = v * r over 21 days measures an expert's
directional edge, not tradable profit, and the verified cells are unchanged, so the
frozen memory and the certification results stand exactly as sealed.

    PYTHONPATH=src python scripts/reprice_next_open.py
    PYTHONPATH=src python scripts/reprice_next_open.py --out-dir reports/reprice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ARMS = ("equal_weight", "global_memory", "market_state_memory",
        "market_state_memory_online")


def adjusted_prices(tickers) -> tuple[dict, dict]:
    """date -> adjusted open, and date -> adjusted close, per ticker."""
    import pyarrow.dataset as ds

    from trading_agent.paths import finsaber_prices

    d = ds.dataset(str(finsaber_prices()), format="parquet", partitioning="hive")
    cols = {f.name for f in d.schema}
    need = ["date", "symbol", "open", "close"]
    px = "adjusted_close" if "adjusted_close" in cols else "adj_close"
    tbl = d.to_table(columns=need + [px],
                     filter=(ds.field("symbol").isin(list(tickers)))).to_pydict()
    o: dict[str, dict[str, float]] = {t: {} for t in tickers}
    c: dict[str, dict[str, float]] = {t: {} for t in tickers}
    for dt, sym, op, cl, ac in zip(tbl["date"], tbl["symbol"], tbl["open"],
                                   tbl["close"], tbl[px]):
        if sym not in o or op is None or cl in (None, 0) or ac is None:
            continue
        # The panel stores raw open with adjusted close; scale the open by the same
        # split/dividend factor so the two legs of a return are on one basis.
        day = str(dt)[:10]
        o[sym][day] = float(op) * float(ac) / float(cl)
        c[sym][day] = float(ac)
    return o, c


def reprice(rows: list, opens: dict[str, float], closes: dict[str, float],
            calendar: list[str]) -> list:
    """[date, gross, exposure] on the account's own profit and loss.

    A decision at the close of t is filled at the open of t+1, so the position held
    through session t+1 is the decision made at t, and the equity mark is the close.
    That makes the return of session t+1 depend on what changed:

      entering  (flat, then long)  close(t+1) / open(t+1)  - 1
      holding   (long, then long)  close(t+1) / close(t)   - 1
      exiting   (long, then flat)  open(t+1)  / close(t)   - 1
      flat                         zero

    This is the convention FINSABER's engine produces for the rule baselines, so the
    arms and the baselines are finally on one accounting. The row keeps the decision
    date so the exposure column still reads as the decision, which is what the cost
    model charges on.
    """
    idx = {d: i for i, d in enumerate(calendar)}
    out, prev = [], 0.0
    for date, _gross, exposure in rows:
        i = idx.get(date)
        nxt = calendar[i + 1] if i is not None and i + 1 < len(calendar) else None
        o1, c1 = (opens.get(nxt), closes.get(nxt)) if nxt else (None, None)
        c0 = closes.get(date)
        e = float(exposure)
        if not (o1 and c1 and c0):
            r = 0.0
        elif prev == 0.0 and e == 1.0:
            r = c1 / o1 - 1.0
        elif prev == 1.0 and e == 1.0:
            r = c1 / c0 - 1.0
        elif prev == 1.0 and e == 0.0:
            r = o1 / c0 - 1.0
        else:
            r = 0.0
        out.append([date, r, e])
        prev = e
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "reprice"))
    args = ap.parse_args(argv)

    from portfolio20 import resolve
    found, missing = resolve()
    if missing:
        raise SystemExit(f"universe incomplete: {' '.join(missing)}")

    tickers = sorted(found)
    opens, closes = adjusted_prices(tickers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'stock':6}{'dates':>7}{'priced':>8}   arms repriced")
    for ticker in tickers:
        report = json.loads(found[ticker].read_text(encoding="utf-8"))
        series = report["daily_arm_series"]
        # The panel's own trading calendar, so t+1 and t+2 are real sessions.
        calendar = sorted(opens[ticker])
        priced = 0
        for arm in ARMS:
            if arm not in series:
                continue
            series[arm] = reprice(series[arm], opens[ticker], closes[ticker], calendar)
            priced += 1
        n = len(series["equal_weight"])
        nz = sum(1 for r in series["equal_weight"] if r[1] != 0.0)
        report["manifest"]["repriced"] = ("next_open account P&L: filled at the open "
                                          "after the decision, marked at the close")
        (out_dir / f"{ticker}__reprice.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
        print(f"{ticker:6}{n:>7}{nz:>8}   {priced}")

    print(f"\nwritten to {out_dir}")
    print("score with: PYTHONPATH=src python scripts/portfolio_arm_metrics.py "
          f"{out_dir}/*__reprice.json --n-boot 40000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
