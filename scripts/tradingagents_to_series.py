"""Convert recorded TradingAgents decisions into scored daily series.

Fairness contract, after the 2026-07-31 audit:

  * one canonical calendar per ticker: the arm report's own scored dates, so the agent
    and the arms are compared on exactly the same sessions;
  * adjusted close, the same price basis every other path uses; the raw close counted
    dividends as losses for this agent only;
  * a failed or missing decision places no order, so the position carries across it
    (the day still earns the held position's return) instead of compressing several
    sessions into one pseudo-daily return;
  * each row is labelled by its decision date and earns the NEXT session's move,
    the convention every arm uses.

TradingAgents answers in a mixed vocabulary. Hold 71.4%, Overweight 17.6%,
Underweight 3.7%, Buy 1.3%, Sell 0.2% over the first recorded run. A parser that knows
only the three trade verbs drops a fifth of its decisions, so the rating words map:
Overweight/Buy -> LONG, Underweight/Sell -> FLAT, Hold -> keep the position.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TA = ROOT / "reports" / "baselines" / "tradingagents"
OUT = ROOT / "reports" / "baselines"

sys.path.insert(0, str(ROOT / "scripts"))
from portfolio20 import resolve  # noqa: E402

LONG_WORDS = ("buy", "overweight", "long")
FLAT_WORDS = ("sell", "underweight", "short", "reduce")
_LABEL_RE = re.compile(r"\b(" + "|".join(LONG_WORDS + FLAT_WORDS + ("hold",)) + r")\b")


def parse_decision(text: str) -> str | None:
    """The LAST whole-word label wins: the agent argues both sides, then calls it."""
    hits = _LABEL_RE.findall(str(text).strip().lower())
    if not hits:
        return None
    last = hits[-1]
    if last in LONG_WORDS:
        return "LONG"
    if last in FLAT_WORDS:
        return "FLAT"
    return "HOLD"


def load_adjusted_closes(tickers) -> dict[str, dict[str, float]]:
    """Adjusted closes from the FULL panel: the slim one lacks the thirteen new names,
    and an absent symbol would silently read as zero return on every day."""
    import pyarrow.dataset as ds

    from trading_agent.paths import finsaber_prices

    d = ds.dataset(str(finsaber_prices()), format="parquet", partitioning="hive")
    cols = [f.name for f in d.schema]
    price_col = next(c for c in ("adjusted_close", "adj_close", "close") if c in cols)
    tbl = d.to_table(columns=["date", "symbol", price_col]).to_pydict()
    out: dict[str, dict[str, float]] = {t: {} for t in tickers}
    for dt, sym, px in zip(tbl["date"], tbl["symbol"], tbl[price_col]):
        if sym in out and px is not None:
            out[sym][str(dt)[:10]] = float(px)
    return out, price_col


def main() -> int:
    found, missing = resolve()
    if missing:
        raise SystemExit(f"universe incomplete: {' '.join(missing)}")
    tickers = sorted(found)
    closes, price_col = load_adjusted_closes(tickers)

    written = []
    for ticker in tickers:
        src = TA / f"{ticker}.jsonl"
        if not src.exists():
            print(f"  {ticker:6s} no recording yet", flush=True)
            continue

        # Latest successful decision per date; failures carry no decision.
        decisions: dict[str, str | None] = {}
        n_fail = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = str(r.get("date", ""))[:10]
            if not d:
                continue
            if r.get("ok"):
                decisions[d] = parse_decision(r.get("decision", ""))
            else:
                n_fail += 1
                decisions.setdefault(d, None)

        # The canonical calendar is the arm report's own scored dates.
        arm = json.loads(found[ticker].read_text(encoding="utf-8"))
        dates = [row[0] for row in arm["daily_arm_series"]["equal_weight"]]

        px = closes[ticker]
        rows, n_dec = [], {"LONG": 0, "FLAT": 0, "HOLD": 0, "carried": 0}
        position = 0.0
        unparsed = 0
        for i, d in enumerate(dates):
            nxt = dates[i + 1] if i + 1 < len(dates) else None
            p0, p1 = px.get(d), px.get(nxt) if nxt else None
            call = decisions.get(d)
            if call is None:
                n_dec["carried"] += 1          # failed or absent: no order, carry
            elif call == "LONG":
                n_dec["LONG"] += 1
            elif call == "FLAT":
                n_dec["FLAT"] += 1
            elif call == "HOLD":
                n_dec["HOLD"] += 1
            else:
                unparsed += 1
            # Row d: the position DECIDED on d earns the next session's move.
            new_position = (1.0 if call == "LONG" else 0.0 if call == "FLAT"
                            else position)
            rows.append([d, new_position * ((p1 / p0 - 1.0) if (p0 and p1) else 0.0),
                         new_position])
            position = new_position

        path = OUT / f"{ticker}__TradingAgents.json"
        path.write_text(json.dumps({
            "mode": "baseline", "run_label": f"{ticker}__TradingAgents",
            "daily_arm_series": {"equal_weight": rows},
            "manifest": {"strategy": "TradingAgents", "ticker": ticker,
                         "model": "gpt-4.1-mini",
                         "returns": "gross",
                         "decisions": n_dec, "n_scored": len(rows),
                         "unparsed": unparsed, "failed_calls_carried": n_fail,
                         "price_source": f"FINSABER panel {price_col} (same as the arms)",
                         "calendar": "the arm report's scored dates",
                         "policy": "a failed call places no order; the position carries"},
        }, indent=1), encoding="utf-8")
        written.append(ticker)
        print(f"  {ticker:6s} {len(rows)} rows on the arm calendar  "
              f"LONG {n_dec['LONG']:>3}  FLAT {n_dec['FLAT']:>3}  HOLD {n_dec['HOLD']:>3}  "
              f"carried {n_dec['carried']:>3}  unparsed {unparsed}", flush=True)
    print(f"\nwrote {len(written)} of {len(tickers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
