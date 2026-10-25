"""Collect point-in-time structured financials from SEC EDGAR XBRL (companyfacts API).

Free, no API key. Each XBRL fact carries a ``filed`` date (when it became public), so a
downstream reader selects the latest financials KNOWN as of a decision date t
(``filed <= t``) -- strictly point-in-time, the discipline this dissertation studies.
Never key on the period-end date: a Q3 figure is not public until the 10-Q is filed
(~40 days later).

TradingAgents pulls balance sheet / income / cash flow via paid live APIs; this obtains
the SAME line items free and PIT-clean from SEC's own XBRL facts. SEC EDGAR is the free,
compliant path.

Run on a machine WITH network. CIKs are read from the local FINSABER filing parquet
(symbol -> cik), so no ticker->CIK lookup is needed.

    python scripts/collect_sec_xbrl.py \
        --names TSLA,NFLX,AMZN,MSFT,ADSK,BIIB,C,CVX \
        --user-agent "Your Name your@email.com" \
        --out data/sec_xbrl/financials.parquet

SEC requires a descriptive User-Agent with real contact info and rate-limits to ~10 req/s.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Canonical metric -> ordered list of us-gaap (or dei) tags; the first tag present wins.
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "RevenuesNetOfInterestExpense"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss",
                   # Retailers (TGT 2013-2022) tag only the available-to-common
                   # variant for long stretches (2026-07-19 sealed preflight catch).
                   "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],   # if absent, feature builder derives assets - equity
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements"],
    "shares": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding",
               "WeightedAverageNumberOfDilutedSharesOutstanding"],
}
# Money metrics report in USD; share counts in 'shares'.
_UNIT_FOR = {m: ("shares" if m == "shares" else "USD") for m in CONCEPTS}
_OUT_COLUMNS = ["symbol", "cik", "metric", "tag", "unit",
                "period_start", "period_end", "filed", "form", "fy", "fp", "val"]


def extract_facts(companyfacts: dict, symbol: str, cik: str,
                  concepts: dict[str, list[str]] = CONCEPTS) -> list[dict]:
    """Flatten a SEC companyfacts JSON into PIT rows -- pure, offline-testable.

    For each canonical metric, pick the FIRST alternate tag present under us-gaap/dei and
    emit every reported instance (keeping restatements: they differ by ``filed``). The
    downstream reader keeps the row with the greatest ``filed <= t``.
    """
    facts = companyfacts.get("facts", {}) or {}
    pools = {}
    pools.update(facts.get("us-gaap", {}) or {})
    pools.update(facts.get("dei", {}) or {})
    rows: list[dict] = []
    for metric, tags in concepts.items():
        want_unit = _UNIT_FOR[metric]
        # UNION across ALL alternate tags (fix 2026-07-18): companies drift between
        # tags over the years (e.g. pre/post ASC 606 revenue), so first-tag-wins for
        # the WHOLE history silently drops every year the first tag does not cover —
        # the TSLA fy2012-2022 revenue hole that starved the fundamental expert.
        # Duplicates of the same reported period (same fy/fp/period_end/filed) keep
        # the highest-priority tag's instance; restatements (different filed) all
        # survive for the downstream latest-filed<=t selection.
        chosen: dict[tuple, dict] = {}
        for tag in tags:                      # priority order: earlier tag wins ties
            if tag not in pools:
                continue
            units = (pools[tag].get("units") or {})
            # exact unit match, else the first unit whose key starts with it (e.g. USD/shares)
            unit_key = want_unit if want_unit in units else next(
                (u for u in units if str(u).startswith(want_unit)), None)
            if unit_key is None:
                continue
            for inst in units[unit_key]:
                filed = inst.get("filed")
                val = inst.get("val")
                if filed is None or val is None:
                    continue
                key = (inst.get("fy"), inst.get("fp"), inst.get("end"), filed)
                if key in chosen:
                    continue
                chosen[key] = {
                    "symbol": symbol, "cik": cik, "metric": metric, "tag": tag,
                    "unit": unit_key, "period_start": inst.get("start"),
                    "period_end": inst.get("end"), "filed": filed,
                    "form": inst.get("form"), "fy": inst.get("fy"), "fp": inst.get("fp"),
                    "val": float(val),
                }
        rows.extend(chosen.values())
    return rows


def resolve_ciks(names, filing_root: str) -> dict[str, str]:
    """{symbol: zero-padded-10-digit CIK} read from the local FINSABER filing parquet."""
    want = {n.upper() for n in names} if names else None
    out: dict[str, str] = {}
    for folder in ("filingk", "filingq"):
        for f in glob.glob(f"{filing_root}/{folder}/**/*.parquet", recursive=True):
            df = pd.read_parquet(f, columns=["symbol", "cik"])
            for sym, cik in df.drop_duplicates("symbol").itertuples(index=False):
                sym = str(sym).upper()
                if (want is None or sym in want) and sym not in out and cik is not None:
                    out[sym] = str(cik).zfill(10)
    return out


def fetch_companyfacts(cik: str, user_agent: str, *, retries: int = 4, sleep: float = 0.2) -> dict | None:
    """GET data.sec.gov companyfacts for a 10-digit CIK. SEC requires a real User-Agent."""
    import requests

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None                       # no XBRL facts for this CIK
            if resp.status_code in (429, 503):    # throttled -> back off
                time.sleep(1.0 + attempt * 1.5)
                continue
            return None
        except Exception:
            time.sleep(1.0 + attempt)
        finally:
            time.sleep(sleep)                     # stay well under SEC's 10 req/s
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", default="TSLA,NFLX,AMZN,MSFT,ADSK,BIIB,C,CVX",
                    help="Comma-separated tickers, or 'all' for every name in the filing data.")
    ap.add_argument("--filing-root", default="data/finsaber/sp500_2000_2025_parquet")
    ap.add_argument("--out", default="data/sec_xbrl/financials.parquet")
    ap.add_argument("--raw-cache", default="data/sec_xbrl/raw",
                    help="Directory to cache raw companyfacts JSON (skips re-download).")
    # SEC blocks requests without a contact, so this cannot be blank -- but it must not be
    # a hardcoded person either, or it ships one author's email to everyone who clones the
    # repository and attributes their traffic to that address.
    ap.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                    help="SEC requires 'Your Name your@email'. Set SEC_USER_AGENT or pass "
                         "this flag; requests without a contact are blocked by EDGAR.")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args(argv)

    if not args.user_agent:
        raise SystemExit(
            "SEC EDGAR blocks requests without a contact. Set SEC_USER_AGENT to "
            "'Your Name your@email' or pass --user-agent. It identifies your traffic to "
            "the SEC, so it has to be yours rather than a value shipped in the repository."
        )

    names = None if args.names.strip().lower() == "all" else \
        [n.strip() for n in args.names.split(",") if n.strip()]
    ciks = resolve_ciks(names, args.filing_root)
    if not ciks:
        print("No CIKs resolved from the filing data; check --filing-root / --names.")
        return 1
    print(f"Resolved {len(ciks)} CIKs. Fetching SEC companyfacts...")

    raw_dir = Path(args.raw_cache)
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for i, (sym, cik) in enumerate(sorted(ciks.items()), 1):
        cache = raw_dir / f"CIK{cik}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = fetch_companyfacts(cik, args.user_agent, sleep=args.sleep)
            if data is not None:
                cache.write_text(json.dumps(data), encoding="utf-8")
        if data is None:
            print(f"  [{i}/{len(ciks)}] {sym} (CIK {cik}): no facts")
            continue
        rows = extract_facts(data, sym, cik)
        all_rows.extend(rows)
        print(f"  [{i}/{len(ciks)}] {sym}: {len(rows)} fact rows")

    if not all_rows:
        print("No facts extracted.")
        return 1
    df = pd.DataFrame(all_rows, columns=_OUT_COLUMNS).sort_values(["symbol", "metric", "filed"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    cov = df.groupby("symbol")["metric"].nunique()
    print(f"\nWrote {len(df)} rows -> {out}")
    print(f"metric coverage per name (of {len(CONCEPTS)}): "
          f"min={cov.min()} max={cov.max()} mean={cov.mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
