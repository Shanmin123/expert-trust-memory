"""Point-in-time fundamental ratios from SEC XBRL financial statements.

Reads the collected SEC EDGAR companyfacts (``scripts/collect_sec_xbrl.py`` output) and
exposes, for a ticker at a decision date t, the ratios from its most recent ANNUAL (10-K)
report that was already public (``filed <= t``). This is the price-ORTHOGONAL fundamental
input the committee needs: an offline check found a real-financials fundamental direction
correlates only +0.005 with technical, versus +0.308 for the old price-derived inputs.

Presentation follows the reference multi-agent systems (TradingAgents' fundamentals analyst
filters statements by the decision date and hands the model margins/growth/leverage ratios),
adapted to this project's deterministic, point-in-time, no-runtime-API contract.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from trading_agent.market import _to_date

_FLOW = ("revenue", "net_income", "operating_income", "gross_profit", "cfo", "capex")
_STOCK = ("assets", "equity", "liabilities")


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


class SecFundamentalsAlpha:
    """Per-ticker point-in-time fundamental ratios from SEC XBRL annual figures."""

    def __init__(self, facts_path: str):
        import pandas as pd

        self._facts = pd.read_parquet(facts_path)
        self._facts["filed_d"] = pd.to_datetime(self._facts["filed"]).dt.date
        self._by_ticker: dict[str, list[dict]] = {}   # ticker -> annual snapshots (asc by filed)
        self._q_by_ticker: dict[str, list[dict]] = {}  # ticker -> quarterly snapshots

    def _annual_snapshots(self, ticker: str) -> list[dict]:
        import pandas as pd

        cached = self._by_ticker.get(ticker)
        if cached is not None:
            return cached
        # PERIOD-END KEYING (2026-07-19, TGT preflight catch): 52/53-week filers
        # get INCONSISTENT companyfacts fy labels ACROSS TAGS (TGT fy2015 frame:
        # revenue carries the own year, net_income is absent entirely, equity
        # carries only comparatives — the own-year rows live in NEIGHBOR frames),
        # so fy-keyed assembly can never complete those years. A fiscal year's
        # identity is its PERIOD END; the fy label is vendor metadata. Assemble
        # each snapshot from all rows sharing an end date, ACROSS frames:
        # annual ends = ends carrying a >=300-day flow row (the duration guard,
        # which also kills Q4-three-month rows); balances (instants, no start)
        # join by exact end. The comparative trap disappears BY CONSTRUCTION —
        # each end is its own snapshot. Names with consistent labels produce
        # identical output (byte-identity asserted against the 8 dev names).
        df = self._facts[self._facts["symbol"] == ticker]
        # Drop 'shares': no ratio reads it, and dei share-count instances are
        # dated at the filing COVER date (2026-07-18 incident).
        df = df[df["metric"] != "shares"]
        snaps: list[dict] = []
        if not df.empty:
            df = df.copy()
            df["period_end_d"] = pd.to_datetime(df["period_end"], errors="coerce")
            df["period_start_d"] = (
                pd.to_datetime(df["period_start"], errors="coerce")
                if "period_start" in df.columns else pd.NaT
            )
            df = df[df["period_end_d"].notna()]
            dur = (df["period_end_d"] - df["period_start_d"]).dt.days
            is_flow = df["metric"].isin(_FLOW)
            annual_flow = is_flow & df["period_start_d"].notna() & (dur >= 300)
            # Annual anchors must come from ANNUAL REPORTS (10-K family): some
            # filers tag >=300-day ROLLING flows in 10-Qs (AMZN cfo), which
            # would promote quarter ends into phantom annual snapshots
            # (identity-check catch 2026-07-19). Fixtures without a form
            # column keep the duration-only rule.
            if "form" in df.columns:
                annual_flow = annual_flow & df["form"].astype(str).str.startswith("10-K")
            no_start_flow = is_flow & df["period_start_d"].isna()
            balance = (~is_flow) & df["period_start_d"].isna()
            # No-start flow rows (legacy fixtures without period_start) keep the
            # pre-guard behaviour; real collector rows always carry start.
            df = df[annual_flow | no_start_flow | balance]
            if not df.empty:
                first_filed = df.groupby(["metric", "period_end_d"])["filed_d"].transform("min")
                first = (df[df["filed_d"] == first_filed]
                         .drop_duplicates(subset=["metric", "period_end_d"]))
                flow_mask = first["metric"].isin(_FLOW)
                annual_ends = sorted(first.loc[flow_mask, "period_end_d"].unique())
                rev_by_end: dict = {}
                for end in annual_ends:
                    g = first[first["period_end_d"] == end]
                    # PER-FACT PIT (identity-check catch 2026-07-19): each metric
                    # keeps its OWN first-filed date; a tag transition whose new
                    # tag first covers this end via a later comparative must not
                    # delay the whole snapshot. The snapshot's availability
                    # anchor = the revenue row's first filing (the annual
                    # report); features_for filters metrics by filed <= t.
                    cells = {m: (float(v), f, tg) for m, v, f, tg in
                             zip(g["metric"], g["val"], g["filed_d"], g["tag"])}
                    if "revenue" in cells:
                        anchor = cells["revenue"][1]
                    else:
                        anchor = min(f for (_v, f, _t) in cells.values())
                    end_date = pd.Timestamp(end).date()
                    snaps.append({
                        "fy": int(pd.Timestamp(end).year),
                        "end": end_date,
                        "filed": anchor,
                        "cells": cells,
                        "rev_tag": (cells.get("revenue") or (None, None, None))[2],
                    })
                    rev_by_end[end_date] = cells.get("revenue")
                snaps.sort(key=lambda s: (s["filed"], s["end"]))
                for s in snaps:
                    # Prior year = the end ~one year earlier (330-400d window,
                    # 52/53-week tolerant); comparable only under the SAME
                    # us-gaap tag; carries its own filed for read-time PIT.
                    prior = None
                    for pe in sorted(rev_by_end):
                        if 330 <= (s["end"] - pe).days <= 400:
                            prior = rev_by_end[pe]
                    s["rev_prior_cell"] = (
                        prior if (prior and prior[2] == s["rev_tag"]) else None
                    )
        self._by_ticker[ticker] = snaps
        return snaps

    def _quarterly_snapshots(self, ticker: str) -> list[dict]:
        """Latest-quarter pulse (10-Q revenue/net income), PIT by ``filed``.

        Quarterly upgrade 2026-07-18 (information-rate fix:
        annual-only fundamentals update once a year, so the expert's inputs are
        a constant for ~12 months). Same discipline as the annual path: first
        release per (metric, fy, fp), one snapshot = one period_end, tag-guarded
        same-quarter YoY, and a 60-120 day DURATION filter so the 10-Q's YTD
        duration rows never masquerade as the quarter.
        """
        import pandas as pd

        cached = self._q_by_ticker.get(ticker)
        if cached is not None:
            return cached
        df = self._facts[(self._facts["symbol"] == ticker)
                         & (self._facts["fp"].isin(("Q1", "Q2", "Q3")))]
        df = df[df["metric"].isin(("revenue", "net_income"))]
        snaps: list[dict] = []
        if not df.empty:
            df = df.copy()
            df["period_end_d"] = pd.to_datetime(df["period_end"], errors="coerce")
            df["period_start_d"] = (
                pd.to_datetime(df["period_start"], errors="coerce")
                if "period_start" in df.columns else pd.NaT
            )
            dur = (df["period_end_d"] - df["period_start_d"]).dt.days
            df = df[df["period_start_d"].notna() & dur.between(60, 120)]
            if not df.empty:
                first_filed = df.groupby(["metric", "fy", "fp"])["filed_d"].transform("min")
                release = df[df["filed_d"] == first_filed]
                first = (release.sort_values("period_end_d")
                         .groupby(["metric", "fy", "fp"], as_index=False).last())
                by_q: dict = {}
                for (fy, fp), g in first.groupby(["fy", "fp"]):
                    snap_end = g["period_end_d"].max()
                    aligned = g[g["period_end_d"] == snap_end]
                    vals = {m: float(v) for m, v in zip(aligned["metric"], aligned["val"])}
                    tags = {m: t for m, t in zip(aligned["metric"], aligned["tag"])}
                    snaps.append({"fy": int(fy), "fp": str(fp), "filed": max(g["filed_d"]),
                                  "vals": vals, "rev_tag": tags.get("revenue")})
                    by_q[(int(fy), str(fp))] = (vals.get("revenue"), tags.get("revenue"))
                snaps.sort(key=lambda s: s["filed"])
                for s in snaps:
                    prior = by_q.get((s["fy"] - 1, s["fp"]))
                    s["rev_prior"] = (
                        prior[0] if (prior and prior[1] == s["rev_tag"]) else None
                    )
        self._q_by_ticker[ticker] = snaps
        return snaps

    def features_for(self, ticker: str, t) -> dict:
        """Ratios from the latest annual report public by date ``t`` (filed <= t)."""
        t = _to_date(t)
        snaps = self._annual_snapshots(ticker)
        latest = None
        for s in snaps:
            if s["filed"] <= t:
                latest = s
            else:
                break
        if latest is None:
            return {"available": False}
        # PER-FACT PIT read (2026-07-19): a metric only enters the snapshot view
        # once its OWN first filing is public by t. Consistent filers share one
        # 10-K filed date, so this is byte-identical for them; a late-first-tag
        # comparative joins when it actually became public.
        v = {m: cell[0] for m, cell in latest["cells"].items() if cell[1] <= t}
        rev = v.get("revenue")
        equity = v.get("equity")
        assets = v.get("assets")
        liab = v.get("liabilities")
        if liab is None and assets is not None and equity is not None:
            liab = assets - equity                    # derive when the tag was absent
        fcf = None
        if v.get("cfo") is not None and v.get("capex") is not None:
            fcf = v["cfo"] - abs(v["capex"])

        def r(x):
            return None if x is None else round(x, 6)

        prior_cell = latest.get("rev_prior_cell")
        rev_prior = (
            prior_cell[0] if (prior_cell is not None and prior_cell[1] <= t) else None
        )
        growth = (_safe_div(rev - rev_prior, rev_prior)
                  if (rev is not None and rev_prior) else None)
        if growth is not None and not (-0.95 <= growth <= 3.0):
            growth = None                             # reject residual tag/units artifacts

        # Quarterly pulse (2026-07-18): the newest 10-Q filed at/after the annual
        # snapshot and public by t. None between a 10-K and the next Q1 — the
        # annual block is the fresher document then.
        quarter_block = None
        chosen_q = None
        for q in self._quarterly_snapshots(ticker):
            if q["filed"] <= t and q["filed"] >= latest["filed"]:
                chosen_q = q
        if chosen_q is not None:
            qv = chosen_q["vals"]
            q_growth = (
                _safe_div(qv["revenue"] - chosen_q["rev_prior"], chosen_q["rev_prior"])
                if (qv.get("revenue") is not None and chosen_q.get("rev_prior"))
                else None
            )
            if q_growth is not None and not (-0.95 <= q_growth <= 3.0):
                q_growth = None
            quarter_block = {
                "fiscal_period": f"{chosen_q['fy']}{chosen_q['fp']}",
                "filed": chosen_q["filed"].isoformat(),
                "days_since_filing": (t - chosen_q["filed"]).days,
                "revenue_yoy_q": r(q_growth),
                "net_margin_q": r(_safe_div(qv.get("net_income"), qv.get("revenue"))),
            }

        return {
            "available": True,
            "fiscal_year": latest["fy"],
            "filed": latest["filed"].isoformat(),
            "days_since_filing": (t - latest["filed"]).days,
            "revenue_growth_yoy": r(growth),
            "gross_margin": r(_safe_div(v.get("gross_profit"), rev)),
            "operating_margin": r(_safe_div(v.get("operating_income"), rev)),
            "net_margin": r(_safe_div(v.get("net_income"), rev)),
            "roa": r(_safe_div(v.get("net_income"), assets)),
            "fcf_margin": r(_safe_div(fcf, rev)),
            # Negative book equity (buyback-heavy issuers, e.g. ADSK 2018-2020)
            # makes liabilities/equity a misleading negative number; report None
            # instead of feeding the expert nonsense (preflight catch 2026-07-18).
            # Cache-safe: names with positive equity produce identical prompts.
            "leverage": r(_safe_div(liab, equity)) if (equity is not None and equity > 0) else None,
            # Latest-quarter pulse (10-Q): the freshest fundamental evidence.
            "quarter": quarter_block,
        }
