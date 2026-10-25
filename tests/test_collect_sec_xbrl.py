"""Offline test of the SEC XBRL companyfacts parser (no network)."""

from __future__ import annotations

from scripts.collect_sec_xbrl import extract_facts, CONCEPTS

_FACTS = {
    "facts": {
        "us-gaap": {
            # 'Revenues' absent -> should fall back to the next alternate tag
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 1000.0,
                     "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
                    # a restatement of the same period, filed later -> kept (differs by 'filed')
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 1010.0,
                     "fy": 2023, "fp": "FY", "form": "10-K/A", "filed": "2024-06-01"},
                ]}
            },
            "NetIncomeLoss": {
                "units": {"USD": [
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 120.0,
                     "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
                ]}
            },
            # missing 'filed' -> must be skipped
            "Assets": {"units": {"USD": [{"end": "2023-12-31", "val": 5000.0}]}},
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-02-01", "val": 50.0, "form": "10-K", "filed": "2024-02-15"},
                ]}
            }
        },
    }
}


def test_extract_facts_maps_metrics_and_keeps_pit_fields():
    rows = extract_facts(_FACTS, "AAA", "0000000001")
    by_metric = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r)

    # revenue resolved via the alternate tag, both filings kept (restatement differs by filed)
    assert by_metric["revenue"][0]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert {r["val"] for r in by_metric["revenue"]} == {1000.0, 1010.0}
    assert {r["filed"] for r in by_metric["revenue"]} == {"2024-02-15", "2024-06-01"}

    # every row carries the PIT filed date + period_end + symbol/cik
    assert all(r["filed"] and r["period_end"] and r["symbol"] == "AAA" for r in rows)

    # net income present; shares pulled from the dei pool with the 'shares' unit
    assert by_metric["net_income"][0]["val"] == 120.0
    assert by_metric["shares"][0]["unit"].startswith("shares")

    # Assets instance had no 'filed' -> dropped (no look-ahead-undatable facts)
    assert "assets" not in by_metric


def test_extract_facts_empty_on_no_facts():
    assert extract_facts({}, "X", "1") == []
    assert extract_facts({"facts": {"us-gaap": {}}}, "X", "1") == []


def test_extract_facts_unions_across_alternate_tags():
    # Tag drift (fix 2026-07-18): the priority tag covers only fy2023; the older
    # years live under a lower-priority tag. First-tag-wins for the whole history
    # dropped them (the TSLA fy2012-2022 revenue hole) — the union must keep both,
    # and a same-period duplicate must resolve to the higher-priority tag.
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-12-31", "val": 500.0,
             "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
        ]}},
        "SalesRevenueNet": {"units": {"USD": [
            {"start": "2012-01-01", "end": "2012-12-31", "val": 40.0,
             "fy": 2012, "fp": "FY", "form": "10-K", "filed": "2013-02-15"},
            # duplicate of the SAME fy2023 period+filed under the lower-priority tag
            {"start": "2023-01-01", "end": "2023-12-31", "val": 499.0,
             "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
        ]}},
    }}}
    rows = [r for r in extract_facts(facts, "AAA", "1") if r["metric"] == "revenue"]
    by_fy = {r["fy"]: r for r in rows}
    assert set(by_fy) == {2012, 2023}                       # old years recovered
    assert by_fy[2012]["tag"] == "SalesRevenueNet"
    assert by_fy[2023]["val"] == 500.0                      # priority tag wins the tie
    assert by_fy[2023]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
