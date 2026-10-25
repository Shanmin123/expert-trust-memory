"""PIT + ratio tests for the SEC XBRL fundamentals adapter and its context wiring."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def _facts(tmp_path):
    # Two fiscal years, same revenue tag (comparable), filed on realistic dates.
    rows = [
        # FY2020 (filed 2021-02-01)
        ("AAA", "revenue", "Revenues", "FY", 2020, "2020-12-31", "2021-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2020, "2020-12-31", "2021-02-01", 10.0),
        ("AAA", "assets", "Assets", "FY", 2020, "2020-12-31", "2021-02-01", 200.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2020, "2020-12-31", "2021-02-01", 80.0),
        ("AAA", "cfo", "NetCashProvidedByUsedInOperatingActivities", "FY", 2020, "2020-12-31", "2021-02-01", 20.0),
        ("AAA", "capex", "PaymentsToAcquirePropertyPlantAndEquipment", "FY", 2020, "2020-12-31", "2021-02-01", 5.0),
        # FY2021 (filed 2022-02-01) -- revenue +20%
        ("AAA", "revenue", "Revenues", "FY", 2021, "2021-12-31", "2022-02-01", 120.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2021, "2021-12-31", "2022-02-01", 18.0),
        ("AAA", "assets", "Assets", "FY", 2021, "2021-12-31", "2022-02-01", 240.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2021, "2021-12-31", "2022-02-01", 100.0),
        ("AAA", "cfo", "NetCashProvidedByUsedInOperatingActivities", "FY", 2021, "2021-12-31", "2022-02-01", 30.0),
        ("AAA", "capex", "PaymentsToAcquirePropertyPlantAndEquipment", "FY", 2021, "2021-12-31", "2022-02-01", 6.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_end", "filed", "val"]
    p = tmp_path / "facts.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)
    return str(p)


def test_features_are_point_in_time_and_ratios_correct(tmp_path):
    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    a = SecFundamentalsAlpha(_facts(tmp_path))

    # Before any 10-K is public -> unavailable.
    assert a.features_for("AAA", date(2020, 6, 1))["available"] is False

    # After the FY2020 10-K but before FY2021's -> uses FY2020 (no prior => growth None).
    f0 = a.features_for("AAA", date(2021, 6, 1))
    assert f0["fiscal_year"] == 2020
    assert f0["net_margin"] == pytest.approx(0.10)          # 10/100
    assert f0["roa"] == pytest.approx(0.05)                 # 10/200
    assert f0["leverage"] == pytest.approx((200 - 80) / 80) # liabilities derived = assets-equity
    assert f0["fcf_margin"] == pytest.approx((20 - 5) / 100)
    assert f0["revenue_growth_yoy"] is None

    # After the FY2021 10-K -> uses FY2021, growth = 20% (same tag => comparable).
    f1 = a.features_for("AAA", date(2022, 6, 1))
    assert f1["fiscal_year"] == 2021
    assert f1["revenue_growth_yoy"] == pytest.approx(0.20)
    assert f1["net_margin"] == pytest.approx(0.15)


def test_comparative_rows_do_not_shadow_the_fiscal_years_own_figures(tmp_path):
    # A 10-K carries the PRIOR year's figures as comparatives, and companyfacts
    # labels both with the filing's fy (fix 2026-07-18). Within the first release
    # the reader must take the LATEST period_end — otherwise row order can hand it
    # the comparative (observed on TSLA: "fy2022 roa" = 2022 income / 2021 assets).
    rows = [
        # FY2022 10-K (filed 2023-02-01): comparative 2021 rows FIRST in file order.
        ("AAA", "assets", "Assets", "FY", 2022, "2021-12-31", "2023-02-01", 100.0),
        ("AAA", "assets", "Assets", "FY", 2022, "2022-12-31", "2023-02-01", 200.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2021-12-31", "2023-02-01", 10.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2022-12-31", "2023-02-01", 20.0),
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-12-31", "2023-02-01", 80.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2022, "2022-12-31", "2023-02-01", 50.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_end", "filed", "val"]
    p = tmp_path / "facts_comparative.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    f = SecFundamentalsAlpha(str(p)).features_for("AAA", date(2023, 6, 1))
    assert f["fiscal_year"] == 2022
    assert f["roa"] == pytest.approx(20.0 / 200.0)          # own-year income / own-year assets
    assert f["net_margin"] == pytest.approx(20.0 / 80.0)
    assert f["leverage"] == pytest.approx((200.0 - 50.0) / 50.0)


def test_fundamental_context_includes_financials_block(tmp_path):
    from types import SimpleNamespace

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    from trading_agent.expert_context import _fundamental_context

    class _StubPanel:                       # price context degrades to None; financials still flow
        def trailing(self, ticker, t, n):
            return []

        def dates_upto(self, t):
            return []

        def _price_on(self, ticker, d):
            return None

    obs = SimpleNamespace(decision_date=date(2022, 6, 1), tickers=("AAA",))
    ctx = _fundamental_context(_StubPanel(), obs, filing_alpha=None,
                               sec_fundamentals=SecFundamentalsAlpha(_facts(tmp_path)))
    assert ctx["financials_available"] is True
    fin = ctx["by_ticker"]["AAA"]["financials"]
    assert fin["available"] is True and fin["fiscal_year"] == 2021
    assert fin["revenue_growth_yoy"] == pytest.approx(0.20)


def test_negative_equity_reports_leverage_none_not_nonsense(tmp_path):
    # Buyback-heavy issuers can carry negative book equity (ADSK 2018-2020):
    # liabilities/equity is then a misleading negative number. Preflight catch
    # 2026-07-18: report None. Cache-safe for positive-equity names.
    rows = [
        ("AAA", "revenue", "Revenues", "FY", 2019, "2019-12-31", "2020-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2019, "2019-12-31", "2020-02-01", 10.0),
        ("AAA", "assets", "Assets", "FY", 2019, "2019-12-31", "2020-02-01", 200.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2019, "2019-12-31", "2020-02-01", -12.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_end", "filed", "val"]
    p = tmp_path / "neg_equity.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    f = SecFundamentalsAlpha(str(p)).features_for("AAA", date(2020, 6, 1))
    assert f["available"] is True
    assert f["leverage"] is None          # not -17.7
    assert f["net_margin"] == pytest.approx(0.10)   # other ratios unaffected


def test_stale_comparative_metric_is_dropped_not_cross_period_mixed(tmp_path):
    # Preflight catch 2026-07-18 (the TSLA fy2011 pattern): equity's own-year row
    # is missing and only a stale comparative exists -> the snapshot must NOT mix
    # 2022 assets with 2021 equity; the stale metric becomes None.
    rows = [
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-12-31", "2023-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2022-12-31", "2023-02-01", 20.0),
        ("AAA", "assets", "Assets", "FY", 2022, "2022-12-31", "2023-02-01", 200.0),
        # equity ONLY as the prior-year comparative
        ("AAA", "equity", "StockholdersEquity", "FY", 2022, "2021-12-31", "2023-02-01", -50.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_end", "filed", "val"]
    p = tmp_path / "stale_comp.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    f = SecFundamentalsAlpha(str(p)).features_for("AAA", date(2023, 6, 1))
    assert f["available"] is True
    assert f["roa"] == pytest.approx(0.10)     # same-period 20/200
    assert f["leverage"] is None               # equity stale -> dropped, no mixing


def test_cover_dated_shares_rows_do_not_poison_the_snapshot_period(tmp_path):
    # 2026-07-18 incident regression: dei share-count instances are dated at the
    # filing COVER date (later than fiscal year end). They must not become the
    # snapshot period_end and nuke every real metric into None.
    rows = [
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-12-31", "2023-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2022-12-31", "2023-02-01", 20.0),
        ("AAA", "assets", "Assets", "FY", 2022, "2022-12-31", "2023-02-01", 200.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2022, "2022-12-31", "2023-02-01", 50.0),
        # dei shares row dated at the COVER date, AFTER fiscal year end
        ("AAA", "shares", "EntityCommonStockSharesOutstanding", "FY", 2022, "2023-02-15", "2023-02-01", 999.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_end", "filed", "val"]
    p = tmp_path / "cover_shares.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    f = SecFundamentalsAlpha(str(p)).features_for("AAA", date(2023, 6, 1))
    assert f["available"] is True
    assert f["net_margin"] == pytest.approx(0.20)   # NOT None
    assert f["roa"] == pytest.approx(0.10)
    assert f["leverage"] == pytest.approx(3.0)


def test_quarter_block_carries_latest_10q_pulse_pit_and_duration_filtered(tmp_path):
    # Quarterly upgrade 2026-07-18: latest 10-Q pulse rides in a 'quarter'
    # sub-block. PIT by filed; YTD duration rows excluded; same-quarter
    # tag-guarded YoY; None between a 10-K and the next Q1.
    rows = [
        # FY2022 10-K (filed 2023-02-01)
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-01-01", "2022-12-31", "2023-02-01", 400.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2022-01-01", "2022-12-31", "2023-02-01", 40.0),
        ("AAA", "assets", "Assets", "FY", 2022, None, "2022-12-31", "2023-02-01", 800.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2022, None, "2022-12-31", "2023-02-01", 300.0),
        # prior-year Q1 (for the YoY base), filed 2022-05-01
        ("AAA", "revenue", "Revenues", "Q1", 2022, "2022-01-01", "2022-03-31", "2022-05-01", 90.0),
        ("AAA", "net_income", "NetIncomeLoss", "Q1", 2022, "2022-01-01", "2022-03-31", "2022-05-01", 9.0),
        # current Q1 10-Q (filed 2023-05-01): 3-month row + a YTD row to be excluded
        ("AAA", "revenue", "Revenues", "Q1", 2023, "2023-01-01", "2023-03-31", "2023-05-01", 120.0),
        ("AAA", "net_income", "NetIncomeLoss", "Q1", 2023, "2023-01-01", "2023-03-31", "2023-05-01", 18.0),
        ("AAA", "revenue", "Revenues", "Q1", 2023, "2022-10-01", "2023-03-31", "2023-05-01", 999.0),  # 6-month: excluded
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_start", "period_end", "filed", "val"]
    p = tmp_path / "quarterly.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    a = SecFundamentalsAlpha(str(p))

    # Between the 10-K and the next Q1: no quarter block.
    f_pre = a.features_for("AAA", date(2023, 3, 1))
    assert f_pre["available"] is True and f_pre["quarter"] is None

    f = a.features_for("AAA", date(2023, 6, 1))
    q = f["quarter"]
    assert q["fiscal_period"] == "2023Q1"
    assert q["revenue_yoy_q"] == pytest.approx((120.0 - 90.0) / 90.0)  # 3-month row, not 999
    assert q["net_margin_q"] == pytest.approx(18.0 / 120.0)
    assert f["net_margin"] == pytest.approx(0.10)   # annual block unchanged (40/400)


def test_annual_duration_guard_rejects_q4_three_month_rows(tmp_path):
    # Some filers tag Q4 3-month flows in the 10-K frame with the SAME period_end
    # as the annual row: the annual figure must be the full-year duration row.
    rows = [
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-10-01", "2022-12-31", "2023-02-01", 110.0),
        ("AAA", "revenue", "Revenues", "FY", 2022, "2022-01-01", "2022-12-31", "2023-02-01", 400.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "2022-01-01", "2022-12-31", "2023-02-01", 40.0),
    ]
    cols = ["symbol", "metric", "tag", "fp", "fy", "period_start", "period_end", "filed", "val"]
    p = tmp_path / "q4guard.parquet"
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)

    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    f = SecFundamentalsAlpha(str(p)).features_for("AAA", date(2023, 6, 1))
    assert f["net_margin"] == pytest.approx(40.0 / 400.0)   # not 40/110


def _facts_frame(rows, tmp_path, name):
    cols = ["symbol", "metric", "tag", "fp", "fy", "form",
            "period_start", "period_end", "filed", "val"]
    p = tmp_path / name
    pd.DataFrame(rows, columns=cols).to_parquet(p, index=False)
    return str(p)


def test_cross_frame_assembly_joins_late_published_metrics_per_fact_pit(tmp_path):
    # TGT pattern (2026-07-19): a metric's own-year row is ABSENT from its own
    # frame and first appears as a comparative in a LATER 10-K. Period-end
    # keying joins it; per-fact PIT delays only THAT metric until its filing.
    p = _facts_frame([
        ("AAA", "revenue", "Revenues", "FY", 2020, "10-K", "2020-01-01", "2020-12-31", "2021-02-01", 100.0),
        ("AAA", "assets", "Assets", "FY", 2020, "10-K", None, "2020-12-31", "2021-02-01", 200.0),
        ("AAA", "equity", "StockholdersEquity", "FY", 2020, "10-K", None, "2020-12-31", "2021-02-01", 80.0),
        # net income for end 2020 first appears in the FY2022 10-K (filed 2023)
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2022, "10-K", "2020-01-01", "2020-12-31", "2023-02-01", 10.0),
    ], tmp_path, "cross_frame.parquet")
    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    a = SecFundamentalsAlpha(p)
    early = a.features_for("AAA", date(2021, 6, 1))
    assert early["fiscal_year"] == 2020 and early["net_margin"] is None   # NI not yet public
    late = a.features_for("AAA", date(2023, 6, 1))
    assert late["net_margin"] == pytest.approx(0.10)                      # joined when filed


def test_rolling_long_flows_in_10q_do_not_create_phantom_annual_ends(tmp_path):
    # AMZN pattern (2026-07-19): >=300-day rolling cfo rows inside 10-Qs must
    # not promote quarter ends into annual snapshots.
    p = _facts_frame([
        ("AAA", "revenue", "Revenues", "FY", 2020, "10-K", "2020-01-01", "2020-12-31", "2021-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2020, "10-K", "2020-01-01", "2020-12-31", "2021-02-01", 10.0),
        ("AAA", "cfo", "NetCashProvidedByUsedInOperatingActivities", "Q1", 2021, "10-Q",
         "2020-04-01", "2021-03-31", "2021-04-25", 999.0),   # 364d rolling window in a 10-Q
    ], tmp_path, "rolling.parquet")
    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    a = SecFundamentalsAlpha(p)
    f = a.features_for("AAA", date(2021, 6, 1))
    assert f["fiscal_year"] == 2020                       # NOT a phantom 2021-03-31 snapshot
    assert f["net_margin"] == pytest.approx(0.10)


def test_snapshot_availability_anchors_on_revenue_filing_not_latest_metric(tmp_path):
    # A late-first-tag comparative must not delay the whole snapshot: the
    # anchor is the revenue row's own 10-K filing.
    p = _facts_frame([
        ("AAA", "revenue", "Revenues", "FY", 2020, "10-K", "2020-01-01", "2020-12-31", "2021-02-01", 100.0),
        ("AAA", "net_income", "NetIncomeLoss", "FY", 2020, "10-K", "2020-01-01", "2020-12-31", "2021-02-01", 20.0),
        ("AAA", "equity", "NewEquityTag", "FY", 2022, "10-K", None, "2020-12-31", "2023-02-01", 50.0),
    ], tmp_path, "anchor.parquet")
    from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
    a = SecFundamentalsAlpha(p)
    f = a.features_for("AAA", date(2021, 6, 1))
    assert f["available"] is True and f["filed"] == "2021-02-01"   # not delayed to 2023
    assert f["net_margin"] == pytest.approx(0.20)
