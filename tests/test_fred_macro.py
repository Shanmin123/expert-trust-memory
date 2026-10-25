"""PIT tests for the FRED macro feature block (design-D news expert input)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from trading_agent.alpha.fred_macro import FredMacroAlpha


def _write(tmp_path, rows):
    p = tmp_path / "fred_macro.parquet"
    pd.DataFrame(rows, columns=["series", "date", "value"]).to_parquet(p, index=False)
    return p


def test_missing_file_reports_unavailable(tmp_path):
    alpha = FredMacroAlpha(tmp_path / "nope.parquet")
    assert alpha.available is False
    assert alpha.features_for(date(2024, 7, 1)) == {"available": False}


def test_daily_series_readable_same_day_and_uses_last_known(tmp_path):
    p = _write(tmp_path, [
        ("DFF", date(2022, 3, 16), 0.33),   # hike day
        ("DFF", date(2022, 3, 15), 0.08),
        ("T10Y2Y", date(2022, 3, 16), 0.21),
    ])
    alpha = FredMacroAlpha(p)
    f = alpha.features_for(date(2022, 3, 16))
    assert f["fed_funds_rate"] == pytest.approx(0.33)
    assert f["yield_curve_10y_minus_2y"] == pytest.approx(0.21)
    # The day BEFORE the hike, only the old rate is known.
    assert alpha.features_for(date(2022, 3, 15))["fed_funds_rate"] == pytest.approx(0.08)


def test_monthly_series_is_embargoed_by_publication_lag(tmp_path):
    p = _write(tmp_path, [("CPIAUCSL", date(2022, 3, 1), 287.5),
                          ("CPIAUCSL", date(2021, 3, 1), 264.9)])
    alpha = FredMacroAlpha(p)
    # On the observation date itself the March CPI is NOT yet published (45-day lag).
    assert alpha.features_for(date(2022, 3, 2))["cpi_yoy_pct"] is None
    # After the lag both this year's and last year's values are readable -> YoY works.
    f = alpha.features_for(date(2022, 5, 1))
    assert f["cpi_yoy_pct"] == pytest.approx((287.5 / 264.9 - 1.0) * 100.0, abs=0.01)


def test_fed_funds_3m_change_detects_a_hiking_cycle(tmp_path):
    p = _write(tmp_path, [("DFF", date(2022, 1, 3), 0.08),
                          ("DFF", date(2022, 6, 1), 0.83)])
    alpha = FredMacroAlpha(p)
    f = alpha.features_for(date(2022, 6, 1))
    assert f["fed_funds_change_3m"] == pytest.approx(0.75)


def test_claims_4wk_change(tmp_path):
    from datetime import timedelta
    rows = [("ICSA", date(2022, 1, 1) + timedelta(days=7 * i), 200000.0 + 10000 * i)
            for i in range(6)]
    alpha = FredMacroAlpha(_write(tmp_path, rows))
    f = alpha.features_for(date(2022, 3, 1))
    assert f["initial_claims_change_4wk"] == pytest.approx(40000.0)
