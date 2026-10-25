"""Tests for the external SPY market proxy (single-name runs)."""

from __future__ import annotations

import os
from datetime import date

import pytest

from trading_agent.data.market_proxy import MarketProxy


def _proxy():
    # closes: 100, 110, 99, 108 on four consecutive dates.
    return MarketProxy(
        [date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5), date(2022, 1, 6)],
        [100.0, 110.0, 99.0, 108.0],
    )


def test_daily_return_is_causal():
    p = _proxy()
    assert p.daily_return(date(2022, 1, 3)) is None          # no prior close
    assert p.daily_return(date(2022, 1, 4)) == pytest.approx(0.10)
    assert p.daily_return(date(2022, 1, 5)) == pytest.approx(99.0 / 110.0 - 1.0)
    # a non-trading query date uses the most recent close on/before it.
    assert p.daily_return(date(2022, 1, 7)) == pytest.approx(108.0 / 99.0 - 1.0)


def test_forward_return_looks_one_proxy_date_ahead():
    p = _proxy()
    assert p.forward_return(date(2022, 1, 3), 1) == pytest.approx(0.10)      # 100 -> 110
    assert p.forward_return(date(2022, 1, 5), 1) == pytest.approx(108.0 / 99.0 - 1.0)
    assert p.forward_return(date(2022, 1, 6), 1) is None                     # no next date


def test_returns_by_date_drops_undefined():
    p = _proxy()
    r = p.returns_by_date([date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)])
    assert date(2022, 1, 3) not in r          # first date has no return
    assert r[date(2022, 1, 4)] == pytest.approx(0.10)


_SPY = "data/market_proxy/spy_daily.parquet"


_VIX = "data/market_proxy/vix_daily.parquet"


@pytest.mark.skipif(not os.path.exists(_VIX), reason="VIX not downloaded")
def test_real_vix_levels_load():
    p = MarketProxy.from_parquet(_VIX)
    assert len(p) > 3000
    assert p.level_on(date(2020, 3, 20)) > 40    # COVID spike
    assert p.level_on(date(2017, 6, 1)) < 20      # calm year


@pytest.mark.skipif(not os.path.exists(_SPY), reason="SPY proxy not downloaded")
def test_real_spy_proxy_loads_and_is_causal():
    p = MarketProxy.from_parquet(_SPY)
    assert len(p) > 3000
    r = p.daily_return(date(2020, 3, 16))     # a real crash day -> large negative
    assert r is not None and r < 0
    assert -0.2 < r < 0.0
