"""Shared fixtures for trading_agent tests.

Tiny in-memory FinsaberDataset fixtures only, never the full parquet dataset (plan rule).
"""

from __future__ import annotations

import math
from datetime import date

import pytest

# Importing trading_agent triggers _bootstrap, which puts the FINSABER clone on sys.path.
from trading_agent._bootstrap import FinsaberDataset
from trading_agent.contracts import BacktestConfig


def _bar(o, h, l, c, v, adj=None):
    adj = c if adj is None else adj
    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "adjusted_close": adj,
        "volume": v,
    }


# Five business days, two clean tickers (no splits: adjusted_close == close).
_DEMO = {
    date(2024, 1, 2): {"price": {"AAA": _bar(100, 101, 99, 100, 1_000_000),
                                  "BBB": _bar(50, 51, 49, 50, 2_000_000)}},
    date(2024, 1, 3): {"price": {"AAA": _bar(101, 103, 100, 102, 1_100_000),
                                  "BBB": _bar(50.5, 52, 50, 51, 2_100_000)}},
    date(2024, 1, 4): {"price": {"AAA": _bar(102, 104, 101, 103, 1_050_000),
                                  "BBB": _bar(51, 53, 50.5, 52, 1_900_000)}},
    date(2024, 1, 5): {"price": {"AAA": _bar(103, 105, 102, 104, 1_200_000),
                                  "BBB": _bar(52, 54, 51, 53, 2_050_000)}},
    date(2024, 1, 8): {"price": {"AAA": _bar(104, 106, 103, 105, 1_150_000),
                                  "BBB": _bar(53, 55, 52, 54, 1_980_000)}},
}


@pytest.fixture
def demo_dataset():
    return FinsaberDataset(data={d: dict(v) for d, v in _DEMO.items()})


@pytest.fixture
def demo_config():
    return BacktestConfig(
        date_from=date(2024, 1, 2),
        date_to=date(2024, 1, 8),
        initial_cash=100_000.0,
        commission_per_share=0.0049,
        min_commission=0.99,
        max_commission_rate=0.01,
        slippage_perc=0.001,
        slippage_impact=0.0,
        liquidity_cap_pct=0.0,
        seed=7,
    )


@pytest.fixture
def missing_data_dataset():
    """BBB is absent on 2024-01-03 (the fill date for a 2024-01-02 decision)."""
    data = {
        date(2024, 1, 2): {"price": {"AAA": _bar(100, 101, 99, 100, 1_000_000),
                                      "BBB": _bar(50, 51, 49, 50, 2_000_000)}},
        date(2024, 1, 3): {"price": {"AAA": _bar(101, 103, 100, 102, 1_100_000)}},
        date(2024, 1, 4): {"price": {"AAA": _bar(102, 104, 101, 103, 1_050_000),
                                      "BBB": _bar(51, 53, 50.5, 52, 1_900_000)}},
    }
    return FinsaberDataset(data=data)


# --- Synthetic regime dataset: a TRENDING segment then a MEAN-REVERTING segment ----------
# In the trend, above-average-momentum names keep rising (signal right). In mean-reversion,
# above-average-momentum names are near peaks and fall next (signal wrong). So memory should
# learn to trust momentum under trend and distrust/abstain under mean-reversion.
def _flat_bar(c, v=1_000_000):
    return {"open": c, "high": c, "low": c, "close": c, "adjusted_close": c, "volume": v}


def build_regime_dataset():
    import pandas as pd

    trend_days, mr_days = 45, 90  # long mean-reversion tail so a mid-MR proposal gets >=3 purged blocks
    dates = list(pd.bdate_range("2022-01-03", periods=trend_days + mr_days))
    # ticker -> (base, trend_slope, phase)
    tickers = {"T1": (100, 0.006, 0.0), "T2": (120, 0.005, 0.7),
               "T3": (80, 0.007, 1.4), "T4": (150, 0.0055, 2.1)}
    data: dict = {}
    last: dict = {}
    # Warmup PREFIX before the decision window: the time-series technical signals
    # z-score against each name's own trailing history (momentum needs 71 closes,
    # lowvol 121), so decisions starting 2022-01-03 need prior bars. Gentle
    # non-degenerate oscillation around the base level; the returned date_from is
    # unchanged, so every hardcoded boundary in the tests stays valid.
    warmup = list(pd.bdate_range(end="2021-12-31", periods=140))
    for i, ts in enumerate(warmup):
        day = {}
        for tk, (base, slope, ph) in tickers.items():
            c = base * (1 + 0.004 * math.sin(i / 4.0 + ph))
            day[tk] = _flat_bar(round(c, 4))
        data[ts.date()] = {"price": day}
    for i, ts in enumerate(dates[:trend_days]):
        day = {}
        for tk, (base, slope, ph) in tickers.items():
            c = base * (1 + slope * i) + base * 0.002 * math.sin(i / 3.0 + ph)
            day[tk] = _flat_bar(round(c, 4))
            last[tk] = c
        data[ts.date()] = {"price": day}
    centers = dict(last)
    for j, ts in enumerate(dates[trend_days:]):
        day = {}
        for tk, (base, slope, ph) in tickers.items():
            c = centers[tk] * (1 + 0.04 * math.sin(2 * math.pi * j / 20.0 + ph))
            day[tk] = _flat_bar(round(c, 4))
        data[ts.date()] = {"price": day}
    return FinsaberDataset(data=data), dates[0].date(), dates[-1].date()


@pytest.fixture
def regime_dataset():
    ds, _, _ = build_regime_dataset()
    return ds


@pytest.fixture
def mem_backtest_config():
    _, dfrom, dto = build_regime_dataset()
    return BacktestConfig(
        date_from=dfrom, date_to=dto, initial_cash=100_000.0,
        commission_per_share=0.0049, min_commission=0.99, max_commission_rate=0.01,
        slippage_perc=0.0, liquidity_cap_pct=0.0, seed=11,
    )


