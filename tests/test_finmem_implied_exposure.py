"""The reconstructed FinMem exposure must recover known positions, not the placeholder.

``finmem_baseline.py`` cannot read positions off an equity curve, so it writes a constant
1.0. On TSLA that placeholder said "fully invested every day" while the agent was in cash
on 42% of the days the stock moved. These tests pin the estimator that replaces it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "finmem_implied_exposure",
    Path(__file__).resolve().parents[1] / "scripts" / "finmem_implied_exposure.py",
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
try:
    _SPEC.loader.exec_module(mod)
except FileNotFoundError as exc:
    # The script resolves an external parquet panel at import time. Raising there is
    # right for a run, but at collection it takes the whole suite down with it, so a
    # machine without the dataset skips this module and keeps the actionable message.
    pytest.skip(str(exc), allow_module_level=True)


def _series(values, start="2024-07-10"):
    idx = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=len(values))]
    return pd.Series(values, index=idx)


def test_a_fully_invested_strategy_recovers_exposure_one():
    stock = _series([0.02, -0.03, 0.05, -0.02])
    exposure, _ = mod.implied_exposure(stock.copy(), stock)
    assert exposure.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_a_strategy_sitting_in_cash_recovers_exposure_zero():
    stock = _series([0.02, -0.03, 0.05, -0.02])
    exposure, _ = mod.implied_exposure(_series([0.0, 0.0, 0.0, 0.0]), stock)
    assert exposure.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_alternating_participation_is_recovered_day_by_day():
    stock = _series([0.02, -0.03, 0.05, -0.02])
    strat = _series([0.02, 0.0, 0.05, 0.0])          # long, cash, long, cash
    exposure, diag = mod.implied_exposure(strat, stock)
    assert exposure.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert diag["share_long"] == pytest.approx(0.5)
    assert diag["share_cash"] == pytest.approx(0.5)


def test_flat_days_carry_the_previous_estimate_rather_than_dividing_by_zero():
    # The middle day barely moves, so its ratio is unusable and must not become an
    # enormous or infinite exposure.
    stock = _series([0.02, 0.0001, -0.03])
    strat = _series([0.02, 0.0050, -0.03])
    exposure, diag = mod.implied_exposure(strat, stock)
    assert diag["days_usable"] == 2
    assert exposure.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert exposure.max() <= 1.0


def test_estimator_noise_is_clipped_to_the_long_or_cash_range():
    stock = _series([0.02, -0.03, 0.05])
    strat = _series([0.05, 0.001, -0.01])            # implies 2.5, then negative
    exposure, _ = mod.implied_exposure(strat, stock)
    assert exposure.min() >= 0.0 and exposure.max() <= 1.0


def test_a_leading_unusable_day_is_backfilled_not_left_missing():
    stock = _series([0.0002, 0.03, 0.04])
    strat = _series([0.0001, 0.03, 0.04])
    exposure, _ = mod.implied_exposure(strat, stock)
    assert not exposure.isna().any()
    assert exposure.iloc[0] == pytest.approx(1.0)


def test_the_diagnostics_count_only_days_the_stock_actually_moved():
    stock = _series([0.02, 0.0001, 0.0002, -0.03])
    strat = _series([0.02, 0.0000, 0.0000, -0.03])
    _, diag = mod.implied_exposure(strat, stock)
    assert diag["days_total"] == 4
    assert diag["days_usable"] == 2


def test_shares_are_taken_over_usable_days_not_the_whole_series():
    """Dividing by the full length understates every bucket and they stop summing to 1."""
    # Six days, only two of which the stock moves; the strategy is long on both.
    stock = _series([0.02, 0.0001, 0.0001, 0.0001, 0.0001, -0.03])
    strat = _series([0.02, 0.0000, 0.0000, 0.0000, 0.0000, -0.03])
    _, diag = mod.implied_exposure(strat, stock)
    assert diag["days_usable"] == 2
    assert diag["share_long"] == pytest.approx(1.0), "2 of 2 usable days, not 2 of 6"
    total = diag["share_long"] + diag["share_cash"] + diag["share_negative"]
    assert total == pytest.approx(1.0)
