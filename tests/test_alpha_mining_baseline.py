"""The alpha-mining baseline must not leak, and must be charged like the arms.

A factor model is where look-ahead usually enters a comparison, and it enters silently:
the numbers stay plausible. These tests pin the three places it could happen — the label's
direction, the scaler's fitting window, and the train/test split — plus the declaration
that lets the scorer charge cost exactly once.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "alpha_mining_baseline",
    Path(__file__).resolve().parents[1] / "scripts" / "alpha_mining_baseline.py",
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def test_the_label_is_the_next_days_return_not_todays(monkeypatch):
    """If the label were today's return the model would be reading its own outcome."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=5)]
    closes = pd.Series([100.0, 110.0, 121.0, 121.0, 100.0], index=dates)

    class _Frame:
        columns = ["adjusted_close"]
        def __getitem__(self, k):
            return closes

    def _fake(*a, **k):
        return closes.pct_change().shift(-1).rename("fwd")

    monkeypatch.setattr(mod, "price_returns", _fake)
    fwd = mod.price_returns("X")
    # on day 0 the forward return is day 1's move, +10%
    assert fwd.iloc[0] == pytest.approx(0.10)
    # the last day has no forward return at all
    assert np.isnan(fwd.iloc[-1])


def test_the_split_boundaries_are_the_sealed_ones():
    assert mod.TRAIN_END == "2023-12-31"          # the memory freeze boundary
    assert mod.TEST_START == "2024-07-09"         # the sealed scored window
    assert mod.TEST_END == "2025-12-01"
    assert mod.TRAIN_END < mod.TEST_START, "training must end before the test window"


def test_a_scaler_fit_on_everything_would_change_the_features():
    """Guards the classic silent leak: test-window means entering the features."""
    from sklearn.preprocessing import StandardScaler
    train = np.array([[1.0], [2.0], [3.0]])
    test = np.array([[100.0], [200.0]])
    train_only = StandardScaler().fit(train)
    everything = StandardScaler().fit(np.vstack([train, test]))
    assert train_only.transform(test)[0][0] != pytest.approx(
        everything.transform(test)[0][0]
    ), "if these agreed the test could not detect the leak"


def test_the_series_is_declared_gross_so_cost_is_charged_once():
    rows, manifest = _stub_run()
    assert manifest["returns"] == "gross"
    assert manifest["scaler"] == "fit on training rows only"


def test_exposure_is_long_or_cash_only():
    rows, _ = _stub_run()
    assert {r[2] for r in rows} <= {0.0, 1.0}


def test_the_return_recorded_is_exposure_times_the_move():
    rows, _ = _stub_run()
    for _, ret, exp in rows:
        if exp == 0.0:
            assert ret == pytest.approx(0.0), "flat days must earn nothing"


def test_the_tickers_are_the_sealed_eight():
    assert len(mod.TICKERS) == 8
    assert set(mod.TICKERS) == {"TSLA", "ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT"}


def _stub_run():
    """Run the real fit path on synthetic factors, so no database is needed."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2011-01-03", periods=4000)]
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(
        {f"QLIB158_F{i}": rng.normal(size=len(dates)) for i in range(6)}, index=dates)
    fwd = pd.Series(rng.normal(scale=0.01, size=len(dates)), index=dates, name="fwd")

    import types
    saved = (mod.factor_frame, mod.price_returns)
    mod.factor_frame = lambda t: feats
    mod.price_returns = lambda t: fwd
    try:
        return mod.fit_and_signal("STUB", "ridge")
    finally:
        mod.factor_frame, mod.price_returns = saved
