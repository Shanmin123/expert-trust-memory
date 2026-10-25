"""Tests for the portfolio-level primary metric (daily export + script math)."""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest

from trading_agent.experiments.expert_weight_holdout_launcher import _daily_series
from trading_agent.memory.expert_reflection import AttributedEpisode

_ATTR = {"market": 0.01, "style": 0.0, "signal": 0.005, "sizing": 0.0, "execution": 0.0}
_ZERO = {c: 0.0 for c in _ATTR}


def _ep(day, action, attribution, ticker="A"):
    return AttributedEpisode(
        decision_date=date(2024, 7, 1) + timedelta(days=day),
        market_state="up_calm",
        attribution=attribution,
        signal_attribution={"technical": 0.001, "news": -0.001},
        action=action,
        ticker=ticker,
    )


class _Loop:
    def __init__(self, episodes):
        self.episodes = episodes


def test_daily_series_gross_and_exposure():
    rows = _daily_series(_Loop([
        _ep(0, "long", _ATTR),
        _ep(1, "flat", _ZERO),
    ]))
    assert rows[0] == ["2024-07-01", pytest.approx(0.015), 1.0]
    assert rows[1] == ["2024-07-02", pytest.approx(0.0), 0.0]


def test_daily_series_averages_multi_ticker_dates():
    rows = _daily_series(_Loop([
        _ep(0, "long", _ATTR, ticker="A"),
        _ep(0, "flat", _ZERO, ticker="B"),
    ]))
    assert rows == [["2024-07-01", pytest.approx(0.0075), 0.5]]


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_arm_metrics.py"
    spec = importlib.util.spec_from_file_location("portfolio_arm_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_net_returns_charges_cost_on_exposure_change_only():
    mod = _load_script()
    per_date = {"d1": (0.01, 1.0), "d2": (0.01, 1.0), "d3": (0.0, 0.0)}
    net = mod._net_returns(per_date, ["d1", "d2", "d3"], bps=10.0)
    assert net[0] == pytest.approx(0.01 - 0.001)   # enter: |1-0| charged
    assert net[1] == pytest.approx(0.01)           # hold: no charge
    assert net[2] == pytest.approx(0.0 - 0.001)    # exit: |0-1| charged


def test_paired_block_bootstrap_detects_positive_shift():
    mod = _load_script()
    diffs = [0.002 + 0.0001 * ((i * 7) % 5 - 2) for i in range(120)]  # clearly > 0
    out = mod._block_bootstrap_ci(diffs, n_boot=400, seed=1)
    assert out["ci95"][0] > 0.0
    assert out["p_diff_le_0"] == 0.0
