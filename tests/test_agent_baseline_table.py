"""The agent comparison must not invent a relationship, and must reuse one cost path.

The interesting claim about the agent baselines is that their shortfall against buy and
hold is a function of the stock's own return. That is a cross-stock claim, so it carries
the project's standing requirement: it only counts if it survives dropping any one stock.
These tests pin the slope estimator and the reuse of baseline_table's cost handling.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "agent_baseline_table",
    Path(__file__).resolve().parents[1] / "scripts" / "agent_baseline_table.py",
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


def test_slope_recovers_a_known_line():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, -0.5, -2.0, -3.5]          # slope -1.5
    assert mod.slope_through(xs, ys) == pytest.approx(-1.5)


def test_a_constant_exposure_gap_gives_that_exposure_minus_one():
    # If the agent always holds e of the stock, gap = (e-1) * r, so the slope is e-1.
    e = 0.4
    xs = [-0.2, 0.0, 0.15, 0.6]
    ys = [(e - 1.0) * r for r in xs]
    assert mod.slope_through(xs, ys) == pytest.approx(e - 1.0)


def test_slope_is_nan_when_every_stock_returned_the_same():
    """Not exact-zero: floating-point noise in x must still refuse, not invent a slope."""
    import math
    assert math.isnan(mod.slope_through([0.1, 0.1, 0.1], [1.0, 2.0, 3.0]))
    assert math.isnan(mod.slope_through([0.1, 0.1 + 1e-17, 0.1], [1.0, 2.0, 3.0]))
    assert math.isnan(mod.slope_through([], []))


def test_stocks_that_merely_returned_something_similar_still_give_a_slope():
    # ADSK and PNC differ by 0.12 of a percentage point in the real data; that is a real
    # difference and must not be swallowed by the guard.
    s = mod.slope_through([0.2351, 0.2363, -0.2362, 0.6339], [-0.380, -0.267, 0.429, -0.804])
    assert not (s != s) and s < 0


def test_the_cost_path_is_baseline_tables_not_a_second_copy():
    assert mod.bt.net is not None
    assert mod.bt.COST == pytest.approx(10.0 / 1e4)
    # a gross series is charged on exposure changes, a net one is passed through
    rows = [["2024-07-09", 0.01, 1.0], ["2024-07-10", 0.01, 0.0]]
    charged = mod.bt.net(rows, charge=True)
    passed = mod.bt.net(rows, charge=False)
    assert charged["2024-07-09"] < passed["2024-07-09"]
    assert passed["2024-07-09"] == pytest.approx(0.01)


def test_an_undeclared_returns_basis_is_refused(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(mod.bt, "BASE", tmp_path)
    (tmp_path / "AAA__FinMem.json").write_text(json.dumps({
        "daily_arm_series": {"equal_weight": [["2024-07-09", 0.0, 1.0]]},
        "manifest": {"ticker": "AAA"},          # no "returns" key
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="net or gross"):
        mod.load_agent("FinMem", "AAA")


def test_a_missing_agent_run_is_reported_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.bt, "BASE", tmp_path)
    assert mod.load_agent("FinMem", "NOPE") is None


def test_the_eight_tickers_come_from_the_sealed_map():
    assert set(mod.TICKERS) == set(mod.bt.SEALED)
    assert len(mod.TICKERS) == 8
