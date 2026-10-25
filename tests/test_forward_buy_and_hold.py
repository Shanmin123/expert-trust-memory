"""The live buy-and-hold comparison must match the sealed convention exactly.

Table tab:baselines reports buy and hold bought at the first open of the window and
held, ten basis points charged once, exposure 1.0 throughout. The forward run reuses
that reference, so it has to be computed the same way or the two are not comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from forward_buy_and_hold import COST_BPS, DRAW, series_for  # noqa: E402


def _rows():
    # (date, open, close): entry open 100, then closes 110, 121.
    return [("2026-08-03", 100.0, 110.0),
            ("2026-08-04", 111.0, 121.0),
            ("2026-08-05", 120.0, 121.0)]


def test_the_first_day_runs_from_the_open_and_pays_one_side():
    s = series_for(_rows(), "2026-08-03")
    # 110/100 - 1 = 0.10, minus 10bps charged once on entry.
    assert s[0][0] == "2026-08-03"
    assert s[0][1] == pytest.approx(0.10 - COST_BPS / 1e4)


def test_later_days_run_close_to_close_and_pay_nothing():
    s = series_for(_rows(), "2026-08-03")
    assert s[1][1] == pytest.approx(121.0 / 110.0 - 1.0)
    assert s[2][1] == pytest.approx(0.0)


def test_exposure_is_one_throughout():
    assert all(row[2] == 1.0 for row in series_for(_rows(), "2026-08-03"))


def test_cost_is_charged_once_not_daily():
    s = series_for(_rows(), "2026-08-03")
    gross = [0.10, 121.0 / 110.0 - 1.0, 0.0]
    charged = sum(g - n for g, (_d, n, _e) in zip(gross, s))
    assert charged == pytest.approx(COST_BPS / 1e4), "entry only, never again"


def test_sessions_before_the_window_are_ignored():
    rows = [("2026-07-31", 90.0, 95.0)] + _rows()
    s = series_for(rows, "2026-08-03")
    assert [r[0] for r in s] == ["2026-08-03", "2026-08-04", "2026-08-05"]


def test_the_draw_is_the_registered_twenty_without_tsla():
    assert len(DRAW) == 20 and "TSLA" not in DRAW


def test_the_convention_matches_a_sealed_artifact():
    """The artifacts declare net returns, exposure 1.0, and the schedule they charged.

    Every baseline moved onto FINSABER's own per-share commission on 2026-08-13, so
    the row here reads that schedule rather than the flat rate it used to carry.
    """
    sealed = ROOT / "reports" / "baselines" / "BIIB__BuyAndHoldStrategy.json"
    if not sealed.exists():
        pytest.skip("sealed buy-and-hold artifact not present")
    m = json.loads(sealed.read_text(encoding="utf-8"))["manifest"]
    assert m["returns"] == "net"
    assert m["cost_model"] == "commission"
    assert m["commission"] == {"commission_per_share": 0.0049,
                               "min_commission": 0.99,
                               "max_commission_rate": 0.01}
    rows = json.loads(sealed.read_text(encoding="utf-8"))["daily_arm_series"]["equal_weight"]
    assert all(r[2] == 1.0 for r in rows), "held throughout, as the live one must be"
