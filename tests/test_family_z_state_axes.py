"""Family Z (prereg Amendment 13): the three balanced state axes.

The 2D label's measured defect: down_calm holds 0.3% of training days, because trend and
risk are correlated (declines are volatile), so the label is effectively three imbalanced
states and certification is per-cell-support-limited. Z1 (vol tercile) and Z3 (down merged)
remove the empty cell by construction; Z2 (trend x conditional risk) splits each trend leg
at its own median so both axes survive with near-balance.

Classification is driven by seeding the internal rvol + trend series directly, the same
layer separation the 2D tests use: the axis logic is tested independent of the derivation.
The volatility path is a repeating ramp so that the trailing percentile actually sweeps
0..1 -- a monotone path would rank every day at the top of its own window.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_agent.alpha.market_state_classifier import (
    MARKET_STATES,
    WARMUP,
    MarketStateClassifier,
    MarketStateConfig,
)

WINDOW = 60
CYCLE = 30


def _dates(n, start=date(2021, 1, 4)):
    return [start + timedelta(days=i) for i in range(n)]


def _blank(cfg):
    clf = MarketStateClassifier.__new__(MarketStateClassifier)
    clf._panel = None
    clf._cfg = cfg
    clf._cache = {}
    return clf


def _seed(clf, universe, dates, rvol, trend):
    clf._cache[tuple(sorted(universe))] = {
        "dates": dates, "level": [None] * len(dates), "logret": [None] * len(dates),
        "rvol": rvol, "trend": trend,
    }


def _classifier(mode, rvol, trend, percentile_window=WINDOW):
    cfg = MarketStateConfig(mode=mode, percentile_window=percentile_window)
    clf = _blank(cfg)
    dts = _dates(len(rvol))
    _seed(clf, ["M"], dts, rvol, trend)
    return clf, dts


def _ramp(n, base=0.10, step=0.001):
    """rvol cycling through CYCLE distinct levels, so a trailing WINDOW (= 2 cycles) holds
    each level exactly twice and today's percentile is a known function of today's level."""
    return [base + (i % CYCLE) * step for i in range(n)]


def test_vol_tercile_uses_three_slots_and_leaves_the_fourth_unused():
    rvol = _ramp(240)
    clf, dts = _classifier("vol_tercile", rvol, ["up"] * 240)
    seen = {clf.classify_at(t, ["M"]) for t in dts}
    seen.discard(WARMUP)
    assert seen == {MARKET_STATES[0], MARKET_STATES[1], MARKET_STATES[3]}
    assert MARKET_STATES[2] not in seen, "the tercile label must not use the fourth slot"


def test_vol_tercile_boundaries_sit_at_the_thirds():
    # With two full cycles in the window, a day at level k has percentile (k+1)/CYCLE.
    # k = 8 -> 0.300 (below 1/3, low bin); k = 9 -> 0.3333 (not below, middle bin);
    # k = 18 -> 0.6333 (middle); k = 19 -> 0.6667 (not below 2/3, high bin).
    rvol = _ramp(240)
    clf, dts = _classifier("vol_tercile", rvol, ["up"] * 240)
    at = {k: clf.classify_at(dts[180 + k], ["M"]) for k in range(CYCLE)}
    assert at[8] == MARKET_STATES[0] and at[9] == MARKET_STATES[1], "1/3 boundary is inclusive upward"
    assert at[18] == MARKET_STATES[1] and at[19] == MARKET_STATES[3], "2/3 boundary is inclusive upward"


def test_down_merged_never_emits_down_calm():
    rvol = _ramp(240)
    trend = ["up"] * 120 + ["down"] * 120
    clf, dts = _classifier("down_merged", rvol, trend)
    seen = {clf.classify_at(t, ["M"]) for t in dts}
    assert "down_calm" not in seen, "the merged label must never emit down_calm"
    assert MARKET_STATES[3] in seen, "the down leg must land in the merged slot"
    assert seen <= {WARMUP, "up_calm", "up_stress", MARKET_STATES[3]}


def test_down_merged_keeps_the_up_leg_split_by_risk():
    rvol = _ramp(240)
    clf, dts = _classifier("down_merged", rvol, ["up"] * 240)
    seen = {clf.classify_at(t, ["M"]) for t in dts}
    assert "up_calm" in seen and "up_stress" in seen


def test_the_sealed_axis_reproduces_the_empty_down_calm_defect():
    """The defect Family Z exists to repair: when the down leg is volatile in absolute terms
    -- which is what declines are -- the unconditional median calls every down day stress and
    down_calm is never emitted. The leg must be SHORTER than the reference window, as real
    trend legs are against the sealed 504-day window; a leg longer than its own reference
    ends up compared against itself and the defect disappears."""
    rvol = _ramp(200, base=0.10) + _ramp(40, base=0.50)
    trend = ["up"] * 200 + ["down"] * 40
    clf, dts = _classifier("trend_risk", rvol, trend)
    seen = {clf.classify_at(t, ["M"]) for t in dts}
    assert "down_stress" in seen
    assert "down_calm" not in seen, "this fixture must exhibit the defect, or it tests nothing"


def test_trend_cond_risk_recovers_a_calm_down_state_on_that_same_path():
    rvol = _ramp(200, base=0.10) + _ramp(40, base=0.50)
    trend = ["up"] * 200 + ["down"] * 40
    clf, dts = _classifier("trend_cond_risk", rvol, trend)
    down = [clf.classify_at(t, ["M"]) for t in dts[200:]]
    down = [s for s in down if s != WARMUP]
    assert down, "the down leg must classify once it has enough same-side days"
    assert "down_calm" in down, "conditioning on the leg must recover a calm down state"
    assert "down_stress" in down


def test_trend_cond_risk_needs_enough_same_side_days():
    """A fresh trend flip has few same-side days; below percentile_window // 4 the label is
    warmup rather than a percentile over a handful of points."""
    rvol = _ramp(200, base=0.10) + _ramp(10, base=0.50)
    trend = ["up"] * 200 + ["down"] * 10
    clf, dts = _classifier("trend_cond_risk", rvol, trend)
    tail = {clf.classify_at(t, ["M"]) for t in dts[-10:]}
    assert tail == {WARMUP}, f"10 same-side days < {WINDOW} // 4 must stay warmup, saw {tail}"


def test_sealed_default_mode_is_untouched():
    rvol = _ramp(240)
    trend = ["up"] * 120 + ["down"] * 120
    clf, dts = _classifier("trend_risk", rvol, trend)
    seen = {clf.classify_at(t, ["M"]) for t in dts}
    assert seen <= {WARMUP, "up_calm", "up_stress", "down_calm", "down_stress"}
    assert {"up_calm", "up_stress", "down_calm", "down_stress"} <= seen


def test_unknown_mode_is_rejected_at_construction():
    """An unrecognised mode used to fall through SILENTLY to the 2D label -- a typo ran the
    wrong axis and reported success. Construction must refuse."""
    with pytest.raises(ValueError, match="unknown market-state mode"):
        MarketStateConfig(mode="bogus_axis")
