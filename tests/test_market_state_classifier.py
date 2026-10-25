"""Tests for the 2D (trend x risk) market_state classifier.

Two layers are tested separately:
  * realized_vol math (zero-mean, log returns) and the MA200 trend side, each driven by a
    controlled price path.
  * risk percentile / state assignment / warmup, driven by seeding the internal series
    (rvol + trend) directly, so classification is tested independent of the derivation.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from trading_agent.alpha.market_state_classifier import (
    MARKET_STATES,
    WARMUP,
    MarketStateClassifier,
    MarketStateConfig,
)
from trading_agent.market import LoaderPricePanel


# ---------------------------------------------------------------------------
# Minimal controllable loader
# ---------------------------------------------------------------------------

class FakeLoader:
    """Implements just the two methods LoaderPricePanel calls."""

    def __init__(self, prices_by_date: dict):
        self._p = prices_by_date

    def get_date_range(self):
        return sorted(self._p.keys())

    def get_data_by_date(self, d):
        return {"price": {tk: {"adjusted_close": px} for tk, px in self._p[d].items()}}


def _dates(n, start=date(2021, 1, 4)):
    return [start + timedelta(days=i) for i in range(n)]


def _panel_from_prices(prices_by_date):
    return LoaderPricePanel(FakeLoader(prices_by_date))


def _panel_from_path(ticker, dts, prices):
    return _panel_from_prices({d: {ticker: p} for d, p in zip(dts, prices)})


def _blank(cfg):
    clf = MarketStateClassifier.__new__(MarketStateClassifier)
    clf._panel = None
    clf._cfg = cfg
    clf._cache = {}
    return clf


def _seed(clf, universe, dates, rvol, trend):
    """Seed the internal series so classify_at reads rvol + trend directly."""
    key = tuple(sorted(universe))
    clf._cache[key] = {
        "dates": dates,
        "level": [None] * len(dates),
        "logret": [None] * len(dates),
        "rvol": rvol,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# realized_vol math: zero-mean, log returns
# ---------------------------------------------------------------------------

def test_constant_returns_give_nonzero_zero_mean_vol():
    # +1%/day -> constant LOG return ln(1.01). Zero-mean RV does NOT subtract the mean,
    # so RV = sqrt(252) * |ln(1.01)| > 0 (this is the whole point of not demeaning).
    dts = _dates(6)
    prices = [100.0]
    for _ in range(5):
        prices.append(prices[-1] * 1.01)
    panel = _panel_from_path("M", dts, prices)
    clf = MarketStateClassifier(panel, MarketStateConfig(vol_window=3))
    rv = clf.realized_vol_at(dts[5], ["M"])
    assert rv == pytest.approx(math.sqrt(252.0) * abs(math.log(1.01)))


def test_realized_vol_matches_zero_mean_log_formula():
    cfg = MarketStateConfig(vol_window=3, annualization=252)
    simple = [None, 0.02, -0.02, 0.02, -0.01, 0.03]  # index 0 has no prior bar
    dts = _dates(len(simple))
    prices = [100.0]
    for r in simple[1:]:
        prices.append(prices[-1] * (1.0 + r))
    panel = _panel_from_path("M", dts, prices)
    clf = MarketStateClassifier(panel, cfg)

    idx = 5  # window = log(1+r) for r in [0.02, -0.01, 0.03]
    logs = [math.log(1.0 + r) for r in simple[idx - cfg.vol_window + 1 : idx + 1]]
    expected = math.sqrt(cfg.annualization * (sum(x * x for x in logs) / len(logs)))
    assert clf.realized_vol_at(dts[idx], ["M"]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# MA200 trend side + 5-day confirmation
# ---------------------------------------------------------------------------

def test_trend_side_up_and_down():
    cfg = MarketStateConfig(trend_window=3, trend_confirm_days=1)
    # rising path -> close above its 3-day SMA -> "up".
    up = _panel_from_path("M", _dates(6), [100, 101, 102, 103, 104, 105])
    clf_up = MarketStateClassifier(up, cfg)
    assert clf_up.trend_at(_dates(6)[5], ["M"]) == "up"
    # falling path -> "down".
    dn = _panel_from_path("M", _dates(6), [105, 104, 103, 102, 101, 100])
    clf_dn = MarketStateClassifier(dn, cfg)
    assert clf_dn.trend_at(_dates(6)[5], ["M"]) == "down"


def test_trend_switch_requires_confirmation():
    # Long uptrend, then a brief 2-day dip below the SMA. With confirm_days=5 the state
    # must NOT flip to "down" on a 2-day excursion.
    cfg = MarketStateConfig(trend_window=3, trend_confirm_days=5)
    dts = _dates(12)
    prices = [100, 102, 104, 106, 108, 110, 103, 101, 112, 114, 116, 118]
    #                                        ^dip^ (2 days) then recovers
    clf = MarketStateClassifier(_panel_from_path("M", dts, prices), cfg)
    # the dip is shorter than confirm_days -> trend stays "up" throughout.
    assert clf.trend_at(dts[7], ["M"]) == "up"
    assert clf.trend_at(dts[11], ["M"]) == "up"


# ---------------------------------------------------------------------------
# Risk percentile -> {calm, stress}, crossed with trend (seed rvol + trend directly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("today,side,expected", [
    (5.0, "up", "up_calm"),        # below all 4 -> pct low -> calm
    (5.0, "down", "down_calm"),
    (45.0, "up", "up_stress"),     # top -> pct 1.0 >= 0.5 -> stress
    (45.0, "down", "down_stress"),
    (25.0, "up", "up_stress"),     # median: <= {10,20,25}? incl current -> 3/5=0.6 >= 0.5
])
def test_state_assignment(today, side, expected):
    cfg = MarketStateConfig(percentile_window=5, stress_percentile=0.50)
    clf = _blank(cfg)
    dts = _dates(5)
    # reference for idx=4 INCLUDES current: rvol[0:5] = [10,20,30,40,today].
    rvol = [10.0, 20.0, 30.0, 40.0, today]
    trend = [side] * 5
    _seed(clf, ["M"], dts, rvol, trend)
    assert clf.classify_at(dts[4], ["M"]) == expected


def test_percentile_includes_current_day():
    # percentile_window=4, reference for idx=3 = rvol[0:4] = [10,20,30,today] (incl current).
    # today=30 -> pct = |{10,20,30}|/4 = 0.75 >= 0.5 -> stress.
    cfg = MarketStateConfig(percentile_window=4, stress_percentile=0.50)
    clf = _blank(cfg)
    dts = _dates(4)
    _seed(clf, ["M"], dts, [10.0, 20.0, 30.0, 30.0], ["up"] * 4)
    assert clf.classify_at(dts[3], ["M"]) == "up_stress"


# ---------------------------------------------------------------------------
# Warmup handling
# ---------------------------------------------------------------------------

def test_warmup_when_rvol_undefined():
    cfg = MarketStateConfig(percentile_window=2)
    clf = _blank(cfg)
    dts = _dates(4)
    _seed(clf, ["M"], dts, [1.0, 2.0, 3.0, None], ["up"] * 4)
    assert clf.classify_at(dts[3], ["M"]) == WARMUP


def test_warmup_when_trend_undefined():
    cfg = MarketStateConfig(percentile_window=2)
    clf = _blank(cfg)
    dts = _dates(4)
    _seed(clf, ["M"], dts, [1.0, 2.0, 3.0, 4.0], ["up", "up", "up", None])
    assert clf.classify_at(dts[3], ["M"]) == WARMUP


def test_warmup_when_reference_window_too_short():
    cfg = MarketStateConfig(percentile_window=4)
    clf = _blank(cfg)
    dts = _dates(3)
    _seed(clf, ["M"], dts, [10.0, 20.0, 30.0], ["up"] * 3)
    assert clf.classify_at(dts[2], ["M"]) == WARMUP


def test_warmup_when_reference_has_none():
    cfg = MarketStateConfig(percentile_window=3)
    clf = _blank(cfg)
    dts = _dates(5)
    # reference for idx=4 incl current = rvol[2:5] = [None, 30, 25] -> gap -> warmup.
    _seed(clf, ["M"], dts, [5.0, 10.0, None, 30.0, 25.0], ["up"] * 5)
    assert clf.classify_at(dts[4], ["M"]) == WARMUP


def test_warmup_on_short_real_price_series():
    cfg = MarketStateConfig(vol_window=21, percentile_window=504, trend_window=200)
    dts = _dates(10)
    prices = [100.0 * (1.01 ** i) for i in range(10)]
    clf = MarketStateClassifier(_panel_from_path("M", dts, prices), cfg)
    assert clf.classify_at(dts[-1], ["M"]) == WARMUP


def test_date_before_range_is_warmup():
    cfg = MarketStateConfig(percentile_window=2)
    clf = _blank(cfg)
    dts = _dates(3)
    _seed(clf, ["M"], dts, [1.0, 2.0, 3.0], ["up"] * 3)
    assert clf.classify_at(dts[0] - timedelta(days=5), ["M"]) == WARMUP


# ---------------------------------------------------------------------------
# Output is always in the allowed set; classify(observation) works
# ---------------------------------------------------------------------------

def test_output_never_outside_allowed_set():
    import random

    rng = random.Random(42)
    cfg = MarketStateConfig(vol_window=21, percentile_window=120, trend_window=50)
    n = 400
    dts = _dates(n)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + rng.uniform(-0.03, 0.03)))
    clf = MarketStateClassifier(_panel_from_path("M", dts, prices), cfg)
    allowed = set(MARKET_STATES) | {WARMUP}
    for d in dts:
        assert clf.classify_at(d, ["M"]) in allowed


def test_classify_from_observation():
    from datetime import datetime

    from trading_agent.contracts import Bar, Observation

    cfg = MarketStateConfig(percentile_window=3, stress_percentile=0.50)
    clf = _blank(cfg)
    dts = _dates(5)
    # reference incl current for idx=4 = rvol[2:5] = [30,40,50], today=50 -> pct=1.0 -> stress.
    _seed(clf, ["M"], dts, [10.0, 20.0, 30.0, 40.0, 50.0], ["down"] * 5)

    bar = Bar(open=1, high=1, low=1, close=1, adjusted_close=1, volume=1)
    obs = Observation(
        decision_date=dts[4],
        decision_time=datetime(2021, 1, 8, 9, 30),
        availability_time=datetime(2021, 1, 8, 9, 30),
        bars={"M": bar},
        tickers=["M"],
    )
    assert clf.classify(obs) == "down_stress"


# ---------------------------------------------------------------------------
# market_state vocabulary is consistent with the memory schema
# ---------------------------------------------------------------------------

def test_market_states_match_memory_schema():
    from trading_agent.memory.expert_weights import MARKET_STATES as MEM_STATES

    assert MARKET_STATES == MEM_STATES


def test_vol_quartile_mode_grades_four_buckets_without_trend():
    # Robustness variant: volatility-only four-state label. The RV percentile is graded
    # into quartiles and mapped to the four state slots; no trend is required (trend seeded
    # as None throughout), which is what makes it test the trend axis directly.
    for today, expected in [(5.0, MARKET_STATES[0]), (15.0, MARKET_STATES[1]),
                            (25.0, MARKET_STATES[2]), (50.0, MARKET_STATES[3])]:
        clf = _blank(MarketStateConfig(percentile_window=5, mode="vol_quartile"))
        dts = _dates(5)
        _seed(clf, ["M"], dts, [10.0, 20.0, 30.0, 40.0, today], [None] * 5)
        assert clf.classify_at(dts[4], ["M"]) == expected


def test_trend_only_mode_gives_two_states_dropping_risk():
    # Coarsening variant: a two-state up/down label. Only the trend side selects the
    # (reused) state slot; the risk level does not enter, which halves the state count so
    # each (expert, state) cell sees more episodes.
    for side, expected in [("up", MARKET_STATES[0]), ("down", MARKET_STATES[2])]:
        clf = _blank(MarketStateConfig(percentile_window=3, mode="trend_only"))
        dts = _dates(3)
        _seed(clf, ["M"], dts, [10.0, 20.0, 30.0], [side] * 3)
        assert clf.classify_at(dts[2], ["M"]) == expected
    # The risk level genuinely does not move the label: max-stress rvol under an up trend
    # still returns the "up" slot, not a stress state.
    clf = _blank(MarketStateConfig(percentile_window=3, mode="trend_only"))
    dts = _dates(3)
    _seed(clf, ["M"], dts, [99.0, 99.0, 99.0], ["up"] * 3)
    assert clf.classify_at(dts[2], ["M"]) == MARKET_STATES[0]


def test_risk_binary_mode_gives_two_states_dropping_trend():
    # Coarsest volatility partition: calm / stress on the median RV threshold, no trend axis.
    # Low RV -> calm slot, high RV -> stress slot, regardless of trend (seeded None).
    for today, expected in [(5.0, MARKET_STATES[0]), (50.0, MARKET_STATES[3])]:
        clf = _blank(MarketStateConfig(percentile_window=5, stress_percentile=0.50, mode="risk_binary"))
        dts = _dates(5)
        _seed(clf, ["M"], dts, [10.0, 20.0, 30.0, 40.0, today], [None] * 5)
        assert clf.classify_at(dts[4], ["M"]) == expected
