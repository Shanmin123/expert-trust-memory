"""PIT Alpha158 feature recomputation: correctness vs hand calc + causality."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean, stdev

from trading_agent._bootstrap import FinsaberDataset
from trading_agent.alpha.alpha158_pit import compute_alpha158_feature_frame
from trading_agent.alpha.alpha158_recipes import REGISTERED_ALPHA158_FEATURES


def _bar(c):
    # adjusted_* == raw so the replica uses these prices directly (factor 1.0)
    return {
        "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
        "adjusted_open": c, "adjusted_high": c * 1.01, "adjusted_low": c * 0.99,
        "adjusted_close": c, "volume": 1_000_000 + int(c) * 10,
    }


def _dataset(n=70, start=date(2022, 1, 3)):
    data = {}
    for i in range(n):
        d = start + timedelta(days=i)
        data[d] = {"price": {"AAA": _bar(100.0 + i)}}  # strictly increasing closes
    return FinsaberDataset(data=data), [start + timedelta(days=i) for i in range(n)]


def _val(frame, d, sid):
    rows = frame[(frame["date"] == d) & (frame["ticker"] == "AAA")]
    if rows.empty or sid not in rows.columns:
        return None
    v = rows.iloc[0][sid]
    return None if v != v else float(v)  # NaN -> None


def test_roc_ma_std_match_hand_calc():
    ds, dates = _dataset()
    frame = compute_alpha158_feature_frame(ds, ["AAA"], dates[0], dates[-1])
    closes = [100.0 + i for i in range(len(dates))]
    i = len(dates) - 1  # last date, all windows warm
    d = dates[i]
    # ROC5 = close[t-5]/close[t]
    assert abs(_val(frame, d, "alpha158_roc5") - closes[i - 5] / closes[i]) < 1e-9
    # ROC20 = close[t-20]/close[t]
    assert abs(_val(frame, d, "alpha158_roc20") - closes[i - 20] / closes[i]) < 1e-9
    # MA5 = mean(last 5 closes)/close[t]
    assert abs(_val(frame, d, "alpha158_ma5") - mean(closes[i - 4 : i + 1]) / closes[i]) < 1e-9
    # STD20 = sample std(last 20)/close[t]
    assert abs(_val(frame, d, "alpha158_std20") - stdev(closes[i - 19 : i + 1]) / closes[i]) < 1e-9
    # MAX20 uses high=close*1.01; MIN20 uses low=close*0.99 (increasing series)
    assert abs(_val(frame, d, "alpha158_max20") - (closes[i] * 1.01) / closes[i]) < 1e-9
    assert abs(_val(frame, d, "alpha158_min20") - (closes[i - 19] * 0.99) / closes[i]) < 1e-9


def test_std60_needs_warmup_and_is_none_early():
    ds, dates = _dataset(n=70)
    frame = compute_alpha158_feature_frame(ds, ["AAA"], dates[0], dates[-1])
    # STD60 needs 60 bars: undefined at index 30, defined at the last index (>=60 bars)
    assert _val(frame, dates[30], "alpha158_std60") is None
    assert _val(frame, dates[-1], "alpha158_std60") is not None


def test_causality_value_independent_of_future_bars():
    ds_full, dates = _dataset(n=70)
    t = dates[40]
    full = compute_alpha158_feature_frame(ds_full, ["AAA"], dates[0], dates[-1])
    # truncate the window at t: feature at t must be identical (no future bars used)
    trunc = compute_alpha158_feature_frame(ds_full, ["AAA"], dates[0], t)
    for sid in [f.strategy_id for f in REGISTERED_ALPHA158_FEATURES]:
        a, b = _val(full, t, sid), _val(trunc, t, sid)
        assert (a is None and b is None) or abs(a - b) < 1e-12, sid
