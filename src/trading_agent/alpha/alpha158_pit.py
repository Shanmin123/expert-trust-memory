"""Point-in-time recomputation of the 11 registered Alpha158 features from FINSABER OHLCV.

Replicates the Qlib Alpha158DL feature DEFINITIONS (research_refs/qlib/qlib/contrib/data/loader.py,
``Alpha158DL.get_feature_config`` lines ~149-274) in this project's OWN causal framework — NO qlib
runtime import and NO qlib-computed values: the definitions are recomputed in our own point-in-time
framework and cited to their source. Each feature at date
t uses ONLY bars with date <= t (trailing windows + backward Ref), so the precomputed frame is
leakage-clean and a per-date lookup is point-in-time.

Qlib definitions replicated (window d parsed from each REGISTERED_ALPHA158_FEATURES qlib_name):
  ROCd  = Ref($close, d) / $close                               # close d bars ago / close today
  MAd   = Mean($close, d) / $close
  STDd  = Std($close, d) / $close                               # sample std, ddof=1 (pandas default)
  MAXd  = Max($high, d) / $close
  MINd  = Min($low, d) / $close
  RSVd  = ($close - Min($low, d)) / (Max($high, d) - Min($low, d) + 1e-12)
  VMAd  = Mean($volume, d) / ($volume + 1e-12)
  VSTDd = Std($volume, d) / ($volume + 1e-12)
Rolling windows are size-d INCLUSIVE of the current bar (qlib convention). Prices are ADJUSTED
(adjusted_close/high/low) for split/dividend consistency; $volume is raw.
"""

from __future__ import annotations

import re
from datetime import date as Date
from statistics import mean, stdev
from typing import Optional, Sequence

import pandas as pd

from trading_agent.alpha.alpha158_recipes import Alpha158FeatureSpec, REGISTERED_ALPHA158_FEATURES
from trading_agent.alpha.alpha158_strategy import FrameAlpha158FeatureProvider
from trading_agent.market import _to_date

_WIN_RE = re.compile(r"(\d+)$")
_FRAME_COLUMNS_BASE = ["date", "ticker"]


def _family_window(qlib_name: str) -> tuple[str, int]:
    """('ROC5') -> ('ROC', 5). Family = qlib_name without the trailing window integer."""
    m = _WIN_RE.search(qlib_name)
    if not m:
        return qlib_name, 0
    return qlib_name[: m.start()], int(m.group(1))


def _series_by_ticker(loader, tickers: Sequence[str], start: Date, end: Date):
    """{ticker: [(date, adj_close, adj_high, adj_low, volume), ...]} sorted ascending by date."""
    wanted = set(tickers)
    rows: dict[str, list] = {tk: [] for tk in tickers}
    for d in [_to_date(x) for x in loader.get_date_range()]:
        if d < _to_date(start) or d > _to_date(end):
            continue
        price = loader.get_data_by_date(d).get("price", {}) or {}
        for tk, bar in price.items():
            if tk not in wanted or not isinstance(bar, dict):
                continue
            c = bar.get("adjusted_close", bar.get("close"))
            if c is None or float(c) <= 0:
                continue
            h = bar.get("adjusted_high", bar.get("high"))
            low = bar.get("adjusted_low", bar.get("low"))
            v = bar.get("volume")
            rows[tk].append((
                d,
                float(c),
                None if h is None else float(h),
                None if low is None else float(low),
                None if v is None else float(v),
            ))
    for tk in rows:
        rows[tk].sort(key=lambda r: r[0])
    return rows


def _feature_value(fam: str, win: int, i: int, closes, highs, lows, vols) -> Optional[float]:
    """One Alpha158 feature at series index i (date t). Returns None if history is insufficient."""
    if win <= 0:
        return None
    c_t = closes[i]
    if fam == "ROC":
        j = i - win  # Ref($close, win) = close exactly `win` bars before t
        if j < 0:
            return None
        return closes[j] / c_t
    lo = i - win + 1  # inclusive size-`win` window ending at t
    if lo < 0:
        return None
    if fam == "MA":
        return mean(closes[lo : i + 1]) / c_t
    if fam == "STD":
        w = closes[lo : i + 1]
        return (stdev(w) if len(w) > 1 else 0.0) / c_t
    if fam == "MAX":
        w = highs[lo : i + 1]
        if any(x is None for x in w):
            return None
        return max(w) / c_t
    if fam == "MIN":
        w = lows[lo : i + 1]
        if any(x is None for x in w):
            return None
        return min(w) / c_t
    if fam == "RSV":
        hw = highs[lo : i + 1]
        lw = lows[lo : i + 1]
        if any(x is None for x in hw) or any(x is None for x in lw):
            return None
        mn, mx = min(lw), max(hw)
        return (c_t - mn) / (mx - mn + 1e-12)
    if fam == "VMA":
        w = vols[lo : i + 1]
        v_t = vols[i]
        if v_t is None or any(x is None for x in w):
            return None
        return mean(w) / (v_t + 1e-12)
    if fam == "VSTD":
        w = vols[lo : i + 1]
        v_t = vols[i]
        if v_t is None or any(x is None for x in w):
            return None
        return (stdev(w) if len(w) > 1 else 0.0) / (v_t + 1e-12)
    return None


def compute_alpha158_feature_frame(
    loader,
    tickers: Sequence[str],
    start: Date,
    end: Date,
    features: Sequence[Alpha158FeatureSpec] = REGISTERED_ALPHA158_FEATURES,
) -> pd.DataFrame:
    """Long frame [date, ticker, <strategy_id>...] of PIT Alpha158 feature values over [start, end]."""
    series = _series_by_ticker(loader, tickers, start, end)
    specs = [(f.strategy_id, *_family_window(f.qlib_name)) for f in features]
    out_rows: list[dict] = []
    for tk, rec in series.items():
        if not rec:
            continue
        dates = [r[0] for r in rec]
        closes = [r[1] for r in rec]
        highs = [r[2] for r in rec]
        lows = [r[3] for r in rec]
        vols = [r[4] for r in rec]
        for i, d in enumerate(dates):
            row = {"date": d, "ticker": tk}
            has_val = False
            for sid, fam, win in specs:
                v = _feature_value(fam, win, i, closes, highs, lows, vols)
                if v is not None:
                    row[sid] = v
                    has_val = True
            if has_val:
                out_rows.append(row)
    columns = _FRAME_COLUMNS_BASE + [f.strategy_id for f in features]
    if not out_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out_rows)


def pit_feature_provider(
    loader,
    tickers: Sequence[str],
    start: Date,
    end: Date,
    features: Sequence[Alpha158FeatureSpec] = REGISTERED_ALPHA158_FEATURES,
) -> FrameAlpha158FeatureProvider:
    """Build a FrameAlpha158FeatureProvider backed by PIT-recomputed Alpha158 features."""
    frame = compute_alpha158_feature_frame(loader, tickers, start, end, features)
    return FrameAlpha158FeatureProvider(frame=frame)


__all__ = ["compute_alpha158_feature_frame", "pit_feature_provider"]
