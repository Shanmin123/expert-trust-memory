"""The committee read may not treat an unfinished session as a close.

Found live on 2026-08-03 at 09:50 ET, twenty minutes into the first day of the forward
run. IB answered with a bar dated that day carrying volume 30,155 against the previous
session's 782,233, and the read used it as a close. The order path already refused this
case; this script did not, because it is a separate entry point.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import live_analysis  # noqa: E402

EASTERN = timezone(timedelta(hours=-4))


def _frame(*dates):
    return pd.DataFrame({
        "date": list(dates),
        "open": [1.0] * len(dates), "high": [1.0] * len(dates),
        "low": [1.0] * len(dates), "close": [1.0] * len(dates),
        "adjusted_close": [1.0] * len(dates), "volume": [1] * len(dates),
    })


def _at(monkeypatch, when):
    real = live_analysis.DateTime

    class Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.astimezone(tz)

    monkeypatch.setattr(live_analysis, "DateTime", Frozen)


def test_todays_bar_is_dropped_during_the_session(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 3, 9, 50, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(
        _frame("2026-07-30", "2026-07-31", "2026-08-03"))
    assert cut == "2026-08-03"
    assert list(frame["date"]) == ["2026-07-30", "2026-07-31"]


def test_nothing_is_dropped_before_the_open(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 3, 8, 0, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(
        _frame("2026-07-30", "2026-07-31"))
    assert cut is None and len(frame) == 2


def test_todays_bar_is_kept_after_the_close(monkeypatch):
    # After 16:00 ET the session is complete and its bar is a real close.
    _at(monkeypatch, datetime(2026, 8, 3, 17, 0, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(
        _frame("2026-07-31", "2026-08-03"))
    assert cut is None and len(frame) == 2


def test_a_stale_last_bar_is_kept_during_the_session(monkeypatch):
    # Mid-session but the newest bar is the previous close: that is the correct input.
    _at(monkeypatch, datetime(2026, 8, 3, 11, 0, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(
        _frame("2026-07-30", "2026-07-31"))
    assert cut is None and len(frame) == 2


def test_a_weekend_run_keeps_everything(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 1, 11, 0, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(_frame("2026-07-31"))
    assert cut is None and len(frame) == 1


def test_an_empty_frame_passes_through(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 3, 11, 0, tzinfo=EASTERN))
    frame, cut = live_analysis.drop_unfinished_session(pd.DataFrame())
    assert cut is None


def test_the_proxy_is_filtered_too():
    """SPY sets the market state, so a partial SPY bar would move the label."""
    src = (ROOT / "scripts" / "live_analysis.py").read_text(encoding="utf-8")
    block = src[src.find("for sym in list(symbols) + [PROXY]:"):]
    assert "drop_unfinished_session(frame)" in block[:400], (
        "the proxy goes through the same loop and must be filtered with the symbols"
    )
