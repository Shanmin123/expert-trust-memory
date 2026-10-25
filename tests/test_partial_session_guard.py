"""A decision may not be built on a session that has not closed.

``request_daily_bars`` asks IB with ``endDateTime=""``, so during regular hours the last
daily bar is today's PARTIAL bar. The protocol decides on a completed close and fills at
the next open, so a mid-session run would read an intraday price as a close and nothing
in the output would say so. Found 2026-08-03, before the forward run started.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_agent import cli

EASTERN = timezone(timedelta(hours=-4))


def _frame(last_date: str):
    return pd.DataFrame({
        "date": ["2026-07-30", "2026-07-31", last_date],
        "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 1.0], "adjusted_close": [1.0, 1.0, 1.0],
        "volume": [1, 1, 1],
    })


def _at(monkeypatch, when: datetime):
    """Freeze the clock the guard reads."""
    real = cli.datetime

    class Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.astimezone(tz)

    monkeypatch.setattr(cli, "datetime", Frozen)


def test_a_bar_dated_today_during_the_session_is_dropped(monkeypatch):
    """Dropping it and deciding on the last completed close is the protocol.

    Refusing the run instead, as the first version did, conflated "the decision may not
    use a partial bar" with "you may not trade mid-session". Only the first is required.
    """
    _at(monkeypatch, datetime(2026, 8, 3, 11, 0, tzinfo=EASTERN))  # Monday, mid-session
    frames = {"BIIB": _frame("2026-08-03")}
    cli._reject_partial_session(frames, ["BIIB"])
    assert list(frames["BIIB"]["date"]) == ["2026-07-30", "2026-07-31"]


def test_the_same_bar_before_the_open_is_allowed(monkeypatch):
    # Running before 09:30 ET is the correct time: the last bar is a completed close.
    _at(monkeypatch, datetime(2026, 8, 3, 8, 0, tzinfo=EASTERN))
    cli._reject_partial_session({"BIIB": _frame("2026-07-31")}, ["BIIB"])


def test_after_the_close_today_s_bar_is_complete_and_allowed(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 3, 17, 0, tzinfo=EASTERN))
    cli._reject_partial_session({"BIIB": _frame("2026-08-03")}, ["BIIB"])


def test_a_weekend_run_is_allowed(monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 1, 11, 0, tzinfo=EASTERN))  # Saturday
    cli._reject_partial_session({"BIIB": _frame("2026-07-31")}, ["BIIB"])


def test_a_stale_bar_during_the_session_is_allowed(monkeypatch):
    # Mid-session, but the last bar is the previous close: that is the correct input.
    _at(monkeypatch, datetime(2026, 8, 3, 11, 0, tzinfo=EASTERN))
    cli._reject_partial_session({"BIIB": _frame("2026-07-31")}, ["BIIB"])


def test_only_the_affected_symbols_are_trimmed(monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 3, 11, 0, tzinfo=EASTERN))
    frames = {"BIIB": _frame("2026-08-03"), "ECL": _frame("2026-08-03"),
              "DHI": _frame("2026-07-31")}
    cli._reject_partial_session(frames, ["BIIB", "ECL", "DHI"])
    assert len(frames["BIIB"]) == 2 and len(frames["ECL"]) == 2
    assert len(frames["DHI"]) == 3, "a stale last bar is the correct input, keep it"
    said = capsys.readouterr().out
    assert "BIIB" in said and "ECL" in said and "DHI" not in said


def test_both_auto_decide_branches_call_the_guard():
    src = Path(cli.__file__).read_text(encoding="utf-8")
    calls = len(re.findall(r"_reject_partial_session\(latest_frames", src))
    assert calls == 2, (
        f"expected the guard in both the preview and the order branch, found {calls}"
    )
