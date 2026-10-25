"""IB history requests must not silently come back empty.

Two failures found on the live paper gateway, 2026-07-29, both of which would have
produced ordinary-looking numbers from a run that was not doing what it claimed:

1. A day-unit duration much past a year times out and returns an EMPTY FRAME rather
   than raising. Measured: 1 Y -> 251 rows, 2 Y -> 501, 3 Y -> 751, 792 D -> 0. The
   frozen windows compute 792, so every request the forward run made came back empty.

2. An empty proxy frame makes the classifier fall back to the traded stock's own bars,
   so the label describing "the market" would be computed from the instrument being
   traded. The only trace was a source string in the manifest.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trading_agent import cli


def _frame(n: int = 5):
    dates = pd.bdate_range("2024-01-02", periods=n)
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame({
        "date": [d.date().isoformat() for d in dates],
        "open": closes, "high": closes, "low": closes,
        "close": closes, "adjusted_close": closes,
        "volume": [1_000_000] * n,
    })


def _proxy_args(**over) -> SimpleNamespace:
    base = dict(
        paper_market_state="auto",
        market_proxy_symbol="SPY",
        lookback_days=760,
        paper_vol_window=21,
        paper_percentile_window=504,
        paper_trend_window=200,
        require_memory=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _StubIB:
    """Returns whatever the test tells it to, the way IB actually behaves."""

    def __init__(self, result):
        self._result = result
        self.durations: list[str] = []

    def reqHistoricalData(self, *a, **k):  # pragma: no cover - not the seam used
        raise AssertionError("tests patch request_daily_bars, not the raw client")


def test_a_year_or_less_is_asked_for_in_days():
    assert cli._paper_history_duration_arg(7) == "7 D"
    assert cli._paper_history_duration_arg(365) == "365 D"


def test_beyond_a_year_is_asked_for_in_years():
    # 792 is what the frozen windows compute, and "792 D" is the request that returns
    # nothing. It must become a year-unit duration.
    assert cli._paper_history_duration_arg(792) == "3 Y"
    assert cli._paper_history_duration_arg(366) == "2 Y"


def test_the_year_count_rounds_up_so_the_window_is_never_short():
    # 500 days needs more than one year of history; asking for "1 Y" would silently
    # truncate the percentile window.
    assert cli._paper_history_duration_arg(500) == "2 Y"
    assert cli._paper_history_duration_arg(730) == "2 Y"
    assert cli._paper_history_duration_arg(731) == "3 Y"


def test_frozen_windows_still_produce_a_duration_ib_accepts():
    args = _proxy_args()
    days = cli._paper_history_duration_days(args)
    assert days >= 527, "must cover the 504 percentile window plus the vol window"
    assert cli._paper_history_duration_arg(days).endswith(" Y")


def test_proxy_fetch_reports_an_empty_frame_as_a_failure(monkeypatch):
    """An empty frame and a raised error are the same outcome and must look the same."""
    monkeypatch.setattr(cli, "request_daily_bars", lambda *a, **k: pd.DataFrame())
    assert cli._fetch_market_proxy_frame(_StubIB(None), _proxy_args(), 792) is None


def test_proxy_fetch_asks_in_years_for_the_frozen_window(monkeypatch):
    seen = {}

    def _spy(ib, symbol, duration=None, **k):
        seen["duration"] = duration
        return _frame()

    monkeypatch.setattr(cli, "request_daily_bars", _spy)
    frame = cli._fetch_market_proxy_frame(_StubIB(None), _proxy_args(), 792)
    assert frame is not None
    assert seen["duration"] == "3 Y", "a day-unit request this large returns nothing"


def test_proxy_fetch_passes_a_real_frame_through(monkeypatch):
    monkeypatch.setattr(cli, "request_daily_bars", lambda *a, **k: _frame(3))
    got = cli._fetch_market_proxy_frame(_StubIB(None), _proxy_args(), 30)
    assert got is not None and len(got) == 3


def test_a_manual_state_override_skips_the_proxy(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not fetch a proxy when the state is overridden")

    monkeypatch.setattr(cli, "request_daily_bars", _boom)
    args = _proxy_args(paper_market_state="up_stress")
    assert cli._fetch_market_proxy_frame(_StubIB(None), args, 792) is None


# The helper above was written, tested and then not used by the branch that submits
# orders: it asked for f"{history_days} D" directly while the dry-run branch converted.
# Testing the helper could not catch that, so these pin the CALL SITES in the source.

def _paper_source() -> str:
    from pathlib import Path
    return Path(cli.__file__).read_text(encoding="utf-8")


def test_no_call_site_builds_a_day_unit_duration_from_history_days():
    import re
    bad = re.findall(r'duration\s*=\s*f?"\{?history_days\}?\s*D"', _paper_source())
    assert not bad, (
        "a bar request is building a day-unit duration from history_days; past a year "
        f"IB returns an empty frame instead of an error: {bad}"
    )


def test_every_bar_request_goes_through_the_duration_helper():
    import re
    src = _paper_source()
    calls = re.findall(r"request_daily_bars\((.*?)\)", src, re.S)
    symbol_calls = [c for c in calls if "duration" in c]
    assert symbol_calls, "expected the paper path to request daily bars"
    for call in symbol_calls:
        assert "_paper_history_duration_arg" in call or "duration=duration" in call, (
            f"bar request bypasses the duration helper: {call.strip()[:120]}"
        )


def test_both_branches_refuse_an_empty_history():
    """Wording is pinned in test_paper_branch_parity; this pins the count."""
    src = _paper_source()
    assert src.count("IB returned no bars for") == 2, (
        "both the preview branch and the order-submitting branch must reject an empty "
        "frame; an empty history silently produces a decision from no data"
    )
