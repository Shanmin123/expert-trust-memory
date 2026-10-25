"""The paper bridge must hold what it already owns and exit what it no longer wants.

Found 2026-08-03, the day the forward run was due to start. ``auto_decide_order_intents``
only ever emitted buys and never read the account, so over a ten-day run a stock the
committee kept calling long was bought again every day, and a stock it turned flat on was
never sold. Neither matches the backtest, whose exposure is binary and whose maintain rule
holds a position instead of re-opening it.

The four cases below are the backtest's own semantics written against the live path.
"""
from __future__ import annotations

from pathlib import Path

from trading_agent.paper_trading import auto_decide_order_intents


class _Series:
    def __init__(self, value):
        self.iloc = [value]


class _Frame:
    def __init__(self, close=100.0):
        self._close = close

    def __getitem__(self, key):
        assert key in {"close", "adjusted_close"}
        return _Series(self._close)

    def __len__(self):
        return 1


def _intents(long_symbols, held=None, cash=10_000.0, frames=None):
    frames = frames or {"AAA": _Frame(100.0), "BBB": _Frame(50.0)}
    return [(i.ticker, i.side, i.quantity) for i in auto_decide_order_intents(
        long_symbols, frames, paper_cash=cash, held=held)]


def test_a_long_call_on_an_unheld_stock_buys():
    assert _intents(["AAA"], held={}) == [("AAA", "buy", 100)]


def test_a_long_call_on_a_stock_already_held_does_nothing():
    """The maintain rule: a repeated long is a hold, not a second purchase."""
    assert _intents(["AAA"], held={"AAA": 100.0}) == []


def test_a_stock_no_longer_called_long_is_sold_to_flat():
    assert _intents([], held={"AAA": 37.0}) == [("AAA", "sell", 37)]


def test_a_stock_neither_held_nor_called_long_produces_nothing():
    assert _intents([], held={}) == []


def test_the_exit_sells_exactly_what_is_held():
    assert _intents(["BBB"], held={"AAA": 5.0, "BBB": 2.0}) == [("AAA", "sell", 5)]


def test_a_short_position_is_left_alone():
    """The system is long-or-flat; it must not act on a position it did not open."""
    assert _intents([], held={"AAA": -10.0}) == []


def test_without_position_data_the_old_behaviour_is_unchanged():
    """held=None means the account was not read; buy as before rather than guess."""
    assert _intents(["AAA", "BBB"], held=None) == [
        ("AAA", "buy", 50), ("BBB", "buy", 100)]


def test_budget_is_split_only_over_the_stocks_actually_bought():
    """A held stock must not consume budget it is not going to spend."""
    got = _intents(["AAA", "BBB"], held={"AAA": 10.0})
    assert got == [("BBB", "buy", 200)], got


# --- the CLI must actually read the account and pass it through -------------------

from types import SimpleNamespace  # noqa: E402

from trading_agent import cli  # noqa: E402


class _Row:
    def __init__(self, symbol, position):
        self.contract = SimpleNamespace(symbol=symbol)
        self.position = position


class _IB:
    def __init__(self, rows=None, boom=False):
        self._rows = rows or []
        self._boom = boom

    def positions(self):
        if self._boom:
            raise RuntimeError("no account")
        return self._rows


def test_held_quantities_reads_only_the_traded_symbols():
    ib = _IB([_Row("AAA", 5.0), _Row("ZZZ", 99.0)])
    assert cli._held_quantities(ib, ["AAA", "BBB"]) == {"AAA": 5.0}


def test_held_quantities_sums_split_lots():
    ib = _IB([_Row("AAA", 5.0), _Row("AAA", 3.0)])
    assert cli._held_quantities(ib, ["AAA"]) == {"AAA": 8.0}


def test_an_unreadable_account_is_none_not_empty():
    """None means 'not read'; {} means 'read, holds nothing'. They must not collapse."""
    assert cli._held_quantities(_IB(boom=True), ["AAA"]) is None
    assert cli._held_quantities(_IB([]), ["AAA"]) == {}


def test_every_order_call_site_passes_the_account():
    """The bug was one call site, so pin both rather than the helper alone."""
    import re
    src = Path(cli.__file__).read_text(encoding="utf-8")
    calls = re.findall(r"auto_decide_order_intents\((.*?)\n\s*\)\)", src, re.S)
    assert len(calls) == 2, f"expected two order call sites, found {len(calls)}"
    for call in calls:
        assert "held=held" in call, f"call site does not read the account: {call[:120]}"
