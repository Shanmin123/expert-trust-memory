"""Positions serialisation, tested without a Gateway.

The account, orders, snapshot and history payloads in ``ib_paper_data`` were verified
against a live paper Gateway; positions was the one holding view the module did not expose.
It is added in the same idiom -- defensive attribute reads, a counted payload, read-only --
and these tests pin the shape so the notebook can rely on it before anyone connects.
"""
from __future__ import annotations

from types import SimpleNamespace

from data_collection.ib_paper_data import (
    portfolio_item_to_dict,
    position_to_dict,
    positions_payload,
)


def _contract(symbol="AAPL", con_id=265598):
    return SimpleNamespace(symbol=symbol, secType="STK", exchange="SMART",
                           primaryExchange="NASDAQ", currency="USD", conId=con_id)


def _position(symbol="AAPL", size=100.0, avg_cost=190.5):
    return SimpleNamespace(account="TESTACCT", contract=_contract(symbol),
                           position=size, avgCost=avg_cost)


def _portfolio_item(symbol="AAPL", size=100.0):
    return SimpleNamespace(account="TESTACCT", contract=_contract(symbol),
                           position=size, marketPrice=201.25, marketValue=20125.0,
                           averageCost=190.5, unrealizedPNL=1075.0, realizedPNL=0.0)


class _IB:
    def __init__(self, positions=(), portfolio=()):
        self._positions, self._portfolio = list(positions), list(portfolio)

    def positions(self):
        return self._positions

    def portfolio(self):
        return self._portfolio


def test_a_held_position_serialises_size_and_cost():
    row = position_to_dict(_position())
    assert row["position"] == 100.0
    assert row["avgCost"] == 190.5
    assert row["contract"]["symbol"] == "AAPL"
    assert row["contract"]["conId"] == 265598


def test_a_portfolio_row_carries_market_value_and_unrealised_pnl():
    row = portfolio_item_to_dict(_portfolio_item())
    assert row["marketValue"] == 20125.0
    assert row["unrealizedPNL"] == 1075.0
    assert row["marketPrice"] == 201.25


def test_an_empty_account_is_a_zero_count_not_an_error():
    payload = positions_payload(_IB())
    assert payload["status"] == "ok"
    assert payload["position_count"] == 0
    assert payload["portfolio_count"] == 0
    assert payload["positions"] == [] and payload["portfolio"] == []


def test_both_views_are_reported_and_counted():
    payload = positions_payload(_IB(
        positions=[_position("AAPL"), _position("MSFT", 50.0, 410.0)],
        portfolio=[_portfolio_item("AAPL")],
    ))
    assert payload["position_count"] == 2
    assert payload["portfolio_count"] == 1
    assert [p["contract"]["symbol"] for p in payload["positions"]] == ["AAPL", "MSFT"]


def test_a_client_missing_either_method_does_not_raise():
    """Older ib_async builds may not expose both; the payload degrades rather than fails."""
    class _OnlyPositions:
        def positions(self):
            return [_position()]

    payload = positions_payload(_OnlyPositions())
    assert payload["position_count"] == 1
    assert payload["portfolio_count"] == 0


def test_a_short_position_keeps_its_sign():
    row = position_to_dict(_position(size=-40.0))
    assert row["position"] == -40.0, "a short must not be reported as flat or positive"


def test_the_payload_declares_itself_read_only():
    payload = positions_payload(_IB())
    assert payload["capability"] == "paper_positions"
    assert any("read-only" in n.lower() for n in payload["notes"])
