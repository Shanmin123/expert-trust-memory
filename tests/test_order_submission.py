"""Submission must actually construct an order the broker library accepts.

Every test until now stopped at the preview. On 2026-08-03, the first time orders were
really submitted, all fifteen order-placing processes died in the factory call:
``MarketOrder`` already sets ``orderType="MKT"``, so passing it again raised
"got multiple values for keyword argument 'orderType'". The five processes that had
nothing to submit exited cleanly, which is why the failure looked selective.
"""
from __future__ import annotations

import pytest

from trading_agent.paper_trading import submit_order_previews


class _Trade:
    def __init__(self, order_id=1, status="Submitted"):
        self.order = type("O", (), {"orderId": order_id})()
        self.orderStatus = type("S", (), {"status": status})()


class _IB:
    def __init__(self):
        self.placed = []

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return _Trade(order_id=len(self.placed))


def _stock(symbol, exchange, currency):
    return {"symbol": symbol, "exchange": exchange, "currency": currency}


def _preview(**over):
    p = {"symbol": "BIIB", "ib_symbol": "BIIB", "action": "BUY", "quantity": 24,
         "order_type": "MKT", "time_in_force": "DAY"}
    p.update(over)
    return p


def test_a_market_order_is_built_with_the_real_library():
    """The regression: this used to raise TypeError before reaching the broker."""
    from ib_async import MarketOrder, Stock

    ib = _IB()
    reports = submit_order_previews(ib, [_preview()],
                                    stock_factory=Stock, market_order_factory=MarketOrder)
    assert len(ib.placed) == 1
    _contract, order = ib.placed[0]
    assert order.orderType == "MKT"
    assert order.tif == "DAY"
    assert order.action == "BUY" and order.totalQuantity == 24
    assert reports[0]["status"] == "Submitted"


def test_every_preview_reaches_the_broker():
    from ib_async import MarketOrder, Stock

    ib = _IB()
    previews = [_preview(symbol=s, ib_symbol=s, quantity=q)
                for s, q in (("BIIB", 24), ("ECL", 18), ("AZO", 1))]
    reports = submit_order_previews(ib, previews,
                                    stock_factory=Stock, market_order_factory=MarketOrder)
    assert [r["symbol"] for r in reports] == ["BIIB", "ECL", "AZO"]
    assert [o.totalQuantity for _c, o in ib.placed] == [24, 18, 1]


def test_a_sell_preview_submits_a_sell():
    from ib_async import MarketOrder, Stock

    ib = _IB()
    submit_order_previews(ib, [_preview(action="SELL", quantity=24)],
                          stock_factory=Stock, market_order_factory=MarketOrder)
    assert ib.placed[0][1].action == "SELL"


def test_a_non_market_preview_is_refused():
    from ib_async import MarketOrder, Stock

    ib = _IB()
    with pytest.raises(ValueError, match="market orders only"):
        submit_order_previews(ib, [_preview(order_type="LMT")],
                              stock_factory=Stock, market_order_factory=MarketOrder)
    assert not ib.placed


def test_an_unresolved_quantity_is_refused():
    from ib_async import MarketOrder, Stock

    ib = _IB()
    with pytest.raises(ValueError, match="ALL quantity"):
        submit_order_previews(ib, [_preview(quantity="ALL")],
                              stock_factory=Stock, market_order_factory=MarketOrder)
    assert not ib.placed
