"""IB Gateway paper-trading bridge tests."""

from __future__ import annotations

import pytest

from trading_agent.contracts import ALL_AVAILABLE, OrderIntent


def test_order_preview_converts_order_intents_to_ib_contracts_and_actions():
    from trading_agent.paper_trading import build_order_previews

    previews = build_order_previews(
        [
            OrderIntent(ticker="BRK-B", side="buy", quantity=3),
            OrderIntent(ticker="MSFT", side="sell", quantity=ALL_AVAILABLE),
        ],
        allow_all_available_sell=True,
    )

    assert [p["symbol"] for p in previews] == ["BRK-B", "MSFT"]
    assert [p["ib_symbol"] for p in previews] == ["BRK B", "MSFT"]
    assert [p["action"] for p in previews] == ["BUY", "SELL"]
    assert [p["quantity"] for p in previews] == [3, "ALL"]
    assert all(p["order_type"] == "MKT" for p in previews)


def test_order_preview_rejects_all_available_buy():
    from trading_agent.paper_trading import build_order_previews

    with pytest.raises(ValueError, match="ALL_AVAILABLE buy"):
        build_order_previews([OrderIntent(ticker="AAPL", side="buy", quantity=ALL_AVAILABLE)])


def test_paper_cli_dry_run_can_emit_order_preview_without_gateway(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_preview.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL,MSFT",
        "--order", "AAPL:buy:2",
        "--order", "MSFT:sell:1",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = output.read_text(encoding="utf-8")
    assert '"gateway_connected": false' in payload
    assert '"orders_submitted": 0' in payload
    assert '"ib_symbol": "AAPL"' in payload
    assert '"action": "SELL"' in payload


def test_paper_manifest_records_preview_and_submission_audit_counts():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(host="127.0.0.1", port=4002, client_id=1),
        symbols=["AAPL", "MSFT"],
        order_intents=[
            OrderIntent(ticker="AAPL", side="buy", quantity=2),
            OrderIntent(ticker="MSFT", side="sell", quantity=1),
        ],
        dry_run=False,
        execute_orders=True,
        gateway_connected=True,
        latest_bars={"AAPL": {"rows": 42}},
        orders_submitted=1,
    )

    assert payload["order_preview_count"] == 2
    assert payload["orders_submitted"] == 1
    assert payload["orders_pending_submission"] == 1
    assert payload["submission_coverage"] == 0.5


def test_paper_manifest_records_nested_deployment_audit_metadata():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(host="127.0.0.1", port=4002, client_id=17),
        symbols=["aapl"],
        order_intents=[OrderIntent(ticker="AAPL", side="buy", quantity=2)],
        dry_run=True,
        execute_orders=False,
        gateway_connected=False,
        latest_bars={"AAPL": {"rows": 5}},
        orders_submitted=0,
        require_memory=True,
    )

    assert payload["system"] == "four_expert_reflection"
    assert payload["manifest"] == {
        "mode": "paper",
        "system": "four_expert_reflection",
        "schema_version": "trading-agent-report/v1",
        "launcher": "trading-agent paper",
        "host": "127.0.0.1",
        "port": 4002,
        "client_id": 17,
        "symbols": ["AAPL"],
        "dry_run": True,
        "execute_orders": False,
        "memory_required": True,
        "gateway_connected": False,
        "connection_smoke": False,
        "run_intent": "dry_run",
        "common_paper_port": True,
        "allowed_paper_ports": [4002, 7497],
        "non_paper_port_override": False,
        "order_preview_count": 1,
        "orders_submitted": 0,
        "orders_pending_submission": 1,
        "submission_coverage": 0.0,
        "ib_connector_source": "data_collection.ib_connector",
        "external_validation_steps": [
            "connection_smoke",
            "memory_dry_run",
            "paper_order_submission",
        ],
        "ib_paper_validation": {
            "requires_external_gateway": True,
            "status": "dry_run_only",
            "connection_verified": False,
            "actual_order_submission": False,
        },
    }


def test_paper_manifest_flags_uncommon_ib_port():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(host="127.0.0.1", port=4001, client_id=17),
        symbols=["AAPL"],
        order_intents=[],
        dry_run=False,
        execute_orders=False,
    )

    assert payload["manifest"]["common_paper_port"] is False
    assert payload["manifest"]["allowed_paper_ports"] == [4002, 7497]


def test_paper_manifest_records_non_paper_port_override():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(host="127.0.0.1", port=4001, client_id=17),
        symbols=["AAPL"],
        order_intents=[OrderIntent(ticker="AAPL", side="buy", quantity=1)],
        dry_run=False,
        execute_orders=True,
        non_paper_port_override=True,
    )

    assert payload["manifest"]["common_paper_port"] is False
    assert payload["manifest"]["non_paper_port_override"] is True
    assert payload["manifest"]["run_intent"] == "paper_order_submission"


def test_paper_manifest_records_live_submission_validation_status():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(host="127.0.0.1", port=4002, client_id=17),
        symbols=["AAPL"],
        order_intents=[OrderIntent(ticker="AAPL", side="buy", quantity=1)],
        dry_run=False,
        execute_orders=True,
        gateway_connected=True,
        orders_submitted=1,
    )

    assert payload["manifest"]["ib_paper_validation"] == {
        "requires_external_gateway": True,
        "status": "paper_order_submitted",
        "connection_verified": True,
        "actual_order_submission": True,
    }


def test_paper_manifest_records_memory_required_flag():
    from trading_agent.paper_trading import IBGatewayConfig, build_paper_manifest

    payload = build_paper_manifest(
        config=IBGatewayConfig(),
        symbols=["AAPL"],
        order_intents=[],
        dry_run=True,
        execute_orders=False,
        require_memory=True,
    )

    assert payload["memory_required"] is True


def test_submit_order_previews_places_orders_with_injected_ib_client():
    from trading_agent.paper_trading import submit_order_previews

    class FakeContract:
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency

    class FakeOrder:
        def __init__(self, action, totalQuantity, orderType="MKT", tif="DAY"):
            self.action = action
            self.totalQuantity = totalQuantity
            self.orderType = orderType
            self.tif = tif

    class FakeStatus:
        orderId = 42
        status = "Submitted"

    class FakeTrade:
        orderStatus = FakeStatus()

    class FakeIB:
        def __init__(self):
            self.calls = []

        def placeOrder(self, contract, order):
            self.calls.append((contract, order))
            return FakeTrade()

    ib = FakeIB()
    reports = submit_order_previews(
        ib,
        [
            {
                "symbol": "AAPL",
                "ib_symbol": "AAPL",
                "action": "BUY",
                "quantity": 2,
                "order_type": "MKT",
                "time_in_force": "DAY",
            }
        ],
        stock_factory=FakeContract,
        market_order_factory=FakeOrder,
    )

    assert len(ib.calls) == 1
    contract, order = ib.calls[0]
    assert contract.symbol == "AAPL"
    assert contract.exchange == "SMART"
    assert order.action == "BUY"
    assert order.totalQuantity == 2
    assert reports == [{
        "symbol": "AAPL",
        "ib_symbol": "AAPL",
        "action": "BUY",
        "quantity": 2,
        "order_id": 42,
        "status": "Submitted",
    }]


def test_account_cash_prefers_available_funds_then_net_liquidation():
    from trading_agent.paper_trading import account_cash_from_summary

    class Row:
        def __init__(self, tag, value, currency="USD"):
            self.tag = tag
            self.value = value
            self.currency = currency

    assert account_cash_from_summary([
        Row("NetLiquidation", "25000", "USD"),
        Row("AvailableFunds", "12000", "USD"),
    ]) == 12000.0
    assert account_cash_from_summary([
        Row("NetLiquidation", "25000", "USD"),
    ]) == 25000.0


def test_auto_decide_order_intents_uses_equal_cash_budget_and_latest_close():
    from trading_agent.paper_trading import auto_decide_order_intents

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __init__(self, close):
            self._close = close

        def __getitem__(self, key):
            if key == "close":
                return Series(self._close)
            raise KeyError(key)

    intents = auto_decide_order_intents(
        ["AAPL", "MSFT"],
        {"AAPL": Frame(100), "MSFT": Frame(50)},
        paper_cash=1000,
    )

    assert [(i.ticker, i.side, i.quantity) for i in intents] == [
        ("AAPL", "buy", 5),
        ("MSFT", "buy", 10),
    ]


def test_auto_decide_order_intents_scales_cash_by_deployment_fraction():
    from trading_agent.paper_trading import auto_decide_order_intents

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __getitem__(self, key):
            if key == "close":
                return Series(100)
            raise KeyError(key)

    intents = auto_decide_order_intents(
        ["AAPL"],
        {"AAPL": Frame()},
        paper_cash=1000,
        deployment_fraction=0.5,
    )

    assert [(i.ticker, i.side, i.quantity) for i in intents] == [("AAPL", "buy", 5)]


def test_auto_decide_order_intents_caps_each_order_value():
    from trading_agent.paper_trading import auto_decide_order_intents

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __getitem__(self, key):
            if key == "close":
                return Series(100)
            raise KeyError(key)

    intents = auto_decide_order_intents(
        ["AAPL"],
        {"AAPL": Frame()},
        paper_cash=100_000,
        max_order_value=1_000,
    )

    assert [(i.ticker, i.side, i.quantity) for i in intents] == [("AAPL", "buy", 10)]
