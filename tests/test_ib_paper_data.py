from __future__ import annotations

import json
from datetime import date


class _AccountRow:
    account = "DU123"
    tag = "AvailableFunds"
    value = "100000"
    currency = "USD"


class _Order:
    orderId = 7
    permId = 77
    clientId = 1
    action = "BUY"
    orderType = "MKT"
    totalQuantity = 1
    lmtPrice = 0.0
    auxPrice = 0.0
    tif = "DAY"


class _Status:
    status = "Submitted"
    filled = 0
    remaining = 1
    avgFillPrice = 0.0
    lastFillPrice = 0.0
    whyHeld = ""


class _Contract:
    symbol = "AAPL"
    secType = "STK"
    exchange = "SMART"
    primaryExchange = "NASDAQ"
    currency = "USD"
    conId = 265598


class _Trade:
    contract = _Contract()
    order = _Order()
    orderStatus = _Status()


class _Ticker:
    bid = 199.1
    bidSize = 100
    ask = 199.2
    askSize = 100
    last = 199.15
    lastSize = 10
    close = 198.5

    def marketPrice(self):
        return self.last


class _FakeIB:
    def __init__(self):
        self.requests = []
        self.disconnected = False

    def accountSummary(self):
        self.requests.append(("accountSummary",))
        return [_AccountRow()]

    def reqAllOpenOrders(self):
        self.requests.append(("reqAllOpenOrders",))

    def sleep(self, seconds):
        self.requests.append(("sleep", seconds))

    def openTrades(self):
        return [_Trade()]

    def openOrders(self):
        return [_Order()]

    def qualifyContracts(self, contract):
        self.requests.append(("qualify", contract.symbol, contract.exchange, contract.currency))
        return [_Contract()]

    def reqMarketDataType(self, data_type):
        self.requests.append(("reqMarketDataType", data_type))

    def reqMktData(self, contract, generic_ticks, snapshot, regulatory_snapshot):
        self.requests.append(("reqMktData", contract.conId, snapshot, regulatory_snapshot))
        return _Ticker()

    def cancelMktData(self, contract):
        self.requests.append(("cancelMktData", contract.conId))

    def disconnect(self):
        self.disconnected = True


def test_account_summary_payload_serializes_requested_tags():
    from data_collection.ib_paper_data import account_summary_payload

    ib = _FakeIB()
    payload = account_summary_payload(ib, tags=["AvailableFunds", "NetLiquidation"])

    assert payload["capability"] == "paper_account_summary"
    assert payload["row_count"] == 1
    assert payload["rows"][0]["tag"] == "AvailableFunds"
    assert ib.requests[0] == ("accountSummary",)


def test_open_orders_payload_requests_and_serializes_status():
    from data_collection.ib_paper_data import open_orders_payload

    ib = _FakeIB()
    payload = open_orders_payload(ib)

    assert payload["capability"] == "paper_open_orders_and_status"
    assert payload["open_trade_count"] == 1
    assert payload["open_trades"][0]["orderStatus"]["status"] == "Submitted"
    assert ("reqAllOpenOrders",) in ib.requests


def test_delayed_snapshot_payload_requests_delayed_market_data(monkeypatch):
    from data_collection import ib_paper_data

    class _Stock:
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency

    monkeypatch.setattr(ib_paper_data, "_import_ib_async", lambda: (object, _Stock, object))
    ib = _FakeIB()

    payload = ib_paper_data.delayed_snapshot_payload(ib, symbol="aapl", wait_seconds=0.1)

    assert payload["symbol"] == "AAPL"
    assert payload["requested_market_data_type"] == 3
    assert payload["snapshot"]["marketPrice"] == 199.15
    assert ("reqMarketDataType", 3) in ib.requests


def test_probe_command_writes_gateway_metadata(tmp_path, monkeypatch):
    from data_collection import ib_paper_data

    output = tmp_path / "probe.json"
    fake = _FakeIB()
    monkeypatch.setattr(ib_paper_data, "connect_ib", lambda settings: fake)
    monkeypatch.setattr(
        ib_paper_data,
        "delayed_snapshot_payload",
        lambda ib, symbol, wait_seconds: {"status": "ok", "symbol": symbol},
    )
    monkeypatch.setattr(
        ib_paper_data,
        "historical_daily_payload",
        lambda ib, symbol: {"status": "ok", "symbol": symbol},
    )

    rc = ib_paper_data.main([
        "probe",
        "--symbol",
        "AAPL",
        "--client-id",
        "42",
        "--output",
        str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gateway"]["client_id"] == 42
    assert payload["checks"]["account_summary"]["status"] == "ok"
    assert fake.disconnected is True


def test_write_json_serializes_date_values(tmp_path):
    from data_collection.ib_paper_data import _write_json

    output = tmp_path / "payload.json"
    _write_json(output, {"rows": [{"date": date(2026, 7, 10)}]})

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"][0]["date"] == "2026-07-10"
