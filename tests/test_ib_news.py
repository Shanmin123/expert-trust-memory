from __future__ import annotations

import json
from datetime import datetime


class _Provider:
    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name


class _Contract:
    symbol = "AAPL"
    conId = 265598
    exchange = "SMART"
    primaryExchange = "NASDAQ"


class _Headline:
    def __init__(self, provider: str, article_id: str, headline: str, when: datetime):
        self.providerCode = provider
        self.articleId = article_id
        self.headline = headline
        self.time = when
        self.extraData = None


class _Article:
    articleText = "full article body"


class _FakeIB:
    def __init__(self):
        self.disconnected = False
        self.requests = []

    def reqNewsProviders(self):
        return [
            _Provider("BRFG", "Briefing.com General Market Columns"),
            _Provider("BRFUPDN", "Briefing.com Analyst Actions"),
            _Provider("DJNL", "Dow Jones Newsletters"),
        ]

    def qualifyContracts(self, contract):
        self.requests.append(("qualify", contract.symbol, contract.exchange, contract.currency))
        return [_Contract()]

    def reqHistoricalNews(self, con_id, providers, start, end, limit, options):
        self.requests.append(("historical", con_id, providers, start, end, limit, options))
        return [
            _Headline(
                "BRFG",
                "BRFG$1",
                "{A:800015:L:en}Apple headline",
                datetime(2026, 7, 9, 10, 32),
            )
        ]

    def reqNewsArticle(self, provider, article_id, options):
        self.requests.append(("article", provider, article_id, options))
        return _Article()

    def disconnect(self):
        self.disconnected = True


def test_clean_headline_removes_ib_metadata_tag():
    from data_collection.ib_news import clean_headline

    assert clean_headline("{A:800015:L:en}Apple headline") == "Apple headline"
    assert clean_headline("Plain headline") == "Plain headline"


def test_list_providers_payload_marks_default_codes_present():
    from data_collection.ib_news import list_providers_payload

    payload = list_providers_payload(_FakeIB())

    assert payload["providers_count"] == 3
    assert payload["default_provider_codes_present"] == ["BRFG", "BRFUPDN", "DJNL"]


def test_fetch_headlines_payload_qualifies_contract_and_serializes_items():
    from data_collection.ib_news import fetch_headlines_payload

    ib = _FakeIB()
    payload = fetch_headlines_payload(
        ib,
        symbol="aapl",
        provider_codes=["BRFG", "BRFUPDN"],
        start="20260701 00:00:00",
        end="20260710 00:00:00",
        limit=25,
    )

    assert payload["symbol"] == "AAPL"
    assert payload["provider_codes"] == ["BRFG", "BRFUPDN"]
    assert payload["provider_query"] == "BRFG+BRFUPDN"
    assert payload["count"] == 1
    assert payload["items"][0]["headline"] == "Apple headline"
    assert payload["request_failed"] is False
    assert ib.requests[0] == ("qualify", "AAPL", "SMART", "USD")


def test_timed_out_headline_request_is_reported_not_read_as_no_news():
    """A timed-out request returns None; it must not look like a quiet stock.

    ib_insync answers a timeout with None instead of raising, so the serialising
    comprehension used to die with TypeError. Swallowing it into an empty list would be
    worse: the news seat would abstain on a transport failure and nothing downstream
    could tell that apart from a stock with genuinely no coverage.
    """
    from data_collection.ib_news import fetch_headlines_payload

    ib = _FakeIB()
    ib.reqHistoricalNews = lambda *a, **k: None

    payload = fetch_headlines_payload(
        ib,
        symbol="aapl",
        provider_codes=["BRFG"],
        start="20260701 00:00:00",
        end="20260710 00:00:00",
        limit=300,
    )

    assert payload["request_failed"] is True
    assert payload["count"] == 0
    assert payload["items"] == []


def test_fetch_article_payload_omits_body_by_default():
    from data_collection.ib_news import fetch_article_payload

    payload = fetch_article_payload(_FakeIB(), provider_code="BRFG", article_id="BRFG$1")

    assert payload["article_fetch_ok"] is True
    assert payload["article_text_length"] == len("full article body")
    assert "article_text" not in payload


def test_fetch_article_payload_can_include_body():
    from data_collection.ib_news import fetch_article_payload

    payload = fetch_article_payload(
        _FakeIB(),
        provider_code="BRFG",
        article_id="BRFG$1",
        include_body=True,
    )

    assert payload["article_text"] == "full article body"


def test_coverage_payload_runs_each_provider_group():
    from data_collection.ib_news import coverage_payload

    payload = coverage_payload(
        _FakeIB(),
        symbol="AAPL",
        provider_groups=[["BRFG"], ["BRFUPDN", "DJNL"]],
        start="20260701 00:00:00",
        end="20260710 00:00:00",
        limit=10,
    )

    assert payload["symbol"] == "AAPL"
    assert payload["coverage"]["BRFG"]["count"] == 1
    assert payload["coverage"]["BRFUPDN+DJNL"]["earliest"] == "2026-07-09T10:32:00"


def test_headlines_command_writes_json(tmp_path, monkeypatch):
    from data_collection import ib_news

    output = tmp_path / "headlines.json"
    fake = _FakeIB()
    seen = {}
    monkeypatch.setattr(
        ib_news,
        "connect_ib",
        lambda settings: seen.setdefault("settings", settings) and fake,
    )

    rc = ib_news.main([
        "headlines",
        "--symbol",
        "AAPL",
        "--providers",
        "BRFG,BRFUPDN",
        "--start",
        "20260701 00:00:00",
        "--end",
        "20260710 00:00:00",
        "--client-id",
        "17",
        "--output",
        str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["gateway"]["client_id"] == 17
    assert seen["settings"].ib_client_id == 17
    assert fake.disconnected is True
