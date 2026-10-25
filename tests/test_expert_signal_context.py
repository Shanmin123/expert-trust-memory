"""System-level expert context construction from alpha engines and FINSABER data."""

from __future__ import annotations

from datetime import date

import pandas as pd

from trading_agent.alpha.alpha158_recipes import REGISTERED_ALPHA158_FEATURES
from trading_agent.alpha.alpha158_strategy import (
    Alpha158StrategyAlpha,
    FrameAlpha158FeatureProvider,
    SQLiteQlibFactorProvider,
)
from trading_agent.alpha.llm_news_sentiment import (
    DEFAULT_LLM_NEWS_MODEL,
    LLMNewsSentimentAlpha,
    SIGNAL_ID as LLM_NEWS_SENTIMENT_SIGNAL_ID,
)
from trading_agent.alpha.text_attention import SIGNAL_ID as NEWS_ATTENTION_SIGNAL_ID
from trading_agent.contracts import Bar, Observation
from trading_agent.data.finsaber_news import FinsaberNewsPanel, NewsItem
from trading_agent.expert_context import (
    ExpertSignalContextBuilder,
    ExpertSignalSources,
    build_expert_signal_sources,
)
from trading_agent.market import LoaderPricePanel
from trading_agent.memory.expert_weights import ExpertWeightStore


class StubNewsSentimentBackend:
    model_name = DEFAULT_LLM_NEWS_MODEL
    model_revision = "test-revision"

    def call(self, system_prompt, user_prompt):
        import json

        payload = json.loads(user_prompt)
        return json.dumps({
            "items": [
                {
                    "index": row["index"],
                    "positive": 0.8,
                    "neutral": 0.1,
                    "negative": 0.1,
                    "confidence": 0.8,
                }
                for row in payload["items"]
            ]
        })


class _Loader:
    def __init__(self, prices_by_date):
        self._prices_by_date = prices_by_date

    def get_date_range(self):
        return sorted(self._prices_by_date)

    def get_data_by_date(self, d):
        return {
            "price": {
                ticker: {"adjusted_close": price, "close": price}
                for ticker, price in self._prices_by_date[d].items()
            }
        }


def _obs(day: date) -> Observation:
    bar = Bar(open=1, high=1, low=1, close=1, adjusted_close=1, volume=1)
    from datetime import datetime

    return Observation(
        decision_date=day,
        decision_time=datetime(day.year, day.month, day.day, 9, 30),
        availability_time=datetime(day.year, day.month, day.day, 9, 30),
        bars={"AAA": bar, "BBB": bar},
        tickers=("AAA", "BBB"),
    )


def test_fundamental_context_includes_filing_signals_when_wired():
    from trading_agent.alpha.filing_signals import FilingSignalAlpha
    from trading_agent.data.finsaber_filings import FilingItem, FinsaberFilingPanel

    dates = pd.bdate_range("2024-01-02", periods=70).date
    prices = {d: {"AAA": 100.0 + i, "BBB": 100.0 + i * 0.5} for i, d in enumerate(dates)}
    panel = LoaderPricePanel(_Loader(prices))

    filing_panel = FinsaberFilingPanel.from_items([
        FilingItem(date(2022, 3, 1), "AAA", "10-K", "alpha beta gamma delta"),
        FilingItem(date(2023, 3, 1), "AAA", "10-K", "alpha beta gamma epsilon zeta"),
    ])
    sources = ExpertSignalSources(filing_alpha=FilingSignalAlpha(filing_panel))
    builder = ExpertSignalContextBuilder(panel=panel, store=ExpertWeightStore(), sources=sources)

    fundamental = builder.build(_obs(dates[-1]), "up_stress")["fundamental"]
    assert fundamental["valuation_available"] is True
    aaa_filing = fundamental["by_ticker"]["AAA"]["filing"]
    assert aaa_filing["available"] is True
    assert aaa_filing["form"] == "10-K"
    assert aaa_filing["text_change"] is not None
    # BBB has no filings -> its filing block is marked unavailable, not fabricated.
    assert fundamental["by_ticker"]["BBB"]["filing"]["available"] is False
    assert 0.0 <= fundamental["filing_coverage"] <= 1.0


def test_fundamental_context_without_filings_leaves_valuation_off():
    dates = pd.bdate_range("2024-01-02", periods=70).date
    prices = {d: {"AAA": 100.0 + i, "BBB": 100.0 + i * 0.5} for i, d in enumerate(dates)}
    panel = LoaderPricePanel(_Loader(prices))
    builder = ExpertSignalContextBuilder(
        panel=panel, store=ExpertWeightStore(), sources=ExpertSignalSources()
    )
    fundamental = builder.build(_obs(dates[-1]), "up_stress")["fundamental"]
    assert fundamental["valuation_available"] is False
    assert "filing" not in fundamental["by_ticker"]["AAA"]


def test_context_builder_uses_system_factor_and_news_sources():
    # 130 days: enough trailing history for the time-series technical signals
    # (momentum needs lookback+min_history+1 = 71, lowvol window+min_history+1 = 121).
    dates = pd.bdate_range("2024-01-02", periods=130).date
    prices = {
        d: {
            "AAA": 100.0 + i,
            "BBB": 100.0 + (i * 0.5),
        }
        for i, d in enumerate(dates)
    }
    panel = LoaderPricePanel(_Loader(prices))
    feature = REGISTERED_ALPHA158_FEATURES[0]
    # Feature history over the whole window: the factor signal is a time-series
    # z-score against each ticker's own prior values.
    factor_frame = pd.DataFrame(
        [
            {"date": d, "ticker": ticker, feature.strategy_id: base + 0.02 * ((i % 5) - 2)}
            for i, d in enumerate(dates)
            for ticker, base in (("AAA", 1.0), ("BBB", -1.0))
        ]
    )
    factor_alpha = Alpha158StrategyAlpha(
        FrameAlpha158FeatureProvider(factor_frame),
        features=(feature,),
        history_window=120,
        min_history=20,
    )
    news_panel = FinsaberNewsPanel.from_items([
        NewsItem(date=dates[-1], ticker="AAA", text="AAA reports strong growth"),
    ])
    sources = ExpertSignalSources(
        factor_alpha=factor_alpha,
        news_panel=news_panel,
        news_sentiment_alpha=LLMNewsSentimentAlpha(
            news_panel,
            backend=StubNewsSentimentBackend(),
            lookback=1,
        ),
    )
    builder = ExpertSignalContextBuilder(panel=panel, store=ExpertWeightStore(), sources=sources)

    context = builder.build(_obs(dates[-1]), "up_stress")

    assert context["factor"]["available"] is True
    assert context["factor"]["coverage"] == 1.0
    assert context["factor"]["by_ticker"]["AAA"][feature.strategy_id] is not None
    assert context["factor"]["by_ticker"]["BBB"][feature.strategy_id] is not None
    assert context["news"]["available"] is True
    assert context["news"]["coverage"] == 0.5
    assert context["news"]["by_ticker"]["AAA"]["attention_novelty"] is not None
    assert context["news"]["by_ticker"]["AAA"]["sentiment"] is not None
    assert "xsec_text_tone" not in context["news"]["by_ticker"]["AAA"]
    assert context["news"]["nlp_model_name"] == DEFAULT_LLM_NEWS_MODEL
    assert context["news"]["nlp_model_revision"] == "test-revision"
    assert context["news"]["nlp_item_count"] == 1
    assert context["news"]["nlp_mean_confidence"] == 0.8
    assert context["news"]["data_quality"] == "medium"
    assert context["news"]["limitations"] == [
        "FINSABER news is date-level; use same-day text signals only through next-bar execution."
    ]
    technical_aaa = context["technical"]["by_ticker"]["AAA"]
    assert technical_aaa["momentum"] is not None
    assert technical_aaa["reversal"] is not None
    assert technical_aaa["lowvol"] is not None
    # TradingAgents-style indicator block (stockstats-backed, close-based subset).
    assert technical_aaa["indicators"]["rsi"] is not None
    assert technical_aaa["indicators"]["close_50_sma_gap"] is not None
    assert technical_aaa["indicators"]["close_200_sma_gap"] is None  # only 130 days
    assert context["fundamental"]["breadth_coverage"] == 1.0
    assert "market_state" not in context["fundamental"]
    assert context["fundamental"]["by_ticker"]["AAA"]["drawdown_60d"] is not None


def test_context_builder_reuses_shared_source_cache(monkeypatch):
    from trading_agent import expert_context

    dates = pd.bdate_range("2024-01-02", periods=70).date
    prices = {
        d: {"AAA": 100.0 + i, "BBB": 90.0 + i}
        for i, d in enumerate(dates)
    }
    panel = LoaderPricePanel(_Loader(prices))
    sources = ExpertSignalSources(context_cache={})
    builder = ExpertSignalContextBuilder(panel=panel, store=ExpertWeightStore(), sources=sources)
    calls = {"technical": 0}
    original = expert_context._technical_context

    def counting_technical_context(observation, engines, indicator_block=None,
                                   technical_rating=None):
        calls["technical"] += 1
        return original(observation, engines, indicator_block, technical_rating)

    monkeypatch.setattr(expert_context, "_technical_context", counting_technical_context)

    first = builder.build(_obs(dates[-1]), "up_stress")
    second = builder.build(_obs(dates[-1]), "up_stress")

    assert first == second
    assert calls["technical"] == 1


def test_build_expert_signal_sources_from_existing_alpha_and_finsaber_data(monkeypatch):
    built = {}

    class FakeProvider:
        def __init__(self, db_path):
            built["db_path"] = db_path

    class FakeNewsPanel:
        def __init__(self, root, start_date, end_date, tickers=None):
            built["news"] = (root, start_date, end_date, tuple(tickers or ()))

    monkeypatch.setattr("trading_agent.expert_context.SQLiteQlibFactorProvider", FakeProvider)
    monkeypatch.setattr("trading_agent.expert_context.FinsaberNewsPanel", FakeNewsPanel)

    sources = build_expert_signal_sources(
        data_root="finsaber-root",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        tickers=("AAA", "BBB"),
        alpha158_factors_db="alpha158.sqlite",
    )

    assert sources.factor_alpha is not None
    assert sources.factor_alpha.provider.__class__ is FakeProvider
    assert built["db_path"] == "alpha158.sqlite"
    assert sources.news_panel.__class__ is FakeNewsPanel
    assert built["news"] == (
        "finsaber-root",
        date(2024, 1, 1),
        date(2024, 1, 31),
        ("AAA", "BBB"),
    )


def test_build_expert_signal_sources_wires_llm_news_sentiment_backend(monkeypatch):
    class FakeNewsPanel:
        def __init__(self, root, start_date, end_date, tickers=None):
            self.root = root

    monkeypatch.setattr("trading_agent.expert_context.FinsaberNewsPanel", FakeNewsPanel)

    sources = build_expert_signal_sources(
        data_root="finsaber-root",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        tickers=("AAA",),
        news_sentiment_backend=StubNewsSentimentBackend(),
    )

    assert sources.news_panel.__class__ is FakeNewsPanel
    assert isinstance(sources.news_sentiment_alpha, LLMNewsSentimentAlpha)
    assert sources.news_sentiment_alpha.metadata()["nlp_model_name"] == DEFAULT_LLM_NEWS_MODEL


def test_build_expert_signal_sources_can_disable_unavailable_news(monkeypatch):
    class BrokenNewsPanel:
        def __init__(self, *args, **kwargs):
            raise FileNotFoundError("no news_items")

    monkeypatch.setattr("trading_agent.expert_context.FinsaberNewsPanel", BrokenNewsPanel)

    sources = build_expert_signal_sources(
        data_root="price-only-root",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        tickers=("AAA",),
        strict_news=False,
    )

    assert sources.news_panel is None


def test_sqlite_qlib_factor_provider_reports_available_tickers(tmp_path):
    import sqlite3

    db_path = tmp_path / "factors.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE qlib_direct_AAA (date TEXT, QLIB158_ROC5 REAL)")
        conn.execute("CREATE TABLE qlib_direct_BBB (date TEXT, QLIB158_ROC5 REAL)")
        conn.execute("CREATE TABLE unrelated_table (date TEXT)")

    provider = SQLiteQlibFactorProvider(db_path)

    assert provider.available_tickers() == ("AAA", "BBB")


def test_prior_lessons_feed_constant_across_experts_and_absent_by_default():
    dates = pd.bdate_range("2024-01-02", periods=70).date
    prices = {d: {"AAA": 100.0 + i, "BBB": 100.0 + i * 0.5} for i, d in enumerate(dates)}
    panel = LoaderPricePanel(_Loader(prices))

    unfed = ExpertSignalContextBuilder(
        panel=panel, store=ExpertWeightStore(), sources=ExpertSignalSources()
    ).build(_obs(dates[-1]), "up_calm")
    assert all(
        "prior_lessons" not in blk for blk in unfed.values() if isinstance(blk, dict)
    )  # unfed system: prompts byte-identical to before E-SKILL

    lessons = ("up_calm: technical led attributed contribution", "global: context line")
    fed = ExpertSignalContextBuilder(
        panel=panel, store=ExpertWeightStore(),
        sources=ExpertSignalSources(prior_lessons=lessons),
    ).build(_obs(dates[-1]), "up_calm")
    for name in ("technical", "news", "fundamental"):
        assert fed[name]["prior_lessons"] == list(lessons)  # constant block, every expert


def test_build_sources_loads_lessons_from_flat_list_and_state_mapping(tmp_path):
    import json as _json

    flat = tmp_path / "flat.json"
    flat.write_text(_json.dumps(["lesson one", "lesson two"]), encoding="utf-8")
    s1 = build_expert_signal_sources(
        data_root=str(tmp_path), start=date(2024, 1, 2), end=date(2024, 3, 1),
        tickers=("AAA",), include_news=False, include_filings=False,
        prior_lessons_path=str(flat),
    )
    assert s1.prior_lessons == ("lesson one", "lesson two")

    mapping = tmp_path / "by_state.json"
    mapping.write_text(_json.dumps({
        "up_calm": ["calm lesson"], "down_stress": ["stress lesson"],
    }), encoding="utf-8")
    s2 = build_expert_signal_sources(
        data_root=str(tmp_path), start=date(2024, 1, 2), end=date(2024, 3, 1),
        tickers=("AAA",), include_news=False, include_filings=False,
        prior_lessons_path=str(mapping),
    )
    assert s2.prior_lessons == ("stress lesson", "calm lesson")  # sorted-state order


def test_prior_lessons_survive_prompt_stripping():
    from trading_agent.experts.news_expert import NewsExpert

    class _B:
        def call(self, system_prompt, user_prompt):
            return ""

    ctx = {"news": {"available": True, "coverage": 1.0,
                    "prior_lessons": ["up_calm: technical led attributed contribution"],
                    "by_ticker": {}}}
    up = NewsExpert(_B())._user_prompt(_obs(date(2024, 7, 1)), "up_calm", ctx)
    assert "technical led attributed contribution" in up
