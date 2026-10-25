"""System-level context builder for the expert prompts.

This module owns the bridge from project alpha engines / FINSABER-derived panels
to compact prompt payloads. Experiments evaluate this system context; they do not
define it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from statistics import fmean, pstdev
from typing import Optional, Sequence

from trading_agent.alpha.alpha158_strategy import (
    Alpha158FeatureProvider,
    Alpha158StrategyAlpha,
    QlibAlpha158FeatureProvider,
    SQLiteQlibFactorProvider,
)
from trading_agent.alpha.lowvol import LowVolAlpha
from trading_agent.alpha.llm_news_sentiment import LLMNewsSentimentAlpha, NewsSentimentCache
from trading_agent.alpha.momentum import MomentumAlpha
from trading_agent.alpha.reversal import ReversalAlpha
from trading_agent.alpha.technical_indicators import TechnicalIndicatorBlock
from trading_agent.alpha.text_attention import NewsAttentionNoveltyAlpha
from trading_agent.contracts import Observation
from trading_agent.data.finsaber_news import FinsaberNewsPanel
from trading_agent.alpha.filing_signals import FilingSignalAlpha
from trading_agent.data.finsaber_filings import FinsaberFilingPanel


@dataclass(frozen=True)
class ExpertSignalSources:
    """Optional system data/alpha sources beyond price-derived technical signals."""

    factor_alpha: object | None = None
    news_panel: object | None = None
    news_sentiment_alpha: object | None = None
    filing_alpha: object | None = None
    sec_fundamentals: object | None = None
    # Design D: a TradingViewRating-like object (rating_for(ticker, date) -> {'ma',
    # 'oscillator','summary'}). When present, the technical context carries the rating and
    # TechnicalExpert votes deterministically (no LLM call).
    technical_rating: object | None = None
    # Design D: FredMacroAlpha-like object (features_for(date) -> interpretable macro dict).
    # TradingAgents' news analyst grounds macro commentary in FRED indicators; the macro
    # block lives INSIDE the news context accordingly.
    macro_alpha: object | None = None
    # E-SKILL fed arm: frozen train-window text lessons fed to EVERY expert as a
    # constant `prior_lessons` block. None = unfed system (prompts unchanged).
    prior_lessons: tuple[str, ...] | None = None
    context_cache: dict | None = None


class _NewsPanelTextAdapter:
    """Adapt FINSABER ``news_items`` access to the text-alpha interface."""

    def __init__(self, news_panel: object) -> None:
        self._news_panel = news_panel

    def trailing_items(
        self,
        ticker: str,
        t,
        lookback: int,
        modalities: Sequence[str] = ("news",),
    ) -> list:
        del modalities
        return self._news_panel.trailing_items(ticker, t, lookback, include_current=True)


def build_expert_signal_sources(
    *,
    data_root: str | Path | None = None,
    start: Date | None = None,
    end: Date | None = None,
    tickers: Sequence[str] | None = None,
    alpha158_provider: Alpha158FeatureProvider | None = None,
    alpha158_factors_db: str | Path | None = None,
    qlib_provider_uri: str | None = None,
    news_sentiment_backend: object | None = None,
    news_sentiment_cache_path: str | Path | None = None,
    include_news: bool = True,
    strict_news: bool = False,
    include_filings: bool = True,
    strict_filings: bool = False,
    sec_facts_path: str | Path | None = None,
    technical_rating: object | None = None,
    fred_macro_path: str | Path | None = None,
    prior_lessons_path: str | Path | None = None,
) -> ExpertSignalSources:
    """Build system-level factor/news sources from existing project data adapters.

    Price-derived technical and fundamental context is always constructed by
    ``ExpertSignalContextBuilder`` from the run's price panel. This factory adds
    optional external Alpha158 and FINSABER news inputs.
    """
    provider = alpha158_provider
    if provider is None and alpha158_factors_db is not None:
        provider = SQLiteQlibFactorProvider(alpha158_factors_db)
    if provider is None and qlib_provider_uri is not None:
        provider = QlibAlpha158FeatureProvider(qlib_provider_uri)
    factor_alpha = Alpha158StrategyAlpha(provider) if provider is not None else None

    news_panel = None
    if include_news and data_root is not None and start is not None and end is not None:
        try:
            news_panel = FinsaberNewsPanel(data_root, start, end, tickers=tickers)
        except (FileNotFoundError, RuntimeError, ValueError):
            if strict_news:
                raise
            news_panel = None

    news_sentiment_alpha = None
    if news_panel is not None and news_sentiment_backend is not None:
        cache = NewsSentimentCache(news_sentiment_cache_path) if news_sentiment_cache_path else None
        news_sentiment_alpha = LLMNewsSentimentAlpha(
            news_panel,
            backend=news_sentiment_backend,
            cache=cache,
        )

    filing_alpha = None
    if include_filings and data_root is not None and start is not None and end is not None:
        try:
            filing_panel = FinsaberFilingPanel(data_root, start, end, tickers=tickers)
            filing_alpha = FilingSignalAlpha(filing_panel)
        except (FileNotFoundError, RuntimeError, ValueError):
            if strict_filings:
                raise
            filing_alpha = None

    sec_fundamentals = None
    if sec_facts_path is not None and Path(sec_facts_path).exists():
        from trading_agent.alpha.sec_fundamentals import SecFundamentalsAlpha
        sec_fundamentals = SecFundamentalsAlpha(str(sec_facts_path))

    macro_alpha = None
    if fred_macro_path is not None:
        from trading_agent.alpha.fred_macro import FredMacroAlpha

        loaded = FredMacroAlpha(fred_macro_path)
        macro_alpha = loaded if loaded.available else None

    prior_lessons = None
    if prior_lessons_path is not None:
        import json as _json

        payload = _json.loads(Path(prior_lessons_path).read_text(encoding="utf-8"))
        # Accept either a flat list of lesson strings or the holdout report's
        # lessons_by_state mapping (flattened in sorted-state order).
        if isinstance(payload, dict):
            flat: list[str] = []
            for state in sorted(payload):
                flat.extend(str(x) for x in payload[state])
            prior_lessons = tuple(flat)
        else:
            prior_lessons = tuple(str(x) for x in payload)

    return ExpertSignalSources(
        factor_alpha=factor_alpha,
        news_panel=news_panel,
        news_sentiment_alpha=news_sentiment_alpha,
        filing_alpha=filing_alpha,
        sec_fundamentals=sec_fundamentals,
        technical_rating=technical_rating,
        macro_alpha=macro_alpha,
        prior_lessons=prior_lessons,
        context_cache={},
    )


def _round_or_none(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def _signal_summary(signals: Sequence, universe_size: int) -> dict:
    total = max(0, int(universe_size))
    if not signals:
        return {
            "n": 0,
            "coverage": 0.0,
            "mean": None,
            "min": None,
            "max": None,
            "positive_fraction": None,
            "top": [],
        }
    values = [float(s.value) for s in signals]
    ranked = sorted(signals, key=lambda s: float(s.value), reverse=True)
    return {
        "n": len(signals),
        "coverage": round(len(signals) / total, 6) if total else 0.0,
        "mean": _round_or_none(fmean(values)),
        "min": _round_or_none(min(values)),
        "max": _round_or_none(max(values)),
        "positive_fraction": round(sum(1 for v in values if v > 0) / len(values), 6),
        "top": [
            {"ticker": s.ticker, "value": _round_or_none(float(s.value))}
            for s in ranked[:3]
        ],
    }


def _signals_by_ticker(signals: Sequence, tickers: Sequence[str]) -> dict[str, dict]:
    values = {s.ticker: float(s.value) for s in signals}
    return {
        ticker: {
            "available": ticker in values,
            "value": _round_or_none(values.get(ticker)),
        }
        for ticker in tickers
    }


def _signal_value_map(signals: Sequence) -> dict[str, float]:
    return {s.ticker: float(s.value) for s in signals}


def _group_signal_summaries(signals: Sequence, universe_size: int) -> dict[str, dict]:
    grouped: dict[str, list] = {}
    for signal in signals:
        grouped.setdefault(signal.source_id, []).append(signal)
    return {
        source_id: _signal_summary(grouped[source_id], universe_size)
        for source_id in sorted(grouped)
    }


def _technical_context(
    observation: Observation,
    engines: dict[str, object],
    indicator_block: object | None = None,
    technical_rating: object | None = None,
) -> dict:
    by_ticker = {
        ticker: {"available": True}
        for ticker in observation.tickers
    }
    if technical_rating is not None:
        # Design D: deterministic TradingView-style rating per ticker. TechnicalExpert
        # votes directly from this block (no LLM call) when it is present.
        for ticker in observation.tickers:
            try:
                r = technical_rating.rating_for(ticker, observation.decision_date)
            except Exception:
                r = None
            if r is not None:
                by_ticker[ticker]["rating"] = {
                    "summary": _round_or_none(r.get("summary")),
                    "ma": _round_or_none(r.get("ma")),
                    "oscillator": _round_or_none(r.get("oscillator")),
                }
    adapter_errors: dict[str, str] = {}
    for name, engine in engines.items():
        try:
            signals = list(engine.score(observation))
        except Exception as exc:
            signals = []
            adapter_errors[name] = type(exc).__name__
        values = _signal_value_map(signals)
        for ticker in observation.tickers:
            if ticker in values:
                by_ticker[ticker][name] = _round_or_none(values[ticker])
            else:
                by_ticker[ticker][name] = None
                by_ticker[ticker]["available"] = False
                by_ticker[ticker].setdefault("missing_signals", []).append(name)
    if indicator_block is not None:
        # TradingAgents-style named indicator readings; advisory context only,
        # so absence never flips the ticker's availability flag.
        for ticker in observation.tickers:
            try:
                indicators = indicator_block.indicators_for(ticker, observation.decision_date)
            except Exception as exc:
                adapter_errors["indicators"] = type(exc).__name__
                break
            if indicators is not None:
                by_ticker[ticker]["indicators"] = indicators
    for row in by_ticker.values():
        if not row["available"]:
            missing = row.get("missing_signals", [])
            if any(name in adapter_errors for name in missing):
                row["unavailable_reason"] = "signal_adapter_error"
            else:
                row["unavailable_reason"] = "insufficient_trailing_history"
    return {
        "available": True,
        "n_tickers": len(observation.tickers),
        "adapter_errors": adapter_errors,
        "by_ticker": by_ticker,
    }


def _factor_context(observation: Observation, factor_alpha=None) -> dict:
    if factor_alpha is None:
        return {
            "available": False,
            "provider": "none",
            "coverage": 0.0,
            "by_ticker": {
                ticker: {"available": False, "score": None}
                for ticker in observation.tickers
            },
        }
    total = len(observation.tickers)
    try:
        if hasattr(factor_alpha, "score_by_strategy"):
            by_strategy = factor_alpha.score_by_strategy(observation)
            values_by_strategy = {
                key: _signal_value_map(list(value))
                for key, value in sorted(by_strategy.items())
            }
        else:
            values_by_strategy = {"score": _signal_value_map(list(factor_alpha.score(observation)))}
    except Exception as exc:
        return {
            "available": False,
            "provider": type(factor_alpha).__name__,
            "coverage": 0.0,
            "error": type(exc).__name__,
            "by_ticker": {
                ticker: {"available": False, "score": None}
                for ticker in observation.tickers
            },
        }
    by_ticker = {}
    for ticker in observation.tickers:
        row = {"available": False}
        for key, values in values_by_strategy.items():
            val = values.get(ticker)
            row[key] = _round_or_none(val) if val is not None else None
            if val is not None:
                row["available"] = True
        by_ticker[ticker] = row
    covered = sum(1 for row in by_ticker.values() if row["available"])
    coverage = round(covered / total, 6) if total else 0.0
    return {
        "available": covered > 0,
        "provider": type(factor_alpha).__name__,
        "coverage": coverage,
        "by_ticker": by_ticker,
    }


def _news_data_quality(coverage: float, raw_count: int) -> str:
    if coverage <= 0.0 or raw_count <= 0:
        return "low"
    if coverage >= 0.6 and raw_count >= 3:
        return "high"
    return "medium"


def _news_context(observation: Observation, news_panel=None, news_alpha=None,
                  news_sentiment_alpha=None, macro_alpha=None) -> dict:
    if news_panel is None and news_alpha is None and news_sentiment_alpha is None:
        return {
            "available": False,
            "provider": "none",
            "coverage": 0.0,
            "by_ticker": {
                ticker: {"available": False}
                for ticker in observation.tickers
            },
            "data_quality": "unknown",
        }
    total = len(observation.tickers)
    out = {
        "available": True,
        "provider": type(news_panel or news_alpha or news_sentiment_alpha).__name__,
        "coverage": 0.0,
        "by_ticker": {
            ticker: {"available": False}
            for ticker in observation.tickers
        },
        "data_quality": "unknown",
        "limitations": [
            "FINSABER news is date-level; use same-day text signals only through next-bar execution."
        ],
    }
    if news_panel is not None:
        try:
            coverage, raw_count, dedup_count = news_panel.coverage(
                observation.tickers,
                observation.decision_date,
            )
            out.update({
                "coverage": round(float(coverage), 6),
                "raw_item_count": int(raw_count),
                "dedup_item_count": int(dedup_count),
                "data_quality": _news_data_quality(float(coverage), int(raw_count)),
            })
            for ticker in observation.tickers:
                items = news_panel.items_on(ticker, observation.decision_date)
                dedup = news_panel.dedup_items(items)
                out["by_ticker"][ticker].update({
                    "available": len(items) > 0,
                    "raw_item_count": len(items),
                    "dedup_item_count": len(dedup),
                })
                # Design D (TradingAgents-style): the expert reads recent news TEXT, not
                # only derived features. Latest-first snippets from the trailing week,
                # capped for prompt size; items are dated <= t (point-in-time).
                try:
                    trailing = news_panel.trailing_items(
                        ticker, observation.decision_date, 7, include_current=True
                    )
                    dedup_trailing = news_panel.dedup_items(trailing)
                    snippets = [
                        " ".join(item.text.split())[:160]
                        for item in sorted(dedup_trailing, key=lambda it: it.date, reverse=True)
                        if getattr(item, "text", "")
                    ][:5]
                    if snippets:
                        out["by_ticker"][ticker]["recent_news"] = snippets
                except Exception:
                    pass
        except Exception as exc:
            out.update({"available": False, "error": type(exc).__name__})
    if macro_alpha is not None:
        # Shared macro backdrop (FRED, publication-lag embargoed) — TA's news analyst
        # grounds macro commentary in these indicators.
        try:
            macro = macro_alpha.features_for(observation.decision_date)
            if macro.get("available"):
                out["macro"] = macro
        except Exception:
            pass
    signals = []
    for alpha in (news_alpha, news_sentiment_alpha):
        if alpha is None:
            continue
        try:
            signals.extend(list(alpha.score(observation)))
        except Exception as exc:
            out.update({"available": False, "error": type(exc).__name__})
            return out
    if signals:
        for signal in signals:
            row = out["by_ticker"].setdefault(signal.ticker, {"available": False})
            if signal.source_id == "news_attention_novelty":
                row["attention_novelty"] = _round_or_none(float(signal.value))
            elif signal.source_id == "llm_news_sentiment":
                row["sentiment"] = _round_or_none(float(signal.value))
            else:
                row[signal.source_id] = _round_or_none(float(signal.value))
            row["available"] = True
    if news_sentiment_alpha is not None and hasattr(news_sentiment_alpha, "metadata"):
        out.update(news_sentiment_alpha.metadata())
    return out


def _market_realized_vol(panel, observation: Observation, window: int = 20) -> Optional[float]:
    dates = panel.dates_upto(observation.decision_date)
    if len(dates) < window + 1:
        return None
    dates = dates[-(window + 1):]
    market_returns: list[float] = []
    for i in range(1, len(dates)):
        per_name = []
        for ticker in observation.tickers:
            p0 = panel._price_on(ticker, dates[i - 1])
            p1 = panel._price_on(ticker, dates[i])
            if p0 and p1 and p0 > 0:
                per_name.append(p1 / p0 - 1.0)
        if per_name:
            market_returns.append(fmean(per_name))
    if len(market_returns) < 2:
        return None
    return pstdev(market_returns) * (252 ** 0.5)


def _ticker_realized_vol(panel, ticker: str, t, window: int = 20) -> Optional[float]:
    closes = panel.trailing(ticker, t, window + 1)
    if len(closes) < 3:
        return None
    returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(returns) < 2:
        return None
    return pstdev(returns) * (252 ** 0.5)


def _ticker_drawdown(panel, ticker: str, t, window: int = 60) -> Optional[float]:
    closes = panel.trailing(ticker, t, window)
    if len(closes) < 2:
        return None
    peak = max(closes)
    if peak <= 0:
        return None
    return closes[-1] / peak - 1.0


def _fundamental_context(panel, observation: Observation, filing_alpha=None,
                         sec_fundamentals=None) -> dict:
    breadth_hits = 0
    breadth_total = 0
    drawdowns: list[float] = []
    for ticker in observation.tickers:
        closes_50 = panel.trailing(ticker, observation.decision_date, 50)
        if len(closes_50) >= 50:
            breadth_total += 1
            if closes_50[-1] > fmean(closes_50):
                breadth_hits += 1
        closes_60 = panel.trailing(ticker, observation.decision_date, 60)
        if len(closes_60) >= 2:
            peak = max(closes_60)
            if peak > 0:
                drawdowns.append(closes_60[-1] / peak - 1.0)
    breadth = None if breadth_total == 0 else breadth_hits / breadth_total
    by_ticker: dict[str, dict] = {}
    filing_covered = 0
    financials_covered = 0
    for ticker in observation.tickers:
        row = {
            "available": True,
            "drawdown_60d": _round_or_none(_ticker_drawdown(panel, ticker, observation.decision_date)),
            "realized_vol_20d": _round_or_none(_ticker_realized_vol(panel, ticker, observation.decision_date)),
        }
        if filing_alpha is not None:
            # Statement-based (10-K/10-Q) text signals: causal, price-orthogonal.
            filing = filing_alpha.signals_for(ticker, observation.decision_date)
            row["filing"] = filing
            if filing.get("available"):
                filing_covered += 1
        if sec_fundamentals is not None:
            # SEC XBRL annual ratios (margins/ROA/leverage/FCF) as of the latest public
            # 10-K -- the price-orthogonal fundamental core (offline: +0.005 vs technical,
            # vs +0.308 for the price-derived proxies above).
            financials = sec_fundamentals.features_for(ticker, observation.decision_date)
            row["financials"] = financials
            if financials.get("available"):
                financials_covered += 1
        by_ticker[ticker] = row
    return {
        "available": True,
        "n_tickers": len(observation.tickers),
        "breadth_above_50d": _round_or_none(breadth),
        "breadth_coverage": round(breadth_total / len(observation.tickers), 6)
        if observation.tickers else 0.0,
        "realized_vol_20d": _round_or_none(_market_realized_vol(panel, observation)),
        "mean_drawdown_60d": _round_or_none(fmean(drawdowns)) if drawdowns else None,
        "macro_available": False,
        "valuation_available": filing_alpha is not None,
        "filing_coverage": round(filing_covered / len(observation.tickers), 6)
        if (filing_alpha is not None and observation.tickers) else 0.0,
        "financials_available": sec_fundamentals is not None,
        "financials_coverage": round(financials_covered / len(observation.tickers), 6)
        if (sec_fundamentals is not None and observation.tickers) else 0.0,
        "by_ticker": by_ticker,
    }


@dataclass
class ExpertSignalContextBuilder:
    """Build compact prompt context for the expert prompts from system sources."""

    panel: object
    store: object
    sources: ExpertSignalSources = ExpertSignalSources()

    def __post_init__(self) -> None:
        self._technical_engines = {
            "momentum": MomentumAlpha(self.panel),
            "reversal": ReversalAlpha(self.panel),
            "lowvol": LowVolAlpha(self.panel),
        }
        self._indicator_block = TechnicalIndicatorBlock(self.panel)
        self._news_alpha = (
            NewsAttentionNoveltyAlpha(self.sources.news_panel, trading_dates=self.panel.dates)
            if self.sources.news_panel is not None
            else None
        )
        self._news_sentiment_alpha = self.sources.news_sentiment_alpha

    def build(self, observation: Observation, market_state: str) -> dict:
        del market_state
        cache = self.sources.context_cache
        cache_key = None
        if cache is not None:
            cache_key = (
                observation.decision_date,
                tuple(observation.tickers),
                id(self.sources.factor_alpha),
                id(self.sources.news_panel),
                id(self.sources.news_sentiment_alpha),
                id(self.sources.filing_alpha),
                id(self.sources.sec_fundamentals),
                id(self.sources.technical_rating),
                id(self.sources.macro_alpha),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        context = {
            "technical": _technical_context(
                observation, self._technical_engines, self._indicator_block,
                self.sources.technical_rating,
            ),
            "factor": _factor_context(observation, self.sources.factor_alpha),
            "news": _news_context(
                observation,
                self.sources.news_panel,
                self._news_alpha,
                self._news_sentiment_alpha,
                self.sources.macro_alpha,
            ),
            "fundamental": _fundamental_context(
                self.panel, observation, self.sources.filing_alpha,
                self.sources.sec_fundamentals,
            ),
        }
        # E-SKILL fed arm: a CONSTANT frozen lesson block for every expert
        # (input-side treatment; interpretation context only, never an order
        # authority). Constant by design — a per-date/state selection would
        # make the experts state-aware and break the regime-blind replay
        # economy. None (the default) leaves prompts byte-identical to the
        # unfed system.
        if self.sources.prior_lessons:
            lessons = list(self.sources.prior_lessons)
            for expert_block in context.values():
                if isinstance(expert_block, dict):
                    expert_block["prior_lessons"] = lessons
        if cache is not None and cache_key is not None:
            cache[cache_key] = context
        return context
