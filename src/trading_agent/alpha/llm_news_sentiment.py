"""LLM-derived point-in-time news sentiment alpha.

This module keeps NLP logic behind a provider interface: the project owns the
point-in-time data contract and score aggregation, while a mature LLM backend
does the language understanding.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from trading_agent.contracts import Observation, Signal

SIGNAL_ID = "llm_news_sentiment"
DEFAULT_LLM_NEWS_MODEL = "llm-news-sentiment"


@runtime_checkable
class NewsSentimentBackend(Protocol):
    model_name: str
    model_revision: str | None

    def call(self, system_prompt: str, user_prompt: str) -> str: ...


def _normalize_probabilities(scores: Mapping[str, object]) -> dict[str, float]:
    out = {}
    for key in ("positive", "neutral", "negative"):
        if key not in scores:
            raise ValueError("LLM news sentiment must return positive/neutral/negative probabilities")
        out[key] = float(scores[key])
    return out


def _items_for_panel(text_panel, ticker: str, t, lookback: int, modalities: Sequence[str]) -> list:
    try:
        return list(text_panel.trailing_items(ticker, t, lookback, modalities))
    except TypeError:
        return list(text_panel.trailing_items(ticker, t, lookback, include_current=True))


def _system_prompt() -> str:
    return (
        "You are a financial-news NLP classifier. For each item, estimate "
        "positive, neutral, and negative probabilities for next-period equity "
        "sentiment. Return only valid JSON with shape "
        '{"items":[{"index":0,"positive":0.0,"neutral":1.0,'
        '"negative":0.0,"confidence":1.0}]}. Do not make a trade call.'
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def sentiment_config_hash(*, system_prompt: str, max_chars_per_item: int) -> str:
    payload = {
        "system_prompt": system_prompt,
        "temperature": 0.0,
        "max_chars_per_item": int(max_chars_per_item),
        "schema": "positive-neutral-negative-v1",
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


@dataclass(frozen=True)
class CachedSentiment:
    probabilities: Mapping[str, float]
    confidence: float
    raw_output: str


class NewsSentimentCache:
    """Persistent JSONL cache keyed by text, model identity, and prompt config."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._loaded = False

    @staticmethod
    def key(
        *,
        text: str,
        model_name: str,
        model_revision: str | None,
        config_hash: str,
    ) -> str:
        payload = {
            "text_hash": _sha256_text(text),
            "model_name": model_name,
            "model_revision": model_revision,
            "config_hash": config_hash,
        }
        return _sha256_text(json.dumps(payload, sort_keys=True))

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._entries = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("cache_key") or "")
                if key:
                    self._entries[key] = row
        self._loaded = True

    def get(self, key: str) -> CachedSentiment | None:
        self._ensure_loaded()
        row = self._entries.get(key)
        if row is None:
            return None
        probs = _normalize_probabilities(row.get("probabilities", {}))
        return CachedSentiment(
            probabilities=probs,
            confidence=float(row.get("confidence", max(probs.values()))),
            raw_output=str(row.get("raw_output", "")),
        )

    def put(
        self,
        key: str,
        *,
        text: str,
        model_name: str,
        model_revision: str | None,
        config_hash: str,
        raw_output: str,
        parsed: Mapping[str, float],
    ) -> None:
        self._ensure_loaded()
        row = {
            "cache_key": key,
            "text_hash": _sha256_text(text),
            "model_name": model_name,
            "model_revision": model_revision,
            "config_hash": config_hash,
            "raw_output": raw_output,
            "probabilities": {
                "positive": float(parsed["positive"]),
                "neutral": float(parsed["neutral"]),
                "negative": float(parsed["negative"]),
            },
            "confidence": float(parsed.get("confidence", max(
                float(parsed["positive"]),
                float(parsed["neutral"]),
                float(parsed["negative"]),
            ))),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._entries[key] = row


class LLMNewsSentimentAlpha:
    """Cross-sectional LLM sentiment from point-in-time news text."""

    source_id = SIGNAL_ID

    def __init__(
        self,
        text_panel,
        *,
        backend: NewsSentimentBackend,
        lookback: int = 5,
        modalities: Sequence[str] = ("news",),
        max_items_per_ticker: int = 8,
        max_chars_per_item: int = 800,
        cache: NewsSentimentCache | None = None,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if max_items_per_ticker < 1:
            raise ValueError("max_items_per_ticker must be >= 1")
        self._text_panel = text_panel
        self._backend = backend
        self._lookback = int(lookback)
        self._modalities = tuple(modalities)
        self._max_items_per_ticker = int(max_items_per_ticker)
        self._max_chars_per_item = int(max_chars_per_item)
        self._cache = cache
        self._config_hash = sentiment_config_hash(
            system_prompt=_system_prompt(),
            max_chars_per_item=self._max_chars_per_item,
        )
        self._last_metadata = {
            "nlp_model_name": getattr(backend, "model_name", DEFAULT_LLM_NEWS_MODEL),
            "nlp_model_revision": getattr(backend, "model_revision", None),
            "nlp_config_hash": self._config_hash,
            "nlp_item_count": 0,
            "nlp_mean_confidence": None,
            "nlp_cache_hits": 0,
            "nlp_cache_misses": 0,
        }

    def metadata(self) -> dict:
        return dict(self._last_metadata)

    def _classify_one(self, text: str) -> tuple[dict[str, float], bool]:
        model_name = getattr(self._backend, "model_name", DEFAULT_LLM_NEWS_MODEL)
        model_revision = getattr(self._backend, "model_revision", None)
        trimmed = text[: self._max_chars_per_item]
        key = NewsSentimentCache.key(
            text=trimmed,
            model_name=model_name,
            model_revision=model_revision,
            config_hash=self._config_hash,
        )
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return {
                    **dict(cached.probabilities),
                    "confidence": float(cached.confidence),
                }, True
        payload = {
            "items": [{"index": 0, "text": trimmed}]
        }
        raw = self._backend.call(_system_prompt(), json.dumps(payload, ensure_ascii=False))
        data = json.loads(raw.strip())
        rows = data.get("items")
        if not isinstance(rows, list):
            raise ValueError("LLM news sentiment response must contain an items list")
        by_index = {}
        for row in rows:
            if not isinstance(row, dict) or "index" not in row:
                raise ValueError("LLM news sentiment item must contain index")
            by_index[int(row["index"])] = row
        if 0 not in by_index:
            raise ValueError("LLM news sentiment response missing an item index")
        probs = _normalize_probabilities(by_index[0])
        confidence = float(by_index[0].get("confidence", max(probs.values())))
        parsed = {**probs, "confidence": confidence}
        if self._cache is not None:
            self._cache.put(
                key,
                text=trimmed,
                model_name=model_name,
                model_revision=model_revision,
                config_hash=self._config_hash,
                raw_output=raw,
                parsed=parsed,
            )
        return parsed, False

    def _classify(self, texts: Sequence[str]) -> tuple[list[dict[str, float]], int, int]:
        if not texts:
            return [], 0, 0
        out = []
        hits = 0
        misses = 0
        for text in texts[: self._max_items_per_ticker]:
            parsed, from_cache = self._classify_one(text)
            out.append(parsed)
            if from_cache:
                hits += 1
            else:
                misses += 1
        return out, hits, misses

    def raw_score(self, ticker: str, t) -> tuple[float, int, float | None, int, int]:
        items = _items_for_panel(self._text_panel, ticker, t, self._lookback, self._modalities)
        texts = [str(getattr(item, "text", "") or "").strip() for item in items]
        texts = [text for text in texts if text][: self._max_items_per_ticker]
        if not texts:
            return 0.0, 0, None, 0, 0
        probs, hits, misses = self._classify(texts)
        scores = [p["positive"] - p["negative"] for p in probs]
        confidences = [p["confidence"] for p in probs]
        return mean(scores), len(texts), mean(confidences), hits, misses

    def score(self, observation: Observation) -> Sequence[Signal]:
        t = observation.decision_date
        raw: dict[str, float] = {}
        item_counts: list[int] = []
        confidences: list[float] = []
        cache_hits = 0
        cache_misses = 0
        for ticker in observation.tickers:
            value, n_items, confidence, hits, misses = self.raw_score(ticker, t)
            if n_items == 0:
                # No text in the lookback window: emit NO signal rather than a
                # neutral 0.0. A fabricated neutral would mark the news context
                # "available" and turn a data gap into a votable judgment.
                continue
            raw[ticker] = value
            item_counts.append(n_items)
            cache_hits += hits
            cache_misses += misses
            if confidence is not None:
                confidences.append(confidence)
        if not raw:
            return []
        # Raw mean polarity in [-1, 1] (positive minus negative probability).
        # Polarity has an absolute meaning, so no cross-sectional normalisation:
        # a cross-sectional z-score would degenerate to 0 for a single name and
        # would discard the sign of the sentiment itself.
        self._last_metadata = {
            "nlp_model_name": getattr(self._backend, "model_name", DEFAULT_LLM_NEWS_MODEL),
            "nlp_model_revision": getattr(self._backend, "model_revision", None),
            "nlp_config_hash": self._config_hash,
            "nlp_item_count": int(sum(item_counts)),
            "nlp_mean_confidence": round(mean(confidences), 6) if confidences else None,
            "nlp_cache_hits": int(cache_hits),
            "nlp_cache_misses": int(cache_misses),
        }
        return [
            Signal(
                ticker=ticker,
                value=float(value),
                source_id=self.source_id,
                as_of=t,
            )
            for ticker, value in raw.items()
        ]
