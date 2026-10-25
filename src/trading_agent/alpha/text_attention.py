"""Deterministic news attention/novelty alpha signal.

This is B1 in the news-component plan: a cheap, auditable information-arrival signal built from
FINSABER ``news_items`` metadata. It emits a normal alpha-engine Signal
(``xsec_news_attention_novelty``); it never emits trades, positions, or instructions.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import date as Date
from statistics import mean, pstdev
from typing import Sequence

from trading_agent.contracts import Observation, Signal
from trading_agent.data.finsaber_news import FinsaberNewsPanel, NewsItem
from trading_agent.market import _to_date

# Attention/novelty is a time-series signal (each name vs its own prior history),
# so the id carries no cross-sectional prefix.
SIGNAL_ID = "news_attention_novelty"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


@dataclass(frozen=True)
class NewsAttentionConfig:
    """Configuration for the B1 attention/novelty extractor."""

    novelty_lookback: int = 60
    baseline_lookback: int = 63
    min_history: int = 20

    def __post_init__(self) -> None:
        if self.novelty_lookback < 1:
            raise ValueError("novelty_lookback must be >= 1")
        if self.baseline_lookback < 1:
            raise ValueError("baseline_lookback must be >= 1")
        if self.min_history < 1:
            raise ValueError("min_history must be >= 1")


@dataclass(frozen=True)
class NewsAttentionFeatures:
    raw_count: int
    dedup_count: int
    dup_share: float
    total_len: int
    novelty_sum: float
    abn_count: float = 0.0
    abn_len: float = 0.0
    abn_novelty: float = 0.0
    raw_score: float = 0.0


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t.strip("'-").lower() for t in _TOKEN_RE.findall(text or "") if t.strip("'-"))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = sum(1 for token in a if token in b)
    return inter / (len(a) + len(b) - inter)


def _zscore(value: float, history: Sequence[float]) -> float:
    if not history:
        return 0.0
    sigma = pstdev(history) if len(history) > 1 else 0.0
    if sigma < 1e-12:
        return 0.0
    return (value - mean(history)) / sigma


class NewsAttentionNoveltyAlpha:
    """Abnormal attention + novelty vs each name's own prior history, from date-level news items.

    T-day news is used to form the date-T signal, and backtests/diagnostics consume it only through
    forward labels / next-bar execution. Rolling baselines and novelty history are strictly prior-only.
    Passing ``trading_dates`` lets the baseline include no-news trading days as zero-attention days; if it
    is omitted, the panel's own news dates are used as a conservative fallback for small fixtures.
    """

    source_id = SIGNAL_ID

    def __init__(
        self,
        news_panel: FinsaberNewsPanel,
        config: NewsAttentionConfig | None = None,
        trading_dates: Sequence[Date] | None = None,
    ):
        self._news_panel = news_panel
        self._config = config or NewsAttentionConfig()
        self._trading_dates = sorted({_to_date(d) for d in trading_dates}) if trading_dates is not None else None
        self._daily_cache: dict[tuple[str, Date], NewsAttentionFeatures] = {}
        self._feature_cache: dict[tuple[str, Date], NewsAttentionFeatures] = {}
        self._token_cache: dict[str, frozenset[str]] = {}

    def shifted_copy(self, shift: int) -> "NewsAttentionNoveltyAlpha":
        """Return a feature-space placebo shifted forward by ``shift`` trading dates.

        This is for diagnostics only. It preserves already-computed same-day news-derived features,
        maps them to future trading dates, and re-runs the prior-only abnormal baseline there. Dates
        without shifted source news stay neutral because the copy uses an empty news panel.
        """
        if shift <= 0 or self._trading_dates is None:
            return self
        out = NewsAttentionNoveltyAlpha(FinsaberNewsPanel.from_items([]), self._config, self._trading_dates)
        out._token_cache = self._token_cache
        date_to_index = {d: i for i, d in enumerate(self._trading_dates)}
        for (ticker, d), features in self._daily_cache.items():
            idx = date_to_index.get(d)
            if idx is None:
                continue
            j = idx + shift
            if j < len(self._trading_dates):
                target = self._trading_dates[j]
                out._daily_cache[(ticker, target)] = features
        return out

    def _prior_dates(self, ticker: str, t, lookback: int) -> list[Date]:
        t = _to_date(t)
        if self._trading_dates is not None:
            dates = self._trading_dates
        else:
            dates = self._news_panel.dates_by_ticker.get(ticker, [])
        hi = bisect.bisect_left(dates, t)
        lo = max(0, hi - lookback)
        return list(dates[lo:hi])

    def _tokens_for_item(self, item: NewsItem) -> frozenset[str]:
        cached = self._token_cache.get(item.dedup_key)
        if cached is not None:
            return cached
        toks = _tokens(item.text)
        self._token_cache[item.dedup_key] = toks
        return toks

    def _novelty_against_prior(self, items: Sequence[NewsItem], prior_items: Sequence[NewsItem]) -> float:
        """Exact max-Jaccard novelty using an inverted token index over prior articles."""
        if not items:
            return 0.0
        if not prior_items:
            return float(len(items))
        prior_tokens = [self._tokens_for_item(item) for item in prior_items]
        token_to_prior: dict[str, list[int]] = {}
        for i, toks in enumerate(prior_tokens):
            for token in toks:
                token_to_prior.setdefault(token, []).append(i)
        total = 0.0
        for item in items:
            toks = self._tokens_for_item(item)
            overlaps: dict[int, int] = {}
            for token in toks:
                for i in token_to_prior.get(token, ()):  # exact intersection counts
                    overlaps[i] = overlaps.get(i, 0) + 1
            max_sim = 0.0
            for i, inter in overlaps.items():
                denom = len(toks) + len(prior_tokens[i]) - inter
                if denom <= 0:
                    continue
                max_sim = max(max_sim, inter / denom)
                if max_sim >= 1.0:
                    break
            total += max(0.0, 1.0 - max_sim)
        return total

    def _novelty_sum_for_items(self, ticker: str, t, items: Sequence[NewsItem]) -> float:
        if not items:
            return 0.0
        prior_items: list[NewsItem] = []
        for d in self._prior_dates(ticker, t, self._config.novelty_lookback):
            prior_items.extend(self._news_panel.items_on(ticker, d))
        prior_items = self._news_panel.dedup_items(prior_items)
        return self._novelty_against_prior(items, prior_items)

    def precompute_ticker(self, ticker: str, start_date: Date | None = None) -> None:
        """Precompute raw daily features for one ticker in chronological order."""
        if self._trading_dates is None:
            return

        window: deque[tuple[Date, NewsItem]] = deque()
        by_key: dict[str, deque[NewsItem]] = {}
        rep_tokens_by_key: dict[str, frozenset[str]] = {}
        token_to_keys: dict[str, set[str]] = {}

        def remove_rep(key: str) -> None:
            toks = rep_tokens_by_key.pop(key, frozenset())
            for token in toks:
                keys = token_to_keys.get(token)
                if not keys:
                    continue
                keys.discard(key)
                if not keys:
                    token_to_keys.pop(token, None)

        def add_rep(key: str, item: NewsItem) -> None:
            toks = self._tokens_for_item(item)
            rep_tokens_by_key[key] = toks
            for token in toks:
                token_to_keys.setdefault(token, set()).add(key)

        def novelty_against_index(items: Sequence[NewsItem]) -> float:
            if not items:
                return 0.0
            if not rep_tokens_by_key:
                return float(len(items))
            total = 0.0
            for item in items:
                toks = self._tokens_for_item(item)
                overlaps: dict[str, int] = {}
                for token in toks:
                    for key in token_to_keys.get(token, ()):
                        overlaps[key] = overlaps.get(key, 0) + 1
                max_sim = 0.0
                for key, inter in overlaps.items():
                    denom = len(toks) + len(rep_tokens_by_key[key]) - inter
                    if denom <= 0:
                        continue
                    max_sim = max(max_sim, inter / denom)
                    if max_sim >= 1.0:
                        break
                total += max(0.0, 1.0 - max_sim)
            return total

        def add_prior_item(d: Date, item: NewsItem) -> None:
            key = item.dedup_key
            q = by_key.setdefault(key, deque())
            was_empty = not q
            q.append(item)
            window.append((d, item))
            if was_empty:
                add_rep(key, item)

        def remove_oldest_prior() -> None:
            _d, item = window.popleft()
            key = item.dedup_key
            q = by_key[key]
            was_rep = q and q[0] == item
            q.popleft()
            if was_rep:
                remove_rep(key)
            if q:
                if was_rep:
                    add_rep(key, q[0])
            else:
                by_key.pop(key, None)

        start_date = _to_date(start_date) if start_date is not None else None
        start_idx = 0
        if start_date is not None:
            lookback = max(self._config.novelty_lookback, self._config.baseline_lookback)
            target_idx = bisect.bisect_left(self._trading_dates, start_date)
            start_idx = max(0, target_idx - lookback)
        for d in self._trading_dates[start_idx:]:
            cutoff = self._prior_dates(ticker, d, self._config.novelty_lookback)
            min_keep = cutoff[0] if cutoff else d
            while window and window[0][0] < min_keep:
                remove_oldest_prior()

            key = (ticker, d)
            if key not in self._daily_cache:
                raw_items = self._news_panel.items_on(ticker, d)
                deduped = self._news_panel.dedup_items(raw_items)
                raw_count = len(raw_items)
                dedup_count = len(deduped)
                dup_share = 0.0 if raw_count == 0 else 1.0 - dedup_count / raw_count
                total_len = sum(item.length for item in deduped)
                novelty_sum = novelty_against_index(deduped)
                self._daily_cache[key] = NewsAttentionFeatures(
                    raw_count=raw_count,
                    dedup_count=dedup_count,
                    dup_share=dup_share,
                    total_len=total_len,
                    novelty_sum=novelty_sum,
                )

            for item in self._news_panel.dedup_items(self._news_panel.items_on(ticker, d)):
                add_prior_item(d, item)

    def daily_features(self, ticker: str, t) -> NewsAttentionFeatures:
        """Return same-day raw attention/novelty features before abnormal z-scoring."""
        t = _to_date(t)
        key = (ticker, t)
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        raw_items = self._news_panel.items_on(ticker, t)
        deduped = self._news_panel.dedup_items(raw_items)
        raw_count = len(raw_items)
        dedup_count = len(deduped)
        dup_share = 0.0 if raw_count == 0 else 1.0 - dedup_count / raw_count
        total_len = sum(item.length for item in deduped)
        novelty_sum = self._novelty_sum_for_items(ticker, t, deduped)
        features = NewsAttentionFeatures(
            raw_count=raw_count,
            dedup_count=dedup_count,
            dup_share=dup_share,
            total_len=total_len,
            novelty_sum=novelty_sum,
        )
        self._daily_cache[key] = features
        return features

    def features(self, ticker: str, t) -> NewsAttentionFeatures:
        """Return firm-day B1 features including prior-only abnormal components."""
        t = _to_date(t)
        key = (ticker, t)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        current = self.daily_features(ticker, t)
        prior_dates = self._prior_dates(ticker, t, self._config.baseline_lookback)
        if len(prior_dates) < self._config.min_history:
            features = NewsAttentionFeatures(
                raw_count=current.raw_count,
                dedup_count=current.dedup_count,
                dup_share=current.dup_share,
                total_len=current.total_len,
                novelty_sum=current.novelty_sum,
            )
            self._feature_cache[key] = features
            return features

        history = [self.daily_features(ticker, d) for d in prior_dates]
        count_hist = [math.log1p(f.dedup_count) for f in history]
        len_hist = [math.log1p(f.total_len) for f in history]
        novelty_hist = [math.log1p(f.novelty_sum) for f in history]
        abn_count = _zscore(math.log1p(current.dedup_count), count_hist)
        abn_len = _zscore(math.log1p(current.total_len), len_hist)
        abn_novelty = _zscore(math.log1p(current.novelty_sum), novelty_hist)
        raw_score = mean([abn_count, abn_len, abn_novelty])
        features = NewsAttentionFeatures(
            raw_count=current.raw_count,
            dedup_count=current.dedup_count,
            dup_share=current.dup_share,
            total_len=current.total_len,
            novelty_sum=current.novelty_sum,
            abn_count=abn_count,
            abn_len=abn_len,
            abn_novelty=abn_novelty,
            raw_score=raw_score,
        )
        self._feature_cache[key] = features
        return features

    def raw_score(self, ticker: str, t) -> float:
        return self.features(ticker, t).raw_score

    def score(self, observation: Observation) -> Sequence[Signal]:
        # raw_score is already a TIME-SERIES abnormal score (mean of the
        # prior-only z-scores of count/length/novelty vs the name's own
        # history). Emit it directly: a second cross-sectional normalisation
        # here degenerates to 0.0 for a single-name universe and silenced the
        # attention channel entirely.
        t = observation.decision_date
        return [
            Signal(ticker=ticker, value=float(self.raw_score(ticker, t)),
                   source_id=self.source_id, as_of=t)
            for ticker in observation.tickers
        ]
