"""Deterministic filing-text signals for the fundamental expert.

From each name's most recent 10-K / 10-Q, compute point-in-time signals that are
orthogonal to price: the finance-specific tone of the filing (Loughran and
McDonald's negative-word proportion and polarity, which the finance literature
links to subsequent returns), the textual change versus the prior same-form
filing (the "Lazy Prices" effect of Cohen, Malloy and Nguyen, where large
year-on-year filing changes precede lower returns), the change in filing length,
and the recency of the filing. The signals are deterministic and need no
per-filing language-model call; the fundamental expert reads them alongside its
market-level inputs.

Textual change reuses the token/Jaccard machinery already used for news novelty.
Tone uses the vendored Loughran-McDonald finance dictionary (see lm_dictionary.py)
with canonical exact word matching, so it is deterministic, reproducible, and free
of any runtime dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from trading_agent.alpha.lm_dictionary import LM_NEGATIVE, LM_POSITIVE, LM_UNCERTAINTY
from trading_agent.alpha.text_attention import _jaccard, _tokens
from trading_agent.data.finsaber_filings import FinsaberFilingPanel
from trading_agent.market import _to_date

SIGNAL_ID = "filing_text_change"

_LM_WORD_RE = re.compile(r"[a-z]+")


def _cosine_tfidf(text_a: str, text_b: str) -> Optional[float]:
    """Cosine similarity of two filings under a TF-IDF term weighting (Lazy Prices).

    Cohen-Malloy-Nguyen (2020) measure filing change by cosine similarity; TF-IDF
    down-weights shared boilerplate and up-weights distinctive terms, so the change
    signal is sharper than a set-overlap Jaccard (validated offline: |IC| ~0.10 at the
    monthly horizon vs ~0.03 for Jaccard). Fit on ONLY the two filings (both dated <= t),
    so it stays strictly point-in-time -- no future filing enters the IDF. ``sklearn`` is
    optional: when it is missing the caller falls back to the Jaccard similarity.
    """
    if not text_a or not text_b:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None
    try:
        m = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform([text_a, text_b])
        num = float((m[0] @ m[1].T).toarray()[0, 0])
        na = float(m[0].multiply(m[0]).sum()) ** 0.5
        nb = float(m[1].multiply(m[1]).sum()) ** 0.5
        return (num / (na * nb)) if (na > 0 and nb > 0) else None
    except Exception:
        return None


def _lm_tone(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Loughran-McDonald tone by exact word match: (negative_fraction, polarity, uncertainty_fraction).

    Polarity is (positive - negative) / (positive + negative) in [-1, 1]; the
    fractions are word counts over total words. Returns (None, None, None) for
    empty text.
    """
    tokens = _LM_WORD_RE.findall((text or "").lower())
    n = len(tokens)
    if n == 0:
        return None, None, None
    neg = sum(1 for t in tokens if t in LM_NEGATIVE)
    pos = sum(1 for t in tokens if t in LM_POSITIVE)
    unc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    polarity = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    return round(neg / n, 6), round(polarity, 6), round(unc / n, 6)


@dataclass(frozen=True)
class _PairSignal:
    form: str
    filing_date: Date
    length: int
    similarity_to_prior: Optional[float]
    cosine_similarity_to_prior: Optional[float]
    length_change: Optional[float]
    has_prior: bool
    lm_negative: Optional[float]
    lm_polarity: Optional[float]
    lm_uncertainty: Optional[float]
    # Tone CHANGE versus the prior same-form filing. Filings are systematically
    # negative in Loughran-McDonald terms (risk-factor sections), so the level
    # alone reads bearish everywhere; the delta carries the information.
    lm_negative_delta: Optional[float]
    lm_polarity_delta: Optional[float]


class FilingSignalAlpha:
    """Per-ticker filing-text signals with point-in-time filing selection."""

    def __init__(self, panel: FinsaberFilingPanel) -> None:
        self._panel = panel
        self._pair_cache: dict[tuple[str, Date], _PairSignal] = {}

    def _pair_signal(self, ticker: str, latest, prior) -> _PairSignal:
        key = (ticker, latest.date)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        similarity = None
        cosine_similarity = None
        length_change = None
        if prior is not None:
            similarity = _jaccard(_tokens(latest.text), _tokens(prior.text))
            cosine_similarity = _cosine_tfidf(latest.text, prior.text)
            if prior.length:
                length_change = latest.length / prior.length - 1.0
        lm_negative, lm_polarity, lm_uncertainty = _lm_tone(latest.text)
        lm_negative_delta = None
        lm_polarity_delta = None
        if prior is not None:
            prior_negative, prior_polarity, _ = _lm_tone(prior.text)
            if lm_negative is not None and prior_negative is not None:
                lm_negative_delta = round(lm_negative - prior_negative, 6)
            if lm_polarity is not None and prior_polarity is not None:
                lm_polarity_delta = round(lm_polarity - prior_polarity, 6)
        signal = _PairSignal(
            form=latest.form,
            filing_date=latest.date,
            length=latest.length,
            similarity_to_prior=None if similarity is None else round(similarity, 6),
            cosine_similarity_to_prior=None if cosine_similarity is None else round(cosine_similarity, 6),
            length_change=None if length_change is None else round(length_change, 6),
            has_prior=prior is not None,
            lm_negative=lm_negative,
            lm_polarity=lm_polarity,
            lm_uncertainty=lm_uncertainty,
            lm_negative_delta=lm_negative_delta,
            lm_polarity_delta=lm_polarity_delta,
        )
        self._pair_cache[key] = signal
        return signal

    def signals_for(self, ticker: str, t) -> dict:
        """Filing-text signals available for ``ticker`` at decision date ``t``."""
        t = _to_date(t)
        latest, prior = self._panel.latest_and_prior(ticker, t)
        if latest is None:
            return {"available": False}
        pair = self._pair_signal(ticker, latest, prior)
        similarity = pair.similarity_to_prior
        cosine = pair.cosine_similarity_to_prior
        return {
            "available": True,
            "form": pair.form,
            "filing_date": pair.filing_date.isoformat(),
            "days_since_filing": (t - pair.filing_date).days,
            "length": pair.length,
            "similarity_to_prior": similarity,
            "text_change": None if similarity is None else round(1.0 - similarity, 6),
            # Lazy-Prices change under TF-IDF weighting -- the sharper, primary change
            # signal (validated |IC| ~0.10 monthly vs ~0.03 for the Jaccard text_change).
            "cosine_similarity_to_prior": cosine,
            "cosine_text_change": None if cosine is None else round(1.0 - cosine, 6),
            "length_change": pair.length_change,
            "has_prior": pair.has_prior,
            "lm_negative": pair.lm_negative,
            "lm_polarity": pair.lm_polarity,
            "lm_uncertainty": pair.lm_uncertainty,
            "lm_negative_delta": pair.lm_negative_delta,
            "lm_polarity_delta": pair.lm_polarity_delta,
            "lm_available": pair.lm_negative is not None,
        }
