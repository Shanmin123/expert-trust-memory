"""Tests for FINSABER filing panel and filing-text signals (fundamental expert)."""

from __future__ import annotations

import os
from datetime import date

import pytest

from trading_agent.alpha.filing_signals import FilingSignalAlpha
from trading_agent.data.finsaber_filings import FilingItem, FinsaberFilingPanel


def _panel():
    items = [
        FilingItem(date(2018, 2, 1), "AAA", "10-Q", "alpha beta gamma one two three"),
        FilingItem(date(2018, 11, 5), "AAA", "10-K", "alpha beta gamma delta epsilon"),
        FilingItem(date(2019, 2, 1), "AAA", "10-Q", "alpha beta gamma one two three four"),
        FilingItem(date(2019, 11, 5), "AAA", "10-K", "zeta eta theta iota kappa lambda"),
    ]
    return FinsaberFilingPanel.from_items(items)


def test_latest_on_is_point_in_time():
    p = _panel()
    assert p.latest_on("AAA", date(2018, 1, 1)) is None  # nothing filed yet
    assert p.latest_on("AAA", date(2018, 6, 1)).form == "10-Q"
    assert p.latest_on("AAA", date(2018, 11, 5)).form == "10-K"   # filing date is usable same day
    assert p.latest_on("AAA", date(2020, 1, 1)).date == date(2019, 11, 5)
    assert p.latest_on("MISSING", date(2020, 1, 1)) is None


def test_latest_and_prior_pairs_same_form():
    p = _panel()
    latest, prior = p.latest_and_prior("AAA", date(2019, 12, 1))
    assert latest.form == "10-K" and latest.date == date(2019, 11, 5)
    assert prior.form == "10-K" and prior.date == date(2018, 11, 5)  # prior 10-K, not the 10-Qs


def test_filing_signals_textual_change_and_recency():
    alpha = FilingSignalAlpha(_panel())

    # After the 2019 10-K (very different from the 2018 10-K) -> high text_change.
    s = alpha.signals_for("AAA", date(2019, 11, 20))
    assert s["available"] is True
    assert s["form"] == "10-K"
    assert s["days_since_filing"] == 15
    assert s["has_prior"] is True
    assert 0.0 <= s["similarity_to_prior"] <= 1.0
    assert s["text_change"] == pytest.approx(1.0 - s["similarity_to_prior"])
    # 2019 10-K shares no tokens with 2018 10-K -> similarity 0, change 1.
    assert s["similarity_to_prior"] == pytest.approx(0.0)

    # The 2019 10-Q barely changed from the 2018 10-Q -> high similarity.
    s_q = alpha.signals_for("AAA", date(2019, 2, 10))
    assert s_q["form"] == "10-Q"
    assert s_q["similarity_to_prior"] > 0.5


def test_filing_signals_expose_cosine_tfidf_change():
    # The TF-IDF cosine change (Lazy Prices, sharper than Jaccard) is exposed alongside
    # the Jaccard text_change and tracks the same direction: a rewritten 10-K reads as a
    # large change (low similarity), a near-identical 10-Q as a small one.
    alpha = FilingSignalAlpha(_panel())

    s_k = alpha.signals_for("AAA", date(2019, 11, 20))   # rewritten 10-K
    s_q = alpha.signals_for("AAA", date(2019, 2, 10))    # near-identical 10-Q
    if s_k["cosine_similarity_to_prior"] is None:
        pytest.skip("sklearn not installed; cosine falls back to None")
    for s in (s_k, s_q):
        assert 0.0 <= s["cosine_similarity_to_prior"] <= 1.0 + 1e-9
        assert s["cosine_text_change"] == pytest.approx(1.0 - s["cosine_similarity_to_prior"])
    # rewritten filing changed more than the near-identical one, on BOTH metrics
    assert s_k["cosine_text_change"] > s_q["cosine_text_change"]


def test_lm_tone_uses_vendored_loughran_mcdonald_dictionary():
    neg = "The company reported losses, litigation, impairment, adverse material weakness."
    pos = "The company achieved strong profit growth, gains, and successful improvements."
    items = [
        FilingItem(date(2019, 3, 1), "NEG", "10-K", "prior year filler"),
        FilingItem(date(2020, 3, 1), "NEG", "10-K", neg),
        FilingItem(date(2019, 3, 1), "POS", "10-K", "prior year filler"),
        FilingItem(date(2020, 3, 1), "POS", "10-K", pos),
    ]
    alpha = FilingSignalAlpha(FinsaberFilingPanel.from_items(items))
    s_neg = alpha.signals_for("NEG", date(2020, 4, 1))
    s_pos = alpha.signals_for("POS", date(2020, 4, 1))
    assert s_neg["lm_available"] is True
    assert 0.0 <= s_neg["lm_negative"] <= 1.0
    assert 0.0 <= s_neg["lm_uncertainty"] <= 1.0
    # The negative filing scores more negative words and a lower polarity.
    assert s_neg["lm_negative"] > s_pos["lm_negative"]
    assert s_neg["lm_polarity"] < s_pos["lm_polarity"]


def test_lm_tone_delta_compares_with_the_prior_same_form_filing():
    # Filings are systematically negative in LM terms, so the fundamental
    # expert reads the CHANGE versus the prior same-form filing, not the level.
    # Reuse sentences already proven (in the dictionary test above) to register
    # as LM-positive / LM-negative respectively.
    positive_text = "The company achieved strong profit growth, gains, and successful improvements."
    negative_text = "The company reported losses, litigation, impairment, adverse material weakness."
    worse = [
        FilingItem(date(2019, 3, 1), "W", "10-K", positive_text),
        FilingItem(date(2020, 3, 1), "W", "10-K", negative_text),
    ]
    better = [
        FilingItem(date(2019, 3, 1), "B", "10-K", negative_text),
        FilingItem(date(2020, 3, 1), "B", "10-K", positive_text),
    ]
    alpha = FilingSignalAlpha(FinsaberFilingPanel.from_items(worse + better))
    s_worse = alpha.signals_for("W", date(2020, 4, 1))
    s_better = alpha.signals_for("B", date(2020, 4, 1))
    assert s_worse["lm_negative_delta"] > 0      # tone deteriorated
    assert s_worse["lm_polarity_delta"] < 0
    assert s_better["lm_negative_delta"] < 0     # tone improved
    assert s_better["lm_polarity_delta"] > 0


def test_lm_tone_delta_is_none_without_a_prior_filing():
    alpha = FilingSignalAlpha(_panel())
    s = alpha.signals_for("AAA", date(2018, 3, 1))  # first filing, no prior
    assert s["lm_negative_delta"] is None
    assert s["lm_polarity_delta"] is None


def test_lm_tone_is_none_for_empty_text():
    from trading_agent.alpha.filing_signals import _lm_tone

    assert _lm_tone("") == (None, None, None)
    assert _lm_tone("losses litigation")[0] > 0  # non-empty scores


def test_filing_signals_absent_before_first_filing():
    alpha = FilingSignalAlpha(_panel())
    assert alpha.signals_for("AAA", date(2017, 1, 1)) == {"available": False}
    assert alpha.signals_for("MISSING", date(2020, 1, 1)) == {"available": False}


def test_first_filing_has_no_prior():
    alpha = FilingSignalAlpha(_panel())
    s = alpha.signals_for("AAA", date(2018, 3, 1))  # only the first 10-Q exists
    assert s["available"] is True
    assert s["has_prior"] is False
    assert s["similarity_to_prior"] is None
    assert s["text_change"] is None


_DATA_ROOT = "data/finsaber/sp500_2000_2025_parquet"


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_DATA_ROOT, "filingk")),
    reason="FINSABER filing data not present",
)
def test_real_filings_load_and_score_for_a_known_ticker():
    panel = FinsaberFilingPanel(_DATA_ROOT, date(2016, 1, 1), date(2020, 12, 31), tickers=["AAPL"])
    alpha = FilingSignalAlpha(panel)
    s = alpha.signals_for("AAPL", date(2020, 6, 1))
    assert s["available"] is True
    assert s["form"] in {"10-K", "10-Q"}
    assert s["length"] > 10000            # real filings are large
    assert s["has_prior"] is True
    assert 0.0 <= s["similarity_to_prior"] <= 1.0
