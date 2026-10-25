"""Tests for the reproducible random-universe selection (coverage + seed logic)."""

from __future__ import annotations

from datetime import date

import pytest

from scripts import select_random_universe as sru


@pytest.fixture
def _stub_coverage(monkeypatch):
    price = {f"N{i:02d}": 1.0 for i in range(20)}
    price.update({"TSLA": 1.0, "NFLX": 1.0, "AMZN": 1.0, "MSFT": 1.0})
    # Half the names lack news coverage.
    news = {name: 1000 + i for i, name in enumerate(sorted(price)) if i % 2 == 0}
    monkeypatch.setattr(sru, "_price_covered_names", lambda *a, **k: dict(price))
    monkeypatch.setattr(sru, "_news_covered_names", lambda *a, **k: dict(news))
    return price, news


def _select(n=4, seed=42):
    return sru.select_random_universe(
        "unused", start=date(2011, 7, 1), end=date(2025, 12, 31),
        n=n, seed=seed, min_coverage=0.9, min_news_dates=900,
    )


def test_pool_is_price_and_news_intersection_minus_reference(_stub_coverage):
    result = _select()
    # Reference names excluded even though they have price coverage.
    assert not (set(result["selected"]) & {"TSLA", "NFLX", "AMZN", "MSFT"})
    # Only news-covered names are eligible.
    price, news = _stub_coverage
    assert set(result["selected"]) <= set(news)


def test_selection_is_reproducible_under_fixed_seed(_stub_coverage):
    assert _select(seed=42)["selected"] == _select(seed=42)["selected"]
    # A different seed generally gives a different draw (pool is large enough).
    assert _select(seed=1)["selected"] != _select(seed=999)["selected"]


def test_manifest_records_coverage_and_thresholds(_stub_coverage):
    result = _select(n=4, seed=7)
    assert result["seed"] == 7
    assert result["thresholds"] == {"min_coverage": 0.9, "min_news_dates": 900}
    assert set(result["coverage"]) == set(result["selected"])
    for stats in result["coverage"].values():
        assert "price_coverage" in stats and "news_dates" in stats


def test_raises_when_pool_too_small(_stub_coverage):
    with pytest.raises(ValueError, match="eligible pool"):
        _select(n=999)
