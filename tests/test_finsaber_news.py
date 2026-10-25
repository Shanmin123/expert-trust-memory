"""Tests for the project-owned FINSABER news_items metadata panel."""

from __future__ import annotations

from datetime import date

from trading_agent.data.finsaber_news import FinsaberNewsPanel, NewsItem


def test_news_item_length_and_dedup_fallbacks():
    with_len = NewsItem(date(2024, 1, 2), "AAA", "short", text_len=123, text_crc32=7)
    no_len = NewsItem(date(2024, 1, 2), "AAA", "abcdef", text_len=None, text_crc32=None)

    assert with_len.length == 123
    assert with_len.dedup_key == "crc32:7"
    assert no_len.length == 6
    assert no_len.dedup_key.startswith("sha1:")


def test_news_panel_preserves_metadata_and_items_on():
    panel = FinsaberNewsPanel.from_items([
        NewsItem(date(2024, 1, 2), "AAA", "first", cik="123", item_index=2, text_len=5, text_crc32=22),
        NewsItem(date(2024, 1, 2), "AAA", "second", cik="123", item_index=1, text_len=6, text_crc32=11),
    ])

    items = panel.items_on("AAA", date(2024, 1, 2))

    assert [it.item_index for it in items] == [1, 2]
    assert items[0].cik == "123"
    assert items[0].text_len == 6
    assert items[0].text_crc32 == 11


def test_news_panel_exact_dedup_by_crc32_and_fallback_hash():
    dup_a = NewsItem(date(2024, 1, 2), "AAA", "same", text_crc32=99)
    dup_b = NewsItem(date(2024, 1, 2), "AAA", "same syndicated copy", text_crc32=99)
    no_crc_a = NewsItem(date(2024, 1, 2), "AAA", "fallback same", text_crc32=None)
    no_crc_b = NewsItem(date(2024, 1, 2), "AAA", "fallback same", text_crc32=None)

    deduped = FinsaberNewsPanel.dedup_items([dup_a, dup_b, no_crc_a, no_crc_b])

    assert deduped == [dup_a, no_crc_a]


def test_news_panel_trailing_items_excludes_current_by_default():
    panel = FinsaberNewsPanel.from_items([
        NewsItem(date(2024, 1, 2), "AAA", "past1", text_crc32=1),
        NewsItem(date(2024, 1, 3), "AAA", "past2", text_crc32=2),
        NewsItem(date(2024, 1, 4), "AAA", "today", text_crc32=3),
        NewsItem(date(2024, 1, 4), "BBB", "other ticker", text_crc32=4),
    ])

    strict_prior = panel.trailing_items("AAA", date(2024, 1, 4), lookback=5)
    through_today = panel.trailing_items("AAA", date(2024, 1, 4), lookback=5, include_current=True)

    assert [it.text for it in strict_prior] == ["past1", "past2"]
    assert [it.text for it in through_today] == ["past1", "past2", "today"]


def test_news_panel_coverage_reports_raw_and_dedup_counts():
    panel = FinsaberNewsPanel.from_items([
        NewsItem(date(2024, 1, 2), "AAA", "a", text_crc32=1),
        NewsItem(date(2024, 1, 2), "AAA", "a dup", text_crc32=1),
        NewsItem(date(2024, 1, 2), "BBB", "b", text_crc32=2),
    ])

    coverage, raw, dedup = panel.coverage(["AAA", "BBB", "CCC"], date(2024, 1, 2))

    assert coverage == 2 / 3
    assert raw == 3
    assert dedup == 2
