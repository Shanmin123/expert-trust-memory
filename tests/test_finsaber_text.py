"""Tests for FINSABER text availability probes and causal text access."""

from __future__ import annotations

from datetime import date

import pytest

from trading_agent._bootstrap import FinsaberDataset
from trading_agent.data.finsaber_text import LoaderTextPanel, probe_text_data, require_text_data


def _bar(c=100.0):
    return {"open": c, "high": c, "low": c, "close": c, "adjusted_close": c, "volume": 1_000_000}


def test_probe_text_data_counts_folder_partitions(tmp_path):
    root = tmp_path / "finsaber"
    (root / "price_daily").mkdir(parents=True)
    (root / "news_items" / "year=2024").mkdir(parents=True)
    (root / "news_items" / "year=2024" / "part-000.parquet").write_bytes(b"")
    (root / "filingk" / "year=2024").mkdir(parents=True)
    (root / "filingk" / "year=2024" / "part-000.parquet").write_bytes(b"")
    (root / "news_text_index.sqlite").write_bytes(b"sqlite")

    avail = probe_text_data(str(root))

    assert avail.has_news
    assert avail.has_filingk
    assert not avail.has_filingq
    assert avail.present_folders == ("news_items", "filingk")
    assert avail.sqlite_index_path is not None
    assert avail.loader_modalities() == ("price", "news", "filing_k")


def test_require_text_data_raises_clear_blocker(tmp_path):
    root = tmp_path / "finsaber"
    (root / "price_daily").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="FINSABER text data is required"):
        require_text_data(str(root), min_modalities=("news_items",))


def test_loader_text_panel_is_causal_and_reports_coverage():
    ds = FinsaberDataset(data={
        date(2024, 1, 2): {
            "price": {"AAA": _bar(100), "BBB": _bar(100)},
            "news": {"AAA": ["AAA reports strong profit growth"]},
        },
        date(2024, 1, 3): {
            "price": {"AAA": _bar(101), "BBB": _bar(99)},
            "news": {"BBB": ["BBB faces weak sales warning"]},
        },
        date(2024, 1, 4): {
            "price": {"AAA": _bar(102), "BBB": _bar(98)},
        },
    })
    panel = LoaderTextPanel(ds)

    assert [it.text for it in panel.trailing_items("AAA", date(2024, 1, 2), lookback=1)] == [
        "AAA reports strong profit growth"
    ]
    assert panel.trailing_items("BBB", date(2024, 1, 2), lookback=2) == []  # future 2024-01-03 news excluded
    assert panel.count_items("BBB", date(2024, 1, 3), lookback=1) == 1
    frac, total = panel.coverage(["AAA", "BBB"], date(2024, 1, 3), lookback=2)
    assert frac == 1.0
    assert total == 2
