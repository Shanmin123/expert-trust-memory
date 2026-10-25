from pathlib import Path

from data_collection.finsaber_import import (
    infer_doc_type,
    is_text_source_file,
    normalize_text_record,
)


def test_finsaber_v2_news_record_maps_text_and_doc_type() -> None:
    path = Path("FINSABER-V2-Data/news_items/year=2024/part-000.parquet")
    row = normalize_text_record(
        {
            "date": "2024-01-02",
            "symbol": "AAPL",
            "item_index": 3,
            "news_text": "Apple investment analysis text",
        },
        path,
    )
    assert row[3] == "AAPL"
    assert row[4].startswith("2024-01-02")
    assert row[5] == "news"
    assert row[7] == "Apple investment analysis text"
    assert "Apple investment analysis text" not in row[8]


def test_finsaber_v2_filing_record_maps_accession() -> None:
    path = Path("FINSABER-V2-Data/filingk/year=2024/part-000.parquet")
    row = normalize_text_record(
        {
            "date": "2024-01-02",
            "symbol": "MSFT",
            "accession": "0000000000-24-000001",
            "filing_text": "10-K text",
        },
        path,
    )
    assert row[2] == "0000000000-24-000001"
    assert row[5] == "10-K"
    assert row[7] == "10-K text"


def test_price_daily_parquet_is_not_text_source() -> None:
    assert not is_text_source_file(
        Path("FINSABER-V2-Data/price_daily/year=2024/part-000.parquet")
    )
    assert infer_doc_type(Path("FINSABER-V2-Data/filingq/year=2024/part-000.parquet")) == "10-Q"
