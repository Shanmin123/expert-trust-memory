import sqlite3

import pandas as pd

from data_collection.config import Settings
from data_collection.preview_data import export_csv_previews, export_finsaber_previews
from data_collection.qlib_provider import qlib_table_name
from data_collection.storage import price_table_name, quote_identifier


def make_settings(tmp_path):
    return Settings(
        ib_host="127.0.0.1",
        ib_port=4002,
        ib_client_id=1,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        rapidapi_key=None,
        rapidapi_host="example.com",
    )


def test_export_csv_previews(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)

    with sqlite3.connect(settings.prices_db) as conn:
        table = price_table_name("AAPL")
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                    "barCount": 10,
                    "average": 1.4,
                    "symbol": "AAPL",
                }
            ]
        ).to_sql(table, conn, index=False)

    with sqlite3.connect(settings.indicators_db) as conn:
        pd.DataFrame(
            [{"symbol": "AAPL", "date": "2024-01-02", "indicator_name": "SMA_20", "value": None}]
        ).to_sql("indicators", conn, index=False)

    with sqlite3.connect(settings.alphas_db) as conn:
        pd.DataFrame(
            [{"symbol": "AAPL", "date": "2024-01-02", "alpha_id": "alpha_1", "value": 0.1}]
        ).to_sql("alphas", conn, index=False)

    with sqlite3.connect(settings.qlib_direct_factors_db) as conn:
        table = qlib_table_name("AAPL")
        pd.DataFrame(
            [{"symbol": "AAPL", "date": "2024-01-02", "QLIB158_KMID": 0.2}]
        ).to_sql(table, conn, index=False)

    written = export_csv_previews(settings, symbol="AAPL")

    assert set(written) == {"prices", "indicators", "alpha101", "qlib"}
    assert (settings.reports_dir / "previews" / "prices_AAPL.csv").exists()
    assert pd.read_csv(written["prices"])["close"].tolist() == [1.5]
    assert pd.read_csv(written["indicators"])["indicator_name"].tolist() == ["SMA_20"]
    assert pd.read_csv(written["alpha101"])["alpha_id"].tolist() == ["alpha_1"]
    assert pd.read_csv(written["qlib"])["QLIB158_KMID"].tolist() == [0.2]


def test_export_finsaber_previews(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.finsaber_dir.mkdir(parents=True)

    rows = [
        {
            "source": "FINSABER",
            "source_file": "sample.parquet",
            "external_id": "n1",
            "symbol": "AAPL",
            "published_at": "2024-01-02",
            "doc_type": "news",
            "title": None,
            "text": "sample news",
            "metadata_json": "{}",
        },
        {
            "source": "FINSABER",
            "source_file": "sample.parquet",
            "external_id": "k1",
            "symbol": "AAPL",
            "published_at": "2024-01-03",
            "doc_type": "10-K",
            "title": None,
            "text": "sample 10-k",
            "metadata_json": "{}",
        },
        {
            "source": "FINSABER",
            "source_file": "sample.parquet",
            "external_id": "q1",
            "symbol": "AAPL",
            "published_at": "2024-01-04",
            "doc_type": "10-Q",
            "title": None,
            "text": "sample 10-q",
            "metadata_json": "{}",
        },
    ]
    with sqlite3.connect(settings.finsaber_dir / "news_text_index.sqlite") as conn:
        pd.DataFrame(rows).to_sql("investment_texts", conn, index=False)

    written = export_finsaber_previews(settings, limit=5)

    assert pd.read_csv(written["finsaber_news"])["text_sample"].tolist() == ["sample news"]
    assert pd.read_csv(written["finsaber_10k"])["text_sample"].tolist() == ["sample 10-k"]
    assert pd.read_csv(written["finsaber_10q"])["text_sample"].tolist() == ["sample 10-q"]
