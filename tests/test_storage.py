import pandas as pd

from data_collection.storage import (
    connect_sqlite,
    get_last_price_date,
    list_price_symbols,
    load_price_frame,
    upsert_price_frame,
)


def test_price_upsert_and_readback(tmp_path) -> None:
    db_path = tmp_path / "prices.db"
    conn = connect_sqlite(db_path)
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
            "barCount": [10, 11],
            "average": [100.5, 101.5],
        }
    )
    assert upsert_price_frame(conn, "AAPL", frame) == 2
    assert get_last_price_date(conn, "AAPL") == "2024-01-03"
    assert list_price_symbols(conn) == ["AAPL"]

    loaded = load_price_frame(conn, "AAPL")
    assert loaded["symbol"].unique().tolist() == ["AAPL"]
    assert loaded["close"].tolist() == [101.0, 102.0]
    conn.close()
