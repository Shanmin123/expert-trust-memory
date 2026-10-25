import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from data_collection.qlib_provider import export_qlib_provider


def create_price_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE prices_AAPL (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                barCount REAL,
                average REAL,
                symbol TEXT NOT NULL
            )
            """
        )
        rows = [
            ("2024-01-02", 99, 102, 98, 100, 1000, None, 100.25, "AAPL"),
            ("2024-01-03", 100, 103, 99, 101, 1100, None, 101.25, "AAPL"),
        ]
        conn.executemany("INSERT INTO prices_AAPL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_export_qlib_provider_writes_calendar_instruments_and_bins(tmp_path: Path) -> None:
    price_db = tmp_path / "prices.db"
    provider_dir = tmp_path / "qlib_provider"
    create_price_db(price_db)

    instruments, calendar_rows = export_qlib_provider(price_db, provider_dir, force=True)

    assert instruments == 1
    assert calendar_rows == 2
    assert (provider_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert (provider_dir / "instruments" / "sp500.txt").read_text(
        encoding="utf-8"
    ).splitlines() == ["AAPL\t2024-01-02\t2024-01-03"]

    close_bin = provider_dir / "features" / "aapl" / "close.day.bin"
    values = np.fromfile(close_bin, dtype="<f")
    assert values.tolist() == [0.0, 100.0, 101.0]


def test_export_qlib_provider_uses_vwap_from_average(tmp_path: Path) -> None:
    price_db = tmp_path / "prices.db"
    provider_dir = tmp_path / "qlib_provider"
    create_price_db(price_db)

    export_qlib_provider(price_db, provider_dir, force=True)

    vwap_bin = provider_dir / "features" / "aapl" / "vwap.day.bin"
    values = np.fromfile(vwap_bin, dtype="<f")
    assert values.tolist() == [0.0, 100.25, 101.25]
