from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_collection.symbols import canonical_symbol, sqlite_table_name


PRICE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "barCount",
    "average",
    "symbol",
]


def connect_sqlite(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def price_table_name(symbol: str) -> str:
    return sqlite_table_name("prices", symbol)


def ensure_price_table(conn: sqlite3.Connection, symbol: str) -> str:
    table = price_table_name(symbol)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(table)} (
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
    return table


def normalize_price_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    out = df.copy()
    if "date" not in out.columns and "time" in out.columns:
        out = out.rename(columns={"time": "date"})
    if "date" not in out.columns:
        raise ValueError("Price frame must include a date or time column.")

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["symbol"] = canonical_symbol(symbol)
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PRICE_COLUMNS].sort_values("date").drop_duplicates("date", keep="last")
    return out


def upsert_price_frame(
    conn: sqlite3.Connection, symbol: str, df: pd.DataFrame
) -> int:
    table = ensure_price_table(conn, symbol)
    normalized = normalize_price_frame(df, symbol)
    if normalized.empty:
        return 0

    rows = list(normalized.itertuples(index=False, name=None))
    placeholders = ",".join(["?"] * len(PRICE_COLUMNS))
    columns = ",".join(quote_identifier(col) for col in PRICE_COLUMNS)
    update_columns = [col for col in PRICE_COLUMNS if col != "date"]
    updates = ",".join(
        f"{quote_identifier(col)}=excluded.{quote_identifier(col)}"
        for col in update_columns
    )
    conn.executemany(
        f"""
        INSERT INTO {quote_identifier(table)} ({columns})
        VALUES ({placeholders})
        ON CONFLICT(date) DO UPDATE SET {updates}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_price_frame(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    table = price_table_name(symbol)
    if not table_exists(conn, table):
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.read_sql_query(
        f"SELECT * FROM {quote_identifier(table)} ORDER BY date", conn
    )


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def list_tables(conn: sqlite3.Connection, prefix: str | None = None) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    tables = [row[0] for row in rows]
    if prefix is not None:
        tables = [table for table in tables if table.startswith(prefix)]
    return tables


def list_price_symbols(conn: sqlite3.Connection) -> list[str]:
    symbols = []
    for table in list_tables(conn, "prices_"):
        row = conn.execute(
            f"SELECT symbol FROM {quote_identifier(table)} LIMIT 1"
        ).fetchone()
        if row:
            symbols.append(row[0])
    return sorted(set(symbols))


def get_last_price_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    table = price_table_name(symbol)
    if not table_exists(conn, table):
        return None
    row = conn.execute(
        f"SELECT MAX(date) FROM {quote_identifier(table)}"
    ).fetchone()
    return row[0] if row and row[0] else None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    if not table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()
    return int(row[0])


def ensure_long_table(
    conn: sqlite3.Connection,
    table: str,
    value_name: str,
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(table)} (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            {quote_identifier(value_name)} TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (symbol, date, {quote_identifier(value_name)})
        )
        """
    )


def upsert_long_values(
    conn: sqlite3.Connection,
    table: str,
    value_name: str,
    rows: Iterable[tuple[str, str, str, float | None]],
) -> int:
    ensure_long_table(conn, table, value_name)
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        f"""
        INSERT INTO {quote_identifier(table)}
            (symbol, date, {quote_identifier(value_name)}, value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(symbol, date, {quote_identifier(value_name)})
        DO UPDATE SET value=excluded.value
        """,
        rows,
    )
    conn.commit()
    return len(rows)
