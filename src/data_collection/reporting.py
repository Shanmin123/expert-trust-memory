from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import connect_sqlite, count_rows, list_tables, quote_identifier


def db_status(path: Path, prefix: str | None = None) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    conn = connect_sqlite(path)
    try:
        tables = [
            table for table in list_tables(conn, prefix) if not table.startswith("sqlite_")
        ]
        rows = sum(count_rows(conn, table) for table in tables)
        return True, len(tables), rows
    finally:
        conn.close()


def prefixed_table_date_range(
    path: Path, prefix: str, date_column: str = "date"
) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    conn = connect_sqlite(path)
    try:
        start_dates: list[str] = []
        end_dates: list[str] = []
        for table in list_tables(conn, prefix):
            if table.startswith("sqlite_"):
                continue
            row = conn.execute(
                f"SELECT MIN({quote_identifier(date_column)}), "
                f"MAX({quote_identifier(date_column)}) FROM {quote_identifier(table)}"
            ).fetchone()
            if row and row[0]:
                start_dates.append(str(row[0])[:10])
            if row and row[1]:
                end_dates.append(str(row[1])[:10])
        return (
            min(start_dates) if start_dates else None,
            max(end_dates) if end_dates else None,
        )
    finally:
        conn.close()


def long_table_date_range(path: Path, table: str, date_column: str = "date") -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    conn = connect_sqlite(path)
    try:
        if table not in list_tables(conn):
            return None, None
        row = conn.execute(
            f"SELECT MIN({quote_identifier(date_column)}), "
            f"MAX({quote_identifier(date_column)}) FROM {quote_identifier(table)}"
        ).fetchone()
        return (
            str(row[0])[:10] if row and row[0] else None,
            str(row[1])[:10] if row and row[1] else None,
        )
    finally:
        conn.close()


def finsaber_split_year_range(settings: Settings, split: str) -> tuple[int | None, int | None]:
    root = settings.finsaber_dir / "FINSABER-V2-Data" / split
    if not root.exists():
        return None, None
    years: list[int] = []
    for path in root.glob("year=*"):
        try:
            years.append(int(path.name.split("=", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (min(years), max(years)) if years else (None, None)


def finsaber_text_index_status(
    settings: Settings,
) -> tuple[str | None, str | None, int, dict[str, int]]:
    index_path = settings.finsaber_dir / "news_text_index.sqlite"
    if not index_path.exists():
        return None, None, 0, {}
    conn = connect_sqlite(index_path)
    try:
        if "investment_texts" not in list_tables(conn):
            return None, None, 0, {}
        row = conn.execute(
            "SELECT MIN(published_at), MAX(published_at), COUNT(*) "
            "FROM investment_texts"
        ).fetchone()
        doc_type_counts = {
            str(doc_type): int(count)
            for doc_type, count in conn.execute(
                "SELECT doc_type, COUNT(*) FROM investment_texts GROUP BY doc_type"
            ).fetchall()
        }
        return (
            str(row[0])[:10] if row and row[0] else None,
            str(row[1])[:10] if row and row[1] else None,
            int(row[2]) if row else 0,
            doc_type_counts,
        )
    finally:
        conn.close()


def universe_status(path: Path) -> tuple[bool, int, str | None, str | None]:
    if not path.exists():
        return False, 0, None, None

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return True, 0, None, None
    return (
        True,
        len(rows),
        rows[0].get("source"),
        rows[0].get("source_as_of"),
    )


def write_data_collection_summary(settings: Settings | None = None) -> Path:
    settings = settings or Settings.from_env()
    ensure_project_dirs(settings)

    price_exists, price_tables, price_rows = db_status(settings.prices_db, "prices_")
    price_start, price_end = prefixed_table_date_range(settings.prices_db, "prices_")
    indicator_exists, indicator_tables, indicator_rows = db_status(
        settings.indicators_db
    )
    indicator_start, indicator_end = long_table_date_range(
        settings.indicators_db, "indicators"
    )
    alpha_exists, alpha_tables, alpha_rows = db_status(settings.alphas_db)
    alpha_start, alpha_end = long_table_date_range(settings.alphas_db, "alphas")
    qlib_exists, qlib_tables, qlib_rows = db_status(
        settings.qlib_direct_factors_db, "qlib_direct_"
    )
    qlib_start, qlib_end = prefixed_table_date_range(
        settings.qlib_direct_factors_db, "qlib_direct_"
    )
    universe_exists, universe_rows, universe_source, universe_as_of = universe_status(
        settings.sp500_universe_snapshot
    )
    finsaber_index_paths = sorted(settings.finsaber_dir.glob("*.sqlite"))
    finsaber_exists = bool(finsaber_index_paths)
    finsaber_tables = 0
    finsaber_rows = 0
    for finsaber_db in finsaber_index_paths:
        _, table_count, row_count = db_status(finsaber_db)
        finsaber_tables += table_count
        finsaber_rows += row_count
    seeking_alpha_files = list(settings.seeking_alpha_dir.glob("*.json"))
    finsaber_price_years = finsaber_split_year_range(settings, "price_daily")
    finsaber_news_years = finsaber_split_year_range(settings, "news_items")
    finsaber_10k_years = finsaber_split_year_range(settings, "filingk")
    finsaber_10q_years = finsaber_split_year_range(settings, "filingq")
    (
        finsaber_text_start,
        finsaber_text_end,
        finsaber_text_rows,
        finsaber_doc_type_counts,
    ) = finsaber_text_index_status(settings)

    output = settings.reports_dir / "data_collection_summary.md"
    output.write_text(
        "\n".join(
            [
                "# Data Collection Summary",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Price Data",
                "",
                f"- Database present: {price_exists}",
                f"- Symbol tables: {price_tables}",
                f"- Rows: {price_rows}",
                f"- Actual IBKR date range: {price_start or 'n/a'} to {price_end or 'n/a'}",
                "- Forward-window role: IBKR is the source for 2024-01-01 onward where fresh price coverage is needed.",
                "",
                "## Universe Snapshot",
                "",
                f"- Snapshot present: {universe_exists}",
                f"- Rows: {universe_rows}",
                f"- Source: {universe_source or 'n/a'}",
                f"- Source as-of date: {universe_as_of or 'n/a'}",
                "",
                "## Technical Indicators",
                "",
                f"- Database present: {indicator_exists}",
                f"- Tables: {indicator_tables}",
                f"- Rows: {indicator_rows}",
                f"- Date range: {indicator_start or 'n/a'} to {indicator_end or 'n/a'}",
                "",
                "## Alpha101 Factors",
                "",
                f"- Database present: {alpha_exists}",
                f"- Tables: {alpha_tables}",
                f"- Rows: {alpha_rows}",
                f"- Date range: {alpha_start or 'n/a'} to {alpha_end or 'n/a'}",
                "- Requires Python >= 3.13 for official generation.",
                "",
                "## Official Qlib Factors",
                "",
                f"- Database present: {qlib_exists}",
                f"- Symbol tables: {qlib_tables}",
                f"- Rows: {qlib_rows}",
                f"- Date range: {qlib_start or 'n/a'} to {qlib_end or 'n/a'}",
                "- Computed with Qlib's Alpha158DL/Alpha360DL feature definitions and expression engine.",
                "",
                "## Combined Coverage Design",
                "",
                "- Historical baseline: FINSABER provides the 2000-2024 historical backtesting window.",
                "- Forward supplement: IBKR collection covers 2024-01-01 to the current latest trading date.",
                "- Combined design: FINSABER + IBKR are intended to form a continuous price sequence after overlap and gap validation.",
                "- Overlap year: 2024 should be used for IBKR-vs-FINSABER price consistency checks before modelling.",
                "",
                "## Investment Text",
                "",
                "- Primary source: FINSABER local dataset import.",
                f"- FINSABER raw price_daily years: {finsaber_price_years[0] or 'n/a'} to {finsaber_price_years[1] or 'n/a'}",
                f"- FINSABER raw news_items years: {finsaber_news_years[0] or 'n/a'} to {finsaber_news_years[1] or 'n/a'}",
                f"- FINSABER raw 10-K years: {finsaber_10k_years[0] or 'n/a'} to {finsaber_10k_years[1] or 'n/a'}",
                f"- FINSABER raw 10-Q years: {finsaber_10q_years[0] or 'n/a'} to {finsaber_10q_years[1] or 'n/a'}",
                f"- FINSABER index present: {finsaber_exists}",
                f"- FINSABER index files: {len(finsaber_index_paths)}",
                f"- FINSABER tables: {finsaber_tables}",
                f"- FINSABER rows: {finsaber_rows}",
                f"- FINSABER indexed text date range: {finsaber_text_start or 'n/a'} to {finsaber_text_end or 'n/a'}",
                f"- FINSABER indexed text rows: {finsaber_text_rows}",
                f"- FINSABER indexed news rows: {finsaber_doc_type_counts.get('news', 0)}",
                f"- FINSABER indexed 10-K rows: {finsaber_doc_type_counts.get('10-K', 0)}",
                f"- FINSABER indexed 10-Q rows: {finsaber_doc_type_counts.get('10-Q', 0)}",
                f"- Seeking Alpha cached article files: {len(seeking_alpha_files)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def summary(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    output = write_data_collection_summary(settings)
    print(f"Wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("summary")
    command.set_defaults(func=summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
