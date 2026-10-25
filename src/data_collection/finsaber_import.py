from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.storage import connect_sqlite, quote_identifier
from data_collection.symbols import canonical_symbol


TEXT_TABLE = "investment_texts"
TEXT_DATASET_PARTS = {"news_items", "filingk", "filingq"}
BATCH_SIZE = 1_000
DOC_TYPE_BY_SPLIT = {
    "news_items": "news",
    "filingk": "10-K",
    "filingq": "10-Q",
}


def ensure_text_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(TEXT_TABLE)} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_file TEXT NOT NULL,
            external_id TEXT,
            symbol TEXT,
            published_at TEXT,
            doc_type TEXT,
            title TEXT,
            text TEXT,
            metadata_json TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_text_symbol_time
        ON {quote_identifier(TEXT_TABLE)} (symbol, published_at)
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_text_doc_type_time
        ON {quote_identifier(TEXT_TABLE)} (doc_type, published_at)
        """
    )


def _first_present(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def infer_doc_type(source_file: Path) -> str | None:
    split = infer_split(source_file)
    if split:
        return DOC_TYPE_BY_SPLIT[split]
    return None


def infer_split(path: Path) -> str | None:
    parts = set(path.parts)
    for split in TEXT_DATASET_PARTS:
        if split in parts:
            return split
    return None


def is_text_source_file(path: Path, splits: set[str] | None = None) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".parquet"}:
        return False
    split = infer_split(path)
    if splits is not None and split not in splits:
        return False
    return split is not None or path.suffix.lower() in {
        ".json",
        ".jsonl",
        ".csv",
    }


def normalize_text_record(record: dict[str, Any], source_file: Path) -> tuple:
    raw_symbol = _first_present(record, ("symbol", "ticker", "stock", "permno"))
    symbol = canonical_symbol(raw_symbol) if raw_symbol else None
    published_at = _first_present(
        record,
        (
            "publishedAt",
            "published_at",
            "datetime",
            "date",
            "time",
            "timestamp",
        ),
    )
    if published_at is not None:
        published_at = str(pd.to_datetime(published_at, errors="coerce"))
        if published_at == "NaT":
            published_at = None

    title = _first_present(record, ("title", "headline", "name"))
    text = _first_present(
        record, ("text", "body", "content", "summary", "news_text", "filing_text")
    )
    doc_type = _first_present(record, ("doc_type", "type", "category")) or infer_doc_type(
        source_file
    )
    external_id = _first_present(
        record, ("id", "article_id", "accession_number", "accession", "item_index")
    )
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"text", "body", "content", "summary", "news_text", "filing_text"}
    }

    return (
        "FINSABER",
        str(source_file),
        None if external_id is None else str(external_id),
        symbol,
        published_at,
        None if doc_type is None else str(doc_type),
        None if title is None else str(title),
        None if text is None else str(text),
        json.dumps(metadata, ensure_ascii=True, default=str),
    )


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        if all(isinstance(item, dict) for item in value.values()):
            yield from value.values()
        else:
            yield value


def iter_tabular_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            yield record
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for FINSABER parquet imports. "
                "Install it with: pip install pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            frame = batch.to_pandas()
            for record in frame.to_dict(orient="records"):
                yield record
    else:
        return


def insert_text_rows(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    conn.executemany(
        f"""
        INSERT INTO {quote_identifier(TEXT_TABLE)}
            (source, source_file, external_id, symbol, published_at,
             doc_type, title, text, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def delete_doc_types(conn: sqlite3.Connection, splits: set[str]) -> int:
    doc_types = [DOC_TYPE_BY_SPLIT[split] for split in sorted(splits)]
    if not doc_types:
        return 0
    placeholders = ",".join("?" for _ in doc_types)
    cursor = conn.execute(
        f"DELETE FROM {quote_identifier(TEXT_TABLE)} WHERE doc_type IN ({placeholders})",
        doc_types,
    )
    conn.commit()
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def import_finsaber_texts(
    source_dir: Path,
    output_db: Path,
    *,
    splits: set[str] | None = None,
    force: bool = False,
    replace_splits: bool = False,
) -> int:
    source_dir.mkdir(parents=True, exist_ok=True)
    if force and output_db.exists():
        output_db.unlink()
    conn = connect_sqlite(output_db)
    ensure_text_table(conn)
    total = 0
    try:
        if replace_splits and splits:
            removed = delete_doc_types(conn, splits)
            print(f"Removed {removed} existing rows for splits: {', '.join(sorted(splits))}")

        files = [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and is_text_source_file(path, splits)
        ]
        for index, path in enumerate(files, start=1):
            if path.suffix.lower() in {".json", ".jsonl"}:
                records = iter_json_records(path)
            else:
                records = iter_tabular_records(path)
            rows: list[tuple] = []
            for record in records:
                rows.append(normalize_text_record(record, path))
                if len(rows) >= BATCH_SIZE:
                    total += insert_text_rows(conn, rows)
                    rows = []
            total += insert_text_rows(conn, rows)
            split = infer_split(path) or "text"
            print(f"[{index}/{len(files)}] indexed {split}: {path}")
    finally:
        conn.close()
    return total


def import_text(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    ensure_project_dirs(settings)
    source = Path(args.source) if args.source else settings.finsaber_dir
    output = (
        Path(args.output)
        if args.output
        else settings.finsaber_dir / "news_text_index.sqlite"
    )
    splits = set(args.splits) if args.splits else None
    total = import_finsaber_texts(
        source,
        output,
        splits=splits,
        force=args.force,
        replace_splits=args.replace_splits,
    )
    print(f"Indexed {total} FINSABER investment text records into {output}.")
    if total == 0:
        print("No local FINSABER text files were found yet; schema is ready.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FINSABER text importer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("import-text", help="Index local FINSABER texts")
    command.add_argument("--source")
    command.add_argument("--output")
    command.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(TEXT_DATASET_PARTS),
        help="Restrict import to selected FINSABER splits.",
    )
    command.add_argument("--force", action="store_true", help="Delete the output DB first.")
    command.add_argument(
        "--replace-splits",
        action="store_true",
        help="Delete existing rows for the selected splits before importing them.",
    )
    command.set_defaults(func=import_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
