from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_collection.config import Settings
from data_collection.qlib_provider import qlib_table_name
from data_collection.storage import price_table_name, quote_identifier
from data_collection.symbols import canonical_symbol


PREVIEW_FILENAMES = {
    "prices": "prices_{symbol}.csv",
    "indicators": "indicators_{symbol}.csv",
    "alpha101": "alpha101_{symbol}.csv",
    "qlib": "qlib_{symbol}.csv",
    "finsaber_news": "finsaber_news_sample.csv",
    "finsaber_10k": "finsaber_10k_sample.csv",
    "finsaber_10q": "finsaber_10q_sample.csv",
}


def _read_table(db_path: Path, query: str, params: tuple = ()) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def _table_exists(db_path: Path, table: str) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    return row is not None


def load_price_preview(settings: Settings, symbol: str) -> pd.DataFrame:
    table = price_table_name(symbol)
    if not _table_exists(settings.prices_db, table):
        return pd.DataFrame()
    return _read_table(
        settings.prices_db,
        f"SELECT * FROM {quote_identifier(table)} ORDER BY date",
    )


def load_indicator_preview(settings: Settings, symbol: str) -> pd.DataFrame:
    return _read_table(
        settings.indicators_db,
        """
        SELECT symbol, date, indicator_name, value
        FROM indicators
        WHERE symbol = ?
        ORDER BY date, indicator_name
        """,
        (symbol,),
    )


def load_alpha101_preview(settings: Settings, symbol: str) -> pd.DataFrame:
    return _read_table(
        settings.alphas_db,
        """
        SELECT symbol, date, alpha_id, value
        FROM alphas
        WHERE symbol = ?
        ORDER BY date, alpha_id
        """,
        (symbol,),
    )


def load_qlib_preview(settings: Settings, symbol: str) -> pd.DataFrame:
    table = qlib_table_name(symbol)
    if not _table_exists(settings.qlib_direct_factors_db, table):
        return pd.DataFrame()
    return _read_table(
        settings.qlib_direct_factors_db,
        f"SELECT * FROM {quote_identifier(table)} ORDER BY date",
    )


def load_finsaber_text_preview(
    settings: Settings,
    *,
    doc_type: str,
    limit: int,
) -> pd.DataFrame:
    db_path = settings.finsaber_dir / "news_text_index.sqlite"
    if not _table_exists(db_path, "investment_texts"):
        return pd.DataFrame()
    return _read_table(
        db_path,
        """
        SELECT
            symbol,
            published_at,
            doc_type,
            external_id,
            LENGTH(text) AS text_length,
            SUBSTR(text, 1, 1200) AS text_sample
        FROM investment_texts
        WHERE doc_type = ?
          AND symbol IS NOT NULL
          AND text IS NOT NULL
        ORDER BY published_at DESC, symbol
        LIMIT ?
        """,
        (doc_type, limit),
    )


def export_csv_previews(
    settings: Settings | None = None,
    *,
    symbol: str = "AAPL",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    settings = settings or Settings.from_env()
    symbol = canonical_symbol(symbol)
    output_dir = output_dir or settings.reports_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "prices": load_price_preview,
        "indicators": load_indicator_preview,
        "alpha101": load_alpha101_preview,
        "qlib": load_qlib_preview,
    }
    written: dict[str, Path] = {}
    for key, loader in loaders.items():
        frame = loader(settings, symbol)
        filename = PREVIEW_FILENAMES[key].format(symbol=symbol)
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written[key] = path
    return written


def export_finsaber_previews(
    settings: Settings | None = None,
    *,
    output_dir: Path | None = None,
    limit: int = 25,
) -> dict[str, Path]:
    settings = settings or Settings.from_env()
    output_dir = output_dir or settings.reports_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_types = {
        "finsaber_news": "news",
        "finsaber_10k": "10-K",
        "finsaber_10q": "10-Q",
    }
    written: dict[str, Path] = {}
    for key, doc_type in doc_types.items():
        frame = load_finsaber_text_preview(settings, doc_type=doc_type, limit=limit)
        path = output_dir / PREVIEW_FILENAMES[key]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written[key] = path
    return written


def export_command(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    output_dir = Path(args.output) if args.output else settings.reports_dir / "previews"
    written = export_csv_previews(settings, symbol=args.symbol, output_dir=output_dir)
    for key, path in written.items():
        rows = len(pd.read_csv(path)) if path.exists() and path.stat().st_size > 0 else 0
        print(f"{key}: {path} ({rows} rows)")
    return 0


def export_finsaber_command(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    output_dir = Path(args.output) if args.output else settings.reports_dir / "previews"
    written = export_finsaber_previews(
        settings, output_dir=output_dir, limit=args.limit
    )
    for key, path in written.items():
        rows = len(pd.read_csv(path)) if path.exists() and path.stat().st_size > 0 else 0
        print(f"{key}: {path} ({rows} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export human-readable CSV previews")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Export CSV previews for one symbol")
    export.add_argument("--symbol", default="AAPL")
    export.add_argument("--output")
    export.set_defaults(func=export_command)

    finsaber = subparsers.add_parser(
        "export-finsaber", help="Export FINSABER news/filing text samples"
    )
    finsaber.add_argument("--limit", type=int, default=25)
    finsaber.add_argument("--output")
    finsaber.set_defaults(func=export_finsaber_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
