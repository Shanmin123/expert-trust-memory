from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_collection.config import Settings, ensure_project_dirs
from data_collection.qlib_provider import (
    QLIB_MARKET,
    compute_direct_qlib_features,
    export_qlib_provider,
)
from data_collection.storage import connect_sqlite, upsert_price_frame
from data_collection.symbols import canonical_symbol
from trading_agent.data.finsaber_panel import load_panel, resolve_data_root


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _unlink_sqlite_bundle(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists():
            target.unlink()


def _normalise_finsaber_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "average", "symbol"]
        )
    out = frame.copy()
    if "symbol" not in out.columns:
        raise ValueError("FINSABER price frame must include symbol")
    if "date" not in out.columns:
        raise ValueError("FINSABER price frame must include date")
    for column in ("open", "high", "low", "close", "volume"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["average"] = out[["open", "high", "low", "close"]].mean(axis=1)
    out["barCount"] = None
    out["symbol"] = out["symbol"].map(canonical_symbol)
    return out


def export_finsaber_prices_to_sqlite(
    *,
    data_root: str | Path,
    start: str | date,
    end: str | date,
    output_db: str | Path,
    symbols: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Materialise FINSABER parquet prices into the SQLite format used by qlib export."""
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    selected = [canonical_symbol(symbol) for symbol in symbols] if symbols else None
    output = Path(output_db)
    if force:
        _unlink_sqlite_bundle(output)

    root = resolve_data_root(str(data_root))
    panel = load_panel(
        str(root),
        start_date,
        end_date,
        tickers=selected,
        modalities=("price",),
    )
    raw = panel.get_price_dataframe(
        tickers=selected or "all",
        date_from=start_date,
        date_to=end_date,
        adjust=True,
    )
    prices = _normalise_finsaber_prices(raw)
    if prices.empty:
        raise ValueError(f"No FINSABER prices found for {start_date}..{end_date}")

    conn = connect_sqlite(output)
    total_rows = 0
    try:
        for symbol, group in prices.groupby("symbol", sort=True):
            total_rows += upsert_price_frame(conn, symbol, group)
    finally:
        conn.close()

    return {
        "data_root": str(root),
        "prices_db": str(output),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "symbols": sorted(prices["symbol"].dropna().unique().tolist()),
        "price_symbols": int(prices["symbol"].nunique()),
        "price_rows": int(total_rows),
    }


def build_finsaber_qlib_pipeline(
    *,
    data_root: str | Path,
    start: str | date,
    end: str | date,
    prices_db: str | Path,
    provider_dir: str | Path,
    factors_db: str | Path,
    symbols: Sequence[str] | None = None,
    market: str = QLIB_MARKET,
    force: bool = False,
    build_factors: bool = True,
) -> dict[str, object]:
    selected = [canonical_symbol(symbol) for symbol in symbols] if symbols else None
    price_manifest = export_finsaber_prices_to_sqlite(
        data_root=data_root,
        start=start,
        end=end,
        output_db=prices_db,
        symbols=selected,
        force=force,
    )
    instruments, calendar_rows = export_qlib_provider(
        Path(prices_db),
        Path(provider_dir),
        market=market,
        symbols=selected,
        force=force,
    )
    factor_symbols = 0
    factor_rows = 0
    if build_factors:
        factor_symbols, factor_rows = compute_direct_qlib_features(
            Path(provider_dir),
            Path(factors_db),
            market=market,
            symbols=selected,
            force=force,
        )

    return {
        "pipeline": "finsaber_to_qlib_factors",
        "market": market,
        "start": price_manifest["start"],
        "end": price_manifest["end"],
        "symbols": price_manifest["symbols"],
        "price_symbols": price_manifest["price_symbols"],
        "price_rows": price_manifest["price_rows"],
        "qlib_provider": {
            "path": str(provider_dir),
            "instruments": instruments,
            "calendar_rows": calendar_rows,
        },
        "qlib_factors": {
            "path": str(factors_db),
            "symbols": factor_symbols,
            "rows": factor_rows,
            "built": bool(build_factors),
        },
        "artifacts": {
            "prices_db": str(prices_db),
            "qlib_provider_uri": str(provider_dir),
            "alpha158_factors_db": str(factors_db),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Build the project qlib factor pipeline from FINSABER parquet data."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--market", default=QLIB_MARKET)
    parser.add_argument("--prices-db", default=str(settings.prices_db))
    parser.add_argument("--provider-dir", default=str(settings.qlib_provider_dir))
    parser.add_argument("--factors-db", default=str(settings.qlib_direct_factors_db))
    parser.add_argument("--manifest", default=str(settings.data_dir / "qlib_pipeline_manifest.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-factors",
        action="store_true",
        help="Export prices/provider only; useful outside the qlib conda environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    ensure_project_dirs(settings)
    args = build_parser().parse_args(argv)
    manifest = build_finsaber_qlib_pipeline(
        data_root=args.data_root,
        start=args.start,
        end=args.end,
        symbols=args.symbols,
        prices_db=args.prices_db,
        provider_dir=args.provider_dir,
        factors_db=args.factors_db,
        market=args.market,
        force=args.force,
        build_factors=not args.skip_factors,
    )
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote qlib pipeline manifest: {out}")
    print(f"Price rows: {manifest['price_rows']}")
    print(f"Provider instruments: {manifest['qlib_provider']['instruments']}")
    print(f"Factor rows: {manifest['qlib_factors']['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
