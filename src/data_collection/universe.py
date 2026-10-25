from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from data_collection.config import Settings, ensure_project_dirs
from data_collection.symbols import Constituent, canonical_symbol, ib_contract_symbol


SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/institutional/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
SOURCE_NAME = "State Street SPDR SPY daily holdings"
SOURCE_NOTE = (
    "Current S&P 500 universe proxy from State Street SPDR SPY holdings; "
    "IBKR is used for contract validation and historical OHLCV collection."
)
REQUEST_HEADERS = {
    "User-Agent": (
        "msc-dissertation-data-collector/0.1 "
        "(educational research; current S&P 500 universe snapshot)"
    )
}
EQUITY_TICKER_RE = re.compile(r"^[A-Z][A-Z.]*$")


def parse_spy_as_of_date(workbook_bytes: bytes) -> str:
    metadata = pd.read_excel(
        BytesIO(workbook_bytes), sheet_name="holdings", header=None, nrows=5
    )
    for value in metadata.stack().dropna():
        text = str(value).strip()
        if text.lower().startswith("as of "):
            parsed = pd.to_datetime(text[6:], errors="raise")
            return parsed.date().isoformat()
    raise ValueError("Could not find the SPY holdings as-of date.")


def normalize_spy_holdings_frame(
    holdings: pd.DataFrame,
    *,
    source_as_of: str,
    source_url: str = SPY_HOLDINGS_URL,
    captured_at: str | None = None,
) -> pd.DataFrame:
    required = {"Name", "Ticker", "Identifier", "SEDOL", "Weight", "Shares Held"}
    missing = required - set(holdings.columns)
    if missing:
        raise ValueError(f"SPY holdings file missing columns: {sorted(missing)}")

    captured_at = captured_at or datetime.now().isoformat(timespec="seconds")
    df = holdings.copy()
    df["raw_ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["shares_held"] = pd.to_numeric(df["Shares Held"], errors="coerce")

    df = df[df["weight"].notna()]
    df = df[df["raw_ticker"].map(lambda ticker: bool(EQUITY_TICKER_RE.fullmatch(ticker)))]
    df = df.drop_duplicates(subset=["raw_ticker"], keep="first")

    out = pd.DataFrame(
        {
            "symbol": df["raw_ticker"].map(canonical_symbol),
            "ib_symbol": df["raw_ticker"].map(ib_contract_symbol),
            "name": df["Name"].astype(str).str.strip(),
            "identifier": df["Identifier"].astype(str).str.strip(),
            "sedol": df["SEDOL"].astype(str).str.strip(),
            "weight": df["weight"],
            "shares_held": df["shares_held"],
            "local_currency": df.get("Local Currency", "USD"),
            "source": SOURCE_NAME,
            "source_url": source_url,
            "source_as_of": source_as_of,
            "captured_at": captured_at,
            "source_note": SOURCE_NOTE,
        }
    )
    return out.reset_index(drop=True)


def parse_spy_holdings_workbook(
    workbook_bytes: bytes, *, source_url: str = SPY_HOLDINGS_URL
) -> pd.DataFrame:
    source_as_of = parse_spy_as_of_date(workbook_bytes)
    holdings = pd.read_excel(BytesIO(workbook_bytes), sheet_name="holdings", header=4)
    return normalize_spy_holdings_frame(
        holdings, source_as_of=source_as_of, source_url=source_url
    )


def fetch_spy_holdings_workbook(url: str = SPY_HOLDINGS_URL) -> tuple[bytes, str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError("SPY holdings response is not an XLSX workbook.")
    return response.content, response.url


def snapshot_path(settings: Settings, source_as_of: str) -> Path:
    return settings.universe_snapshots_dir / f"sp500_{source_as_of}.csv"


def write_snapshot_files(df: pd.DataFrame, settings: Settings) -> tuple[Path, Path]:
    ensure_project_dirs(settings)
    source_as_of = str(df["source_as_of"].iloc[0])
    dated_path = snapshot_path(settings, source_as_of)
    current_path = settings.sp500_universe_snapshot
    df.to_csv(dated_path, index=False, encoding="utf-8")
    shutil.copyfile(dated_path, current_path)
    return dated_path, current_path


def display_path(path: Path, settings: Settings) -> str:
    try:
        return str(path.relative_to(settings.data_dir.parent)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_universe_summary(
    df: pd.DataFrame, dated_path: Path, current_path: Path, settings: Settings
) -> Path:
    output = settings.reports_dir / "universe_snapshot_summary.md"
    ensure_project_dirs(settings)
    source_as_of = str(df["source_as_of"].iloc[0])
    output.write_text(
        "\n".join(
            [
                "# S&P 500 Universe Snapshot",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                f"- Source: {SOURCE_NAME}",
                f"- Source as-of date: {source_as_of}",
                f"- Rows after equity-ticker filtering: {len(df)}",
                f"- Dated snapshot: `{display_path(dated_path, settings)}`",
                f"- Current snapshot: `{display_path(current_path, settings)}`",
                f"- Source URL: {df['source_url'].iloc[0]}",
                "",
                "## Method",
                "",
                "- Download the official SPY daily holdings workbook from State Street.",
                "- Keep ordinary equity tickers and remove cash/contra rows.",
                "- Normalize class-share tickers for storage, e.g. `BRK.B` -> `BRK-B`.",
                "- Convert the normalized ticker separately for IBKR contracts, e.g. `BRK-B` -> `BRK B`.",
                "- Write both dated and current local CSV snapshots.",
                "",
                "## Scope Note",
                "",
                (
                    "This is a current-universe snapshot for IBKR collection, not a "
                    "survivorship-bias-free historical constituent database. Historical "
                    "backtesting should continue to rely on FINSABER for broader "
                    "historical constituent coverage and text/price history."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def refresh_sp500_snapshot(settings: Settings | None = None) -> tuple[pd.DataFrame, Path, Path]:
    settings = settings or Settings.from_env()
    workbook, resolved_url = fetch_spy_holdings_workbook()
    df = parse_spy_holdings_workbook(workbook, source_url=resolved_url)
    dated_path, current_path = write_snapshot_files(df, settings)
    write_universe_summary(df, dated_path, current_path, settings)
    return df, dated_path, current_path


def load_universe_snapshot(path: Path) -> list[Constituent]:
    df = pd.read_csv(path)
    columns = {col.lower(): col for col in df.columns}
    if "symbol" not in columns:
        raise ValueError("Universe snapshot must include a symbol column.")

    symbol_col = columns["symbol"]
    name_col = columns.get("name")
    out: list[Constituent] = []
    for _, row in df.iterrows():
        symbol = canonical_symbol(row[symbol_col])
        name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else None
        out.append(
            Constituent(
                symbol=symbol,
                ib_symbol=ib_contract_symbol(symbol),
                name=name,
            )
        )
    return out


def refresh_command(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    df, dated_path, current_path = refresh_sp500_snapshot(settings)
    print(f"Fetched {len(df)} SPY holding rows for S&P 500 universe.")
    print(f"Source as-of: {df['source_as_of'].iloc[0]}")
    print(f"Dated snapshot: {dated_path}")
    print(f"Current snapshot: {current_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S&P 500 universe snapshot tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser(
        "refresh-sp500",
        help="Refresh the current S&P 500 universe snapshot from SPY holdings.",
    )
    refresh.add_argument(
        "--source",
        choices=["spy-holdings"],
        default="spy-holdings",
        help="Universe source. Only the SPY holdings source is currently implemented.",
    )
    refresh.set_defaults(func=refresh_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
