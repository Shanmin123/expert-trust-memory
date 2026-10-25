"""FINSABER parquet panel + survivorship-aware universe selection.

Loads the real FINSABER-2 price panel (delisted names included) and selects a tradable universe.
Historical point-in-time S&P 500 membership is not available locally, so we fall back to a
transparently fixed always-present subset (full coverage over the test window) and LABEL the run
survivorship-limited. A hard guard refuses to run on the current-membership snapshot CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional, Sequence

from trading_agent import _bootstrap  # noqa: F401  (puts FINSABER clone on sys.path)


def resolve_data_root(root: Optional[str] = None) -> Path:
    root = root or os.getenv("FINSABER_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "FINSABER_DATA_ROOT is not set and no root was passed. Point it at a parquet dataset "
            "root containing price_daily/ (e.g. .../sp500_2000_2025_parquet)."
        )
    p = Path(root)
    low = str(p).lower()
    if low.endswith(".csv") or "current_snapshot" in low or "universe_current" in low:
        raise ValueError(
            f"Refusing to run on a current-membership universe ({p}). Use the FINSABER parquet panel "
            "(survivorship-bias-free; includes delisted names)."
        )
    if not (p / "price_daily").exists():
        raise FileNotFoundError(f"{p} has no price_daily/ partition; not a FINSABER parquet root.")
    return p


def load_panel(root: Optional[str], start: Date, end: Date, tickers=None, modalities=("price",)):
    from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset

    return FinsaberParquetDataset(
        root=str(resolve_data_root(root)), start_date=start, end_date=end,
        tickers=tickers, modalities=tuple(modalities),
    )


@dataclass(frozen=True)
class UniverseSelection:
    tickers: list
    label: str
    n_candidates_considered: int
    coverage_floor: float


def select_fixed_universe(
    root: Optional[str],
    start: Date,
    end: Date,
    n: int = 25,
    min_coverage: float = 0.98,
    allowed_tickers: Sequence[str] | None = None,
) -> UniverseSelection:
    """Pick the n most-liquid names with >= min_coverage of trading days over [start, end].

    This is survivorship-limited by construction (only names present throughout the window) — the
    label says so. Deterministic: rank by median dollar volume, tie-break by symbol.
    """
    panel = load_panel(root, start, end, tickers=None, modalities=("price",))
    df = panel.get_price_dataframe(tickers="all", date_from=start, date_to=end, adjust=True)
    if df.empty:
        raise ValueError(f"No price data in window {start}..{end}.")
    total_days = df["date"].nunique()
    g = df.groupby("symbol")
    stats = g.agg(
        n_obs=("date", "nunique"),
        dollar_vol=("volume", "median"),
        last_close=("close", "last"),
    )
    # liquidity proxy = median(volume) * median(close)
    med_close = g["close"].median()
    stats["liquidity"] = stats["dollar_vol"] * med_close
    stats["coverage"] = stats["n_obs"] / total_days
    if allowed_tickers is not None:
        allowed = {str(ticker) for ticker in allowed_tickers}
        stats = stats[stats.index.astype(str).isin(allowed)]
    eligible = stats[stats["coverage"] >= min_coverage].copy()
    eligible = eligible.sort_values(["liquidity", "coverage"], ascending=[False, False])
    tickers = sorted(eligible.head(n).index.tolist())  # sorted for deterministic order
    constraint = ""
    if allowed_tickers is not None:
        constraint = f"; constrained to {len(set(str(t) for t in allowed_tickers))} externally-covered names"
    label = (
        f"survivorship-LIMITED fixed universe: {len(tickers)} most-liquid names with "
        f">={min_coverage:.0%} coverage over {start}..{end} (historical S&P 500 membership "
        f"unavailable; names absent for part of the window are excluded{constraint})."
    )
    return UniverseSelection(
        tickers=tickers, label=label,
        n_candidates_considered=int(len(stats)), coverage_floor=min_coverage,
    )
