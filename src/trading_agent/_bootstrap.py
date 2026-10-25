"""Single dependency seam onto the vendored FINSABER clone.

`finsaber` is NOT pip-installed in this environment (verified Phase 0); the cloned source
lives at ``external/FINSABER/`` and exposes two importable top-level packages:
``finsaber`` (a thin facade) and ``backtest`` (the real implementation). We import from
``backtest`` directly and reuse ONLY:

* the point-in-time data loaders (`FinsaberDataset`, `FinsaberParquetDataset`), and
* the pure execution *formulas* in ``backtest.toolkit.execution``.

We deliberately do NOT import ``FINSABERFrameworkHelper`` / ``BaseStrategyIso`` (which would
pull in matplotlib/tqdm/colorlog and impose a push-based, single-ticker-isolated loop). The
adapter implements its own shared-portfolio accounting — see plan decision DD1 / Approach A.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

def _possible_finsaber_roots(package_file: Path | None = None) -> list[Path]:
    """Return FINSABER source locations for source trees and installed packages."""
    package_file = Path(package_file or __file__).resolve()
    roots: list[Path] = []
    env_root = os.getenv("FINSABER_SOURCE_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    if len(package_file.parents) > 2:
        roots.append(package_file.parents[2] / "external" / "FINSABER")
    for base in (Path.cwd(), *Path.cwd().parents):
        roots.append(base / "external" / "FINSABER")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


FINSABER_ROOT = next(
    (root for root in _possible_finsaber_roots() if root.exists()),
    _possible_finsaber_roots()[0],
)


def ensure_finsaber_on_path() -> Path:
    """Idempotently prepend the FINSABER clone to ``sys.path``. Returns its root."""
    root = next((path for path in _possible_finsaber_roots() if path.exists()), FINSABER_ROOT)
    if not root.exists():  # pragma: no cover - defensive
        searched = ", ".join(str(path) for path in _possible_finsaber_roots())
        raise RuntimeError(
            "FINSABER clone not found. "
            "Set FINSABER_SOURCE_ROOT or run from a project containing external/FINSABER. "
            f"Searched: {searched}"
        )
    path_str = str(root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return root


ensure_finsaber_on_path()

# Re-export the verbatim FINSABER pieces the rest of the package is allowed to use.
from backtest.toolkit.execution import (  # noqa: E402
    adjusted_bar_price,
    apply_liquidity_cap,
    apply_slippage,
    calculate_commission,
    prior_volume_stats,
)
from backtest.data_util.finsaber_dataset import FinsaberDataset  # noqa: E402
from backtest.data_util.trading_data import TradingData  # noqa: E402


def load_parquet_dataset(*args, **kwargs):
    """Lazy accessor for ``FinsaberParquetDataset`` (imports pyarrow on demand)."""
    from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset

    return FinsaberParquetDataset(*args, **kwargs)


__all__ = [
    "FINSABER_ROOT",
    "ensure_finsaber_on_path",
    "adjusted_bar_price",
    "apply_liquidity_cap",
    "apply_slippage",
    "calculate_commission",
    "prior_volume_stats",
    "FinsaberDataset",
    "TradingData",
    "load_parquet_dataset",
]
