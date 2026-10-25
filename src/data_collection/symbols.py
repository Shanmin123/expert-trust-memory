from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Constituent:
    symbol: str
    ib_symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None


def canonical_symbol(raw_symbol: str) -> str:
    """Normalize symbols for storage and reports.

    Some public holdings files use dots for class shares, while many data tools
    prefer hyphens. The canonical layer keeps storage stable and API-independent.
    """
    symbol = str(raw_symbol).strip().upper()
    symbol = re.sub(r"\s+", "", symbol)
    return symbol.replace(".", "-")


def ib_contract_symbol(raw_symbol: str) -> str:
    """Convert a canonical symbol into the IB contract symbol.

    IB commonly represents class-share separators as spaces, e.g. BRK B.
    """
    return canonical_symbol(raw_symbol).replace("-", " ")


def sqlite_table_name(prefix: str, raw_symbol: str) -> str:
    symbol = canonical_symbol(raw_symbol)
    safe_symbol = re.sub(r"[^A-Z0-9_]", "_", symbol)
    return f"{prefix}_{safe_symbol}"
