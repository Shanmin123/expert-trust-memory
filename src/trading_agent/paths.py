"""Where the data lives, resolved once instead of hardcoded in seventeen scripts.

Every large input sits outside the repository and is reached through ``data/``, which on the
development machine holds directory junctions to a separate drive. A clone will not have
those junctions, so each path is overridable by an environment variable and every accessor
fails loudly when the target is missing.

Failing loudly is the point. A missing parquet directory does not raise on its own -- pyarrow
returns an empty dataset and the run produces plausible, wrong numbers. That failure mode has
cost this project real time, so the resolvers here check existence and say what to set.

    TRADING_AGENT_DATA_ROOT       overrides the whole tree (default: <repo>/data)
    FINSABER_PANEL      full FINSABER parquet panel
    FINSABER_SLIM       slim panel, the evaluated names only
    FINMEM_PYTHON       interpreter for the vendored FinMem environment
    TRADINGAGENTS_PYTHON interpreter for the vendored TradingAgents environment

    from trading_agent.paths import finsaber_panel, slim_panel
    panel = finsaber_panel()          # raises with instructions if absent
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """The tree holding every large input. Not required to exist yet."""
    return Path(os.environ.get("TRADING_AGENT_DATA_ROOT") or (REPO_ROOT / "data"))


def _resolve(env_var: str, *relative: str, what: str) -> Path:
    """An overridable path that must exist, with an actionable error when it does not."""
    override = os.environ.get(env_var)
    path = Path(override) if override else data_root().joinpath(*relative)
    if not path.exists():
        source = f"{env_var}={override}" if override else f"{path} (from TRADING_AGENT_DATA_ROOT)"
        raise FileNotFoundError(
            f"{what} not found at {source}.\n"
            f"  Set {env_var} to its location, or point TRADING_AGENT_DATA_ROOT at a tree that "
            f"contains {Path(*relative)}.\n"
            f"  This is raised rather than returned empty on purpose: an absent parquet "
            f"directory reads as zero rows and produces numbers that look fine."
        )
    return path


def finsaber_panel() -> Path:
    """Full FINSABER parquet panel: price_daily, news_items, filingk, filingq."""
    return _resolve("FINSABER_PANEL", "finsaber", "sp500_2000_2025_parquet",
                    what="the FINSABER panel")


def slim_panel() -> Path:
    """Slim panel carrying only the evaluated names. Faster where it suffices."""
    return _resolve("FINSABER_SLIM", "finsaber_slim", "sp500_2000_2025_parquet",
                    what="the slim FINSABER panel")


def finsaber_prices() -> Path:
    """price_daily subdirectory of the full panel."""
    return finsaber_panel() / "price_daily"


def slim_prices() -> Path:
    return slim_panel() / "price_daily"


def corpus_dir() -> Path:
    """Recorded expert-view corpora, when kept outside the repository."""
    return _resolve("MSC_CORPUS", "corpus", what="the corpus directory")


def finmem_python() -> Path:
    """Interpreter for the vendored FinMem environment, which needs its own pins."""
    return _resolve("FINMEM_PYTHON", "finmem_venv", "Scripts", "python.exe",
                    what="the FinMem interpreter")


def tradingagents_python() -> Path:
    return _resolve("TRADINGAGENTS_PYTHON", "ta_venv", "Scripts", "python.exe",
                    what="the TradingAgents interpreter")


def describe() -> str:
    """One block showing what resolves and what does not, for a setup check."""
    lines = [f"repo root      {REPO_ROOT}", f"data root      {data_root()}"]
    for label, fn in (("FINSABER panel", finsaber_panel), ("slim panel", slim_panel),
                      ("corpus", corpus_dir), ("FinMem python", finmem_python),
                      ("TradingAgents python", tradingagents_python)):
        try:
            lines.append(f"{label:<14} {fn()}")
        except FileNotFoundError:
            lines.append(f"{label:<14} MISSING")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
