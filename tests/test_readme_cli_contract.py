"""The README documents the launcher it ships with.

The README was cut back on 2026-08-21 to the project itself: what the code is, how to
install it, where the data comes from, and how to run the three modes. Five tests here
asserted sections that came out with it, the wheel-build recipe, the two validation
scripts, the memory-handoff section, the paper-status block and the IB validation
sequence, and were removed with their subject. What remains is the part that has to stay
in step with the CLI: if a mode, a flag or an output name changes, the README says so.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_documents_the_three_modes_and_their_outputs():
    readme = _readme()

    for mode in ("backtest", "holdout", "paper"):
        assert f"python -m trading_agent {mode}" in readme

    # the flags that define the holdout protocol
    for flag in ("--validation-boundary", "--memory-output", "--temperature-grid",
                 "--reflect-every"):
        assert flag in readme, flag

    # what a run writes, and the report fields a reader is told to look for
    for name in ("reports/trading_agent_backtest.json", "reports/expert_weight_holdout",
                 "memory_snapshot", "split_protocol", "validation_arms",
                 "system_completion"):
        assert name in readme, name


def test_readme_documents_the_paper_mode_safety_flags():
    readme = _readme()

    for flag in ("--connection-smoke", "--dry-run", "--execute-orders",
                 "--require-memory", "--use-account-cash", "--max-order-value"):
        assert flag in readme, flag


def test_readme_documents_where_the_data_and_finsaber_come_from():
    readme = _readme()

    assert "TRADING_AGENT_DATA_ROOT" in readme
    assert "FINSABER_SOURCE_ROOT" in readme
    assert "external/FINSABER" in readme
