"""Shared report schema contract tests."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_schema_version_is_single_source_of_truth():
    from trading_agent.reporting import REPORT_SCHEMA_VERSION, SYSTEM_IDENTITY

    assert REPORT_SCHEMA_VERSION == "trading-agent-report/v1"
    assert SYSTEM_IDENTITY == "four_expert_reflection"

    production_files = [
        ROOT / "src" / "trading_agent" / "cli.py",
        ROOT / "src" / "trading_agent" / "paper_trading.py",
        ROOT / "scripts" / "run_expert_weight_holdout.py",
    ]
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        assert '"trading-agent-report/v1"' not in text
        assert '"four_expert_reflection"' not in text
