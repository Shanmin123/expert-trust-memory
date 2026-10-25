"""Tests 8, 9, 10: idempotent persistence, deterministic replay, end-to-end smoke run."""

from __future__ import annotations

from trading_agent.baselines import equal_weight_engines
from trading_agent.persistence import RunLogger
from trading_agent.runner import run_backtest


# --- Test 10: end-to-end integration run on an in-memory fixture ---------------------
def test_end_to_end_equity_curve(demo_dataset, demo_config):
    alpha, risk, portfolio = equal_weight_engines()
    result = run_backtest(demo_dataset, demo_config, alpha, risk, portfolio, run_id="smoke")

    assert len(result.equity_curve) == len(demo_dataset.get_date_range())
    assert result.summary["n_fills"] >= 1.0           # at least one baseline fill
    assert result.final_account.equity > 0.0
    # equity curve is well-formed (date, float) pairs in chronological order
    dates = [d for d, _ in result.equity_curve]
    assert dates == sorted(dates)


# --- Test 9: deterministic replay ----------------------------------------------------
def test_deterministic_replay(demo_dataset, demo_config):
    alpha, risk, portfolio = equal_weight_engines()
    r1 = run_backtest(demo_dataset, demo_config, alpha, risk, portfolio, run_id="rep")
    # fresh engines + fresh dataset object to prove statelessness
    from datetime import date
    from trading_agent._bootstrap import FinsaberDataset

    same = FinsaberDataset(data={d: dict(demo_dataset.get_data_by_date(d))
                                 for d in demo_dataset.get_date_range()})
    a2, k2, p2 = equal_weight_engines()
    r2 = run_backtest(same, demo_config, a2, k2, p2, run_id="rep")

    assert r1.equity_curve == r2.equity_curve
    assert r1.fills == r2.fills
    assert r1.rejected == r2.rejected
    assert r1.summary == r2.summary
    assert r1.final_account == r2.final_account


# --- Test 8: idempotent / duplicate writes ------------------------------------------
def test_run_logger_idempotent(tmp_path):
    logger = RunLogger(tmp_path / "run.jsonl")
    assert logger.log("decision", "run1", "d1", {"x": 1}) is True
    # same (run_id, decision_id, kind) -> skipped, no duplicate line
    assert logger.log("decision", "run1", "d1", {"x": 2}) is False
    assert len(logger.records()) == 1

    # a new logger over the same file recovers the seen-set and stays idempotent
    reopened = RunLogger(tmp_path / "run.jsonl")
    assert reopened.log("decision", "run1", "d1", {"x": 3}) is False
    assert len(reopened.records()) == 1


def test_runner_writes_are_idempotent_on_replay(demo_dataset, demo_config, tmp_path):
    logger = RunLogger(tmp_path / "run.jsonl")
    alpha, risk, portfolio = equal_weight_engines()
    run_backtest(demo_dataset, demo_config, alpha, risk, portfolio, run_id="once", logger=logger)
    count_after_first = len(logger.records())
    assert count_after_first >= 2  # decisions + summary

    # Re-run with the same run_id + same logger file: no new lines (all keys already seen).
    reopened = RunLogger(tmp_path / "run.jsonl")
    a2, k2, p2 = equal_weight_engines()
    run_backtest(demo_dataset, demo_config, a2, k2, p2, run_id="once", logger=reopened)
    assert len(reopened.records()) == count_after_first
