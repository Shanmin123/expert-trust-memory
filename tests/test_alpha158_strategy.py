from __future__ import annotations

import sqlite3
from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent.alpha.alpha158_recipes import Alpha158FeatureSpec
from trading_agent.alpha.alpha158_strategy import (
    Alpha158StrategyAlpha,
    FrameAlpha158FeatureProvider,
    SQLiteQlibFactorProvider,
)
from trading_agent.contracts import Observation
from data_collection.qlib_provider import qlib_table_name


def _obs(tickers=("AAA", "BBB", "CCC")):
    d = date(2024, 1, 3)
    return Observation(
        decision_date=d,
        decision_time=datetime.combine(d, time.min),
        availability_time=datetime.combine(d, time.min),
        bars={},
        tickers=tuple(tickers),
    )


def _history_frame(values_by_ticker: dict[str, list[float]], end="2024-01-03"):
    """One row per (date, ticker) ending at ``end`` (the decision date)."""
    rows = []
    n = max(len(v) for v in values_by_ticker.values())
    dates = pd.bdate_range(end=end, periods=n).date
    for ticker, values in values_by_ticker.items():
        for day, value in zip(dates[-len(values):], values):
            rows.append({"date": day, "ticker": ticker, "alpha158_roc20": value})
    return pd.DataFrame(rows)


def test_features_are_ts_normalised_against_own_history():
    # Alpha158 ratios hover around 1.0; the signal is the TIME-SERIES z-score
    # against the ticker's own prior history, so a value at its own norm reads
    # ~0 and a spike reads positive -- even though every raw value is positive.
    spec = Alpha158FeatureSpec("alpha158_roc20", "ROC20", "momentum")
    steady = [1.0 + 0.02 * ((i % 5) - 2) for i in range(39)] + [1.0]  # oscillates ~1.0, ends at norm
    spiked = steady[:-1] + [1.2]                                      # ends on a spike
    frame = _history_frame({"AAA": spiked, "BBB": steady})
    alpha = Alpha158StrategyAlpha(
        FrameAlpha158FeatureProvider(frame), [spec], history_window=39, min_history=10,
    )
    by_ticker = {s.ticker: s.value for s in alpha.score(_obs(tickers=("AAA", "BBB")))}
    assert by_ticker["AAA"] > 2.0            # spike far above own history
    assert abs(by_ticker["BBB"]) < 1.0       # at its own norm despite raw value ~1.0


def test_single_name_ts_score_is_nonzero():
    # A cross-sectional z-score would degenerate to 0 for a one-ticker universe.
    spec = Alpha158FeatureSpec("alpha158_roc20", "ROC20", "momentum")
    values = [1.0 + 0.02 * ((i % 5) - 2) for i in range(39)] + [1.15]
    frame = _history_frame({"AAA": values})
    alpha = Alpha158StrategyAlpha(
        FrameAlpha158FeatureProvider(frame), [spec], history_window=39, min_history=10,
    )
    signals = alpha.score(_obs(tickers=("AAA",)))
    assert len(signals) == 1
    assert signals[0].value > 1.0


def test_provider_without_history_falls_back_to_raw_values():
    spec = Alpha158FeatureSpec("alpha158_roc20", "ROC20", "momentum")

    class CurrentOnlyProvider:
        def values(self, decision_date, tickers, features):
            return {"alpha158_roc20": {"AAA": 0.7}}

    alpha = Alpha158StrategyAlpha(CurrentOnlyProvider(), [spec])
    signals = alpha.score(_obs(tickers=("AAA",)))
    assert len(signals) == 1
    assert signals[0].value == 0.7


def test_insufficient_history_yields_no_factor_signal_not_a_raw_leak():
    spec = Alpha158FeatureSpec("alpha158_roc20", "ROC20", "momentum")
    frame = _history_frame({"AAA": [1.0, 1.01, 1.02]})
    alpha = Alpha158StrategyAlpha(
        FrameAlpha158FeatureProvider(frame), [spec], history_window=252, min_history=60,
    )
    assert list(alpha.score(_obs(tickers=("AAA",)))) == []


def test_frame_provider_uses_decision_date_only_so_future_rows_do_not_change_score():
    spec = Alpha158FeatureSpec("alpha158_ma20", "MA20", "trend")
    base = pd.DataFrame(
        [
            {"date": "2024-01-03", "ticker": "AAA", "alpha158_ma20": 1.0},
            {"date": "2024-01-03", "ticker": "BBB", "alpha158_ma20": 2.0},
        ]
    )
    future = pd.concat(
        [
            base,
            pd.DataFrame([
                {"date": "2024-01-04", "ticker": "AAA", "alpha158_ma20": 100.0},
                {"date": "2024-01-04", "ticker": "BBB", "alpha158_ma20": -100.0},
            ]),
        ],
        ignore_index=True,
    )
    got1 = Alpha158StrategyAlpha(FrameAlpha158FeatureProvider(base), [spec]).score(_obs(("AAA", "BBB")))
    got2 = Alpha158StrategyAlpha(FrameAlpha158FeatureProvider(future), [spec]).score(_obs(("AAA", "BBB")))
    assert {s.ticker: s.value for s in got1} == pytest.approx({s.ticker: s.value for s in got2})


def test_sqlite_qlib_factor_provider_reads_not_recomputes(tmp_path):
    db = tmp_path / "qlib.db"
    spec = Alpha158FeatureSpec("alpha158_roc5", "ROC5", "momentum")
    with sqlite3.connect(db) as conn:
        table = qlib_table_name("AAA")
        conn.execute(f'CREATE TABLE "{table}" (symbol TEXT, date TEXT, QLIB158_ROC5 REAL)')
        conn.execute(f'INSERT INTO "{table}" VALUES (?,?,?)', ("AAA", "2024-01-03", 1.23))
    provider = SQLiteQlibFactorProvider(db)
    values = provider.values(date(2024, 1, 3), ["AAA"], [spec])
    assert values["alpha158_roc5"]["AAA"] == pytest.approx(1.23)
