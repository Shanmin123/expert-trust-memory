"""Paper market_state is conditioned on an external proxy (SPY), never the traded name.

The frozen system forbids a single-name run from conditioning on its own trend/vol
(market_state_classifier.py docstring). These tests pin that the paper bridge honours it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trading_agent import cli


def _frame(closes: list[float], start: str = "2024-01-02"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "date": [d.date().isoformat() for d in dates],
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "adjusted_close": closes,
        "volume": [1_000_000] * len(closes),
    })


def _paper_args() -> SimpleNamespace:
    # Small classifier windows so the trend/vol resolve on a modest test frame.
    return SimpleNamespace(
        paper_market_state="auto",
        paper_vol_window=3,
        paper_percentile_window=5,
        paper_trend_window=10,
        paper_trend_confirm_days=2,
        market_proxy_symbol="SPY",
    )


def test_market_state_trend_follows_the_proxy_not_the_stock():
    n = 30
    rising = [100.0 + i for i in range(n)]    # SPY trends up
    falling = [200.0 - i for i in range(n)]   # traded stock trends down
    stock_frame = _frame(falling)
    spy_frame = _frame(rising)
    args = _paper_args()
    observation = cli._paper_observation(["STK"], {"STK": stock_frame})

    state, source = cli._resolve_paper_market_state(
        args, observation, {"STK": stock_frame}, proxy_frame=spy_frame
    )
    assert source == "classifier_proxy"
    assert state.startswith("up"), state  # from SPY, not the falling stock


def test_without_proxy_the_label_refuses_to_read_the_stock_itself():
    """A failed proxy fetch must not silently condition a stock on its own trend.

    The path that did exactly that was removed on 2026-08-02. The backtest already
    refused to run a single name without an external proxy; the live path was the
    one place the same mistake could still happen quietly. A steeply falling stock
    must NOT produce a down label from its own prices.
    """
    n = 30
    falling = [200.0 - i for i in range(n)]
    stock_frame = _frame(falling)
    args = _paper_args()
    observation = cli._paper_observation(["STK"], {"STK": stock_frame})

    state, source = cli._resolve_paper_market_state(
        args, observation, {"STK": stock_frame}, proxy_frame=None
    )
    assert source == "proxy_unavailable_no_self_conditioning"
    assert not state.startswith("down"), state


def test_manual_state_ignores_the_proxy():
    args = _paper_args()
    args.paper_market_state = "up_stress"
    stock_frame = _frame([100.0] * 30)
    observation = cli._paper_observation(["STK"], {"STK": stock_frame})
    state, source = cli._resolve_paper_market_state(
        args, observation, {"STK": stock_frame}, proxy_frame=_frame([100.0] * 30)
    )
    assert state == "up_stress"
    assert source == "manual_cli"
