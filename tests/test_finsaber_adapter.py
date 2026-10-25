"""Phase 2 adapter tests: next-bar execution, reconciliation, formula fidelity, rejections."""

from __future__ import annotations

from datetime import date

import pytest

from trading_agent._bootstrap import apply_slippage, calculate_commission, prior_volume_stats
from trading_agent.contracts import BacktestConfig, Order
from trading_agent.execution.finsaber_adapter import FinsaberBacktestAdapter


def _order(ticker, side, qty, signal_date, ref):
    return Order(
        order_id=f"{ticker}-{side}-{qty}",
        decision_id="d0",
        signal_date=signal_date,
        ticker=ticker,
        side=side,
        quantity=qty,
        reference_price=ref,
    )


def _expected_buy(adapter, config, ticker, qty, fill_date):
    """Independently recompute fill via FINSABER formulas (proves we reuse them)."""
    bar = adapter._window.get_data_by_date(fill_date)["price"][ticker]
    from trading_agent._bootstrap import adjusted_bar_price

    price = adjusted_bar_price(bar, "adjusted_open")
    avg_vol = prior_volume_stats(adapter._window, ticker, fill_date,
                                 config.liquidity_lookback_days)["average_volume"]
    fill_price, slippage = apply_slippage(price, "buy", qty, average_volume=avg_vol,
                                          slippage_perc=config.slippage_perc,
                                          slippage_impact=config.slippage_impact)
    commission = calculate_commission(qty, fill_price,
                                      commission_per_share=config.commission_per_share,
                                      min_commission=config.min_commission,
                                      max_commission_rate=config.max_commission_rate)
    return price, fill_price, slippage, commission


def _buy_filled_next_bar(adapter, ticker="AAA", qty=10):
    dates = adapter.dates
    adapter.open_bar(dates[0])
    obs0 = adapter.get_observation(dates[0])
    adapter.submit_order(_order(ticker, "buy", qty, dates[0], obs0.bars[ticker].adjusted_close))
    fills_before = adapter.fills
    adapter.advance()
    adapter.open_bar(dates[1])
    return dates, fills_before


# --- Test 1: no same-bar or future-bar execution -------------------------------------
def test_next_bar_execution_only(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates, fills_before = _buy_filled_next_bar(adapter)
    assert fills_before == []  # nothing fills on the signal bar
    fills = adapter.fills
    assert len(fills) == 1
    assert fills[0].execution_date == dates[1]
    # No fill ever executes on or before its own signal bar.
    assert all(f.execution_date > f.signal_date for f in fills)


# --- Test 3 & 4: cash and position reconciliation ------------------------------------
def test_cash_and_position_reconciliation(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    _, _ = _buy_filled_next_bar(adapter, "AAA", 10)
    _, fill_price, _, commission = _expected_buy(adapter, demo_config, "AAA", 10, dates[1])

    account = adapter.get_account()
    assert account.cash == pytest.approx(demo_config.initial_cash - (10 * fill_price + commission))

    positions = adapter.get_positions().by_ticker()
    assert positions["AAA"].quantity == 10
    assert positions["AAA"].avg_price == pytest.approx(fill_price)
    # equity = cash + mark-to-market (marked at the current date's adjusted_close)
    assert account.equity == pytest.approx(account.cash + account.positions_value)


# --- Test 5: commission & slippage equal the execution.py formula outputs ------------
def test_commission_and_slippage_match_formulas(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    _buy_filled_next_bar(adapter, "AAA", 10)
    ref_price, fill_price, slippage, commission = _expected_buy(adapter, demo_config, "AAA", 10, dates[1])

    fill = adapter.fills[0]
    assert fill.reference_price == pytest.approx(ref_price)
    assert fill.fill_price == pytest.approx(fill_price)
    assert fill.slippage_cost == pytest.approx(slippage)
    assert fill.commission == pytest.approx(commission)


def test_bps_cost_model_charges_traded_notional_when_configured(demo_dataset, demo_config):
    config = BacktestConfig(
        date_from=demo_config.date_from,
        date_to=demo_config.date_to,
        initial_cash=demo_config.initial_cash,
        tickers=demo_config.tickers,
        execution_cost_bps_per_side=10.0,
        commission_per_share=99.0,
        min_commission=99.0,
        max_commission_rate=99.0,
        slippage_perc=0.0,
        liquidity_cap_pct=0.0,
        seed=demo_config.seed,
    )
    adapter = FinsaberBacktestAdapter(demo_dataset, config)
    dates = adapter.dates
    _buy_filled_next_bar(adapter, "AAA", 10)

    fill = adapter.fills[0]
    assert fill.commission == pytest.approx(fill.quantity * fill.fill_price * 0.001)
    assert fill.commission < 99.0


def test_preload_window_supports_indicators_before_active_start(regime_dataset, mem_backtest_config):
    start = mem_backtest_config.date_from
    # 20 trading days into the DECISION window (the dataset carries a warmup
    # prefix before date_from, so index into the window, not the raw range).
    window = [
        (d.date() if hasattr(d, "date") else d)
        for d in regime_dataset.get_date_range()
    ]
    active_start = [d for d in window if d >= start][20]
    config = BacktestConfig(
        date_from=active_start,
        date_to=mem_backtest_config.date_to,
        preload_date_from=start,
        initial_cash=mem_backtest_config.initial_cash,
        tickers=mem_backtest_config.tickers,
    )

    adapter = FinsaberBacktestAdapter(regime_dataset, config)

    assert adapter.dates[0] == active_start
    raw_window_start = adapter._window.get_date_range()[0]
    window_start = raw_window_start.date() if hasattr(raw_window_start, "date") else raw_window_start
    assert window_start == start


def test_adding_to_position_keeps_quantity_weighted_average_price(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    _buy_filled_next_bar(adapter, "AAA", 10)  # first buy fills at dates[1] open
    obs1 = adapter.get_observation(dates[1])
    adapter.submit_order(_order("AAA", "buy", 30, dates[1], obs1.bars["AAA"].adjusted_close))
    adapter.advance()
    adapter.open_bar(dates[2])  # second buy fills at dates[2] open

    fills = adapter.fills
    assert len(fills) == 2
    expected_avg = (
        fills[0].quantity * fills[0].fill_price + fills[1].quantity * fills[1].fill_price
    ) / (fills[0].quantity + fills[1].quantity)
    position = adapter.get_positions().by_ticker()["AAA"]
    assert position.quantity == 40
    assert position.avg_price == pytest.approx(expected_avg)


def test_finalize_refreshes_last_equity_point_with_liquidation_costs(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    adapter.open_bar(dates[0])
    obs0 = adapter.get_observation(dates[0])
    adapter.submit_order(_order("AAA", "buy", 10, dates[0], obs0.bars["AAA"].adjusted_close))
    for d in dates[1:]:
        adapter.advance()
        adapter.open_bar(d)
    adapter.finalize()

    result = adapter.result("equity-refresh")
    # The last equity point reflects the forced liquidation, so curve-based
    # total return matches summary/final_account total return.
    last_date, last_equity = result.equity_curve[-1]
    assert last_date == dates[-1]
    assert last_equity == pytest.approx(result.final_account.equity)
    curve_return = result.equity_curve[-1][1] / demo_config.initial_cash - 1.0
    assert curve_return == pytest.approx(result.summary["total_return"])


# --- Test 6: rejected invalid orders (FINSABER reason codes) -------------------------
def test_rejected_sell_without_holdings(demo_dataset, demo_config):
    # FINSABER clamps a sell to min(requested, holding); with 0 held the resolved
    # quantity is 0, so the engine rejects it as "zero_quantity" (verified semantics —
    # the insufficient_holdings branch is unreachable after the clamp).
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    adapter.open_bar(dates[0])
    adapter.submit_order(_order("AAA", "sell", 5, dates[0], 100.0))  # nothing held
    adapter.advance()
    adapter.open_bar(dates[1])
    reasons = [r.reason for r in adapter.rejected_orders]
    assert reasons == ["zero_quantity"]


def test_rejected_invalid_price_unknown_ticker(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    dates = adapter.dates
    adapter.open_bar(dates[0])
    adapter.submit_order(_order("ZZZ", "buy", 5, dates[0], None))  # unknown + no ref price
    adapter.advance()
    adapter.open_bar(dates[1])
    assert [r.reason for r in adapter.rejected_orders] == ["invalid_price"]


# --- Test 7: missing data handling ---------------------------------------------------
def test_missing_data_handling(missing_data_dataset):
    from trading_agent.contracts import BacktestConfig

    config = BacktestConfig(date_from=date(2024, 1, 2), date_to=date(2024, 1, 4),
                            initial_cash=100_000.0, liquidity_cap_pct=0.0)
    adapter = FinsaberBacktestAdapter(missing_data_dataset, config)

    # (a) observation on a date where BBB is absent: BBB omitted, AAA present, no crash.
    obs = adapter.get_observation(date(2024, 1, 3))
    assert "AAA" in obs.bars and "BBB" not in obs.bars

    # (b) order whose execution-date bar is missing and has no reference price -> rejected.
    dates = adapter.dates
    adapter.open_bar(dates[0])
    adapter.submit_order(_order("BBB", "buy", 5, dates[0], None))  # BBB missing on 1/3
    adapter.advance()
    adapter.open_bar(dates[1])
    assert [r.reason for r in adapter.rejected_orders] == ["invalid_price"]
