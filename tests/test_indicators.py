import pandas as pd
import pytest

from data_collection.indicators import build_indicators, compute_indicators


def make_prices(days: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    close = pd.Series(range(100, 100 + days), dtype="float64")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
            "barCount": 10,
            "average": close,
            "symbol": "AAPL",
        }
    )


def test_initial_rolling_values_are_null() -> None:
    indicators = compute_indicators(make_prices())
    sma20 = indicators[indicators["indicator_name"] == "SMA_20"].reset_index(drop=True)
    assert sma20.loc[:18, "value"].isna().all()
    assert pd.notna(sma20.loc[19, "value"])


def test_future_prices_do_not_change_prior_indicator_values() -> None:
    prices = make_prices()
    baseline = compute_indicators(prices)

    changed = prices.copy()
    changed.loc[100:, "close"] = changed.loc[100:, "close"] * 10
    revised = compute_indicators(changed)

    baseline_cut = baseline[
        (baseline["date"] == prices.loc[80, "date"])
        & (baseline["indicator_name"].isin(["SMA_20", "EMA_12", "ROC_10"]))
    ].sort_values("indicator_name")
    revised_cut = revised[
        (revised["date"] == prices.loc[80, "date"])
        & (revised["indicator_name"].isin(["SMA_20", "EMA_12", "ROC_10"]))
    ].sort_values("indicator_name")

    assert baseline_cut["value"].tolist() == revised_cut["value"].tolist()


def test_build_indicators_requires_existing_price_database(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Run IB collection first"):
        build_indicators(tmp_path / "missing.db", tmp_path / "indicators.db")
