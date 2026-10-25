import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from data_collection.storage import quote_identifier


def test_finsaber_qlib_pipeline_exports_prices_and_builds_factor_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    from data_collection import finsaber_qlib_pipeline as pipeline

    class FakePanel:
        def get_price_dataframe(self, *, tickers, date_from, date_to, adjust):
            assert tickers == ["AAPL", "MSFT"]
            assert date_from == date(2022, 1, 3)
            assert date_to == date(2022, 1, 4)
            assert adjust is True
            return pd.DataFrame(
                [
                    {
                        "date": "2022-01-03",
                        "symbol": "AAPL",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1000.0,
                        "adjusted_close": 100.5,
                    },
                    {
                        "date": "2022-01-04",
                        "symbol": "AAPL",
                        "open": 101.0,
                        "high": 102.0,
                        "low": 100.0,
                        "close": 101.5,
                        "volume": 1100.0,
                        "adjusted_close": 101.5,
                    },
                    {
                        "date": "2022-01-03",
                        "symbol": "MSFT",
                        "open": 200.0,
                        "high": 201.0,
                        "low": 199.0,
                        "close": 200.5,
                        "volume": 2000.0,
                        "adjusted_close": 200.5,
                    },
                ]
            )

    calls = []

    monkeypatch.setattr(
        pipeline,
        "load_panel",
        lambda root, start, end, tickers, modalities: FakePanel(),
    )

    def fake_export_provider(prices_db, provider_dir, *, market, symbols, force):
        calls.append(("provider", Path(prices_db), Path(provider_dir), market, symbols, force))
        return 2, 2

    def fake_compute(provider_dir, output_db, *, market, symbols, force):
        calls.append(("factors", Path(provider_dir), Path(output_db), market, symbols, force))
        return 2, 4

    monkeypatch.setattr(pipeline, "export_qlib_provider", fake_export_provider)
    monkeypatch.setattr(pipeline, "compute_direct_qlib_features", fake_compute)

    price_db = tmp_path / "sp500_prices.db"
    provider_dir = tmp_path / "qlib_provider" / "us_sp500"
    factors_db = tmp_path / "sp500_qlib_direct_factors.db"
    data_root = tmp_path / "finsaber"
    (data_root / "price_daily").mkdir(parents=True)

    manifest = pipeline.build_finsaber_qlib_pipeline(
        data_root=data_root,
        start=date(2022, 1, 3),
        end=date(2022, 1, 4),
        symbols=["AAPL", "MSFT"],
        prices_db=price_db,
        provider_dir=provider_dir,
        factors_db=factors_db,
        force=True,
    )

    with sqlite3.connect(price_db) as conn:
        aapl_rows = conn.execute(
            f"SELECT date, close, average, symbol FROM {quote_identifier('prices_AAPL')} ORDER BY date"
        ).fetchall()
        msft_rows = conn.execute(
            f"SELECT date, close, average, symbol FROM {quote_identifier('prices_MSFT')} ORDER BY date"
        ).fetchall()

    assert aapl_rows == [
        ("2022-01-03", 100.5, 100.125, "AAPL"),
        ("2022-01-04", 101.5, 101.125, "AAPL"),
    ]
    assert msft_rows == [("2022-01-03", 200.5, 200.125, "MSFT")]
    assert calls == [
        ("provider", price_db, provider_dir, "sp500", ["AAPL", "MSFT"], True),
        ("factors", provider_dir, factors_db, "sp500", ["AAPL", "MSFT"], True),
    ]
    assert manifest["pipeline"] == "finsaber_to_qlib_factors"
    assert manifest["price_rows"] == 3
    assert manifest["price_symbols"] == 2
    assert manifest["qlib_provider"]["instruments"] == 2
    assert manifest["qlib_factors"]["symbols"] == 2
    assert manifest["artifacts"]["alpha158_factors_db"] == str(factors_db)
