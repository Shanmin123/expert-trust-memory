from pathlib import Path

import pandas as pd

from data_collection.config import Settings
from data_collection.universe import (
    load_universe_snapshot,
    normalize_spy_holdings_frame,
    snapshot_path,
    write_snapshot_files,
)


def test_normalize_spy_holdings_filters_non_equity_rows() -> None:
    holdings = pd.DataFrame(
        [
            {
                "Name": "APPLE INC",
                "Ticker": "AAPL",
                "Identifier": "037833100",
                "SEDOL": "2046251",
                "Weight": "7.0",
                "Shares Held": "100",
                "Local Currency": "USD",
            },
            {
                "Name": "BERKSHIRE HATHAWAY INC-CL B",
                "Ticker": "BRK.B",
                "Identifier": "084670702",
                "SEDOL": "2073390",
                "Weight": "1.5",
                "Shares Held": "10",
                "Local Currency": "USD",
            },
            {
                "Name": "US DOLLAR",
                "Ticker": "-",
                "Identifier": "999USDZ92",
                "SEDOL": "-",
                "Weight": "0.1",
                "Shares Held": "1000",
                "Local Currency": "USD",
            },
            {
                "Name": "CONTRA ROW",
                "Ticker": "2602335D",
                "Identifier": "436CVR021",
                "SEDOL": "-",
                "Weight": "0.000003",
                "Shares Held": "1",
                "Local Currency": "USD",
            },
        ]
    )

    normalized = normalize_spy_holdings_frame(
        holdings,
        source_as_of="2026-05-28",
        source_url="https://example.test/spy.xlsx",
        captured_at="2026-05-30T12:00:00",
    )

    assert normalized["symbol"].tolist() == ["AAPL", "BRK-B"]
    assert normalized["ib_symbol"].tolist() == ["AAPL", "BRK B"]
    assert normalized["source_as_of"].unique().tolist() == ["2026-05-28"]


def test_write_and_load_universe_snapshot(tmp_path: Path) -> None:
    settings = Settings(
        ib_host="127.0.0.1",
        ib_port=4002,
        ib_client_id=1,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        rapidapi_key=None,
        rapidapi_host="example.test",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "BRK-B",
                "ib_symbol": "BRK B",
                "name": "Berkshire Hathaway",
                "source_as_of": "2026-05-28",
            }
        ]
    )

    dated_path, current_path = write_snapshot_files(df, settings)
    constituents = load_universe_snapshot(current_path)

    assert dated_path == snapshot_path(settings, "2026-05-28")
    assert current_path == settings.sp500_universe_snapshot
    assert constituents[0].symbol == "BRK-B"
    assert constituents[0].ib_symbol == "BRK B"
