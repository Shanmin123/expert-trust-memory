from data_collection.config import Settings
from data_collection.reporting import write_data_collection_summary


def test_summary_report_writes_without_databases(tmp_path) -> None:
    settings = Settings(
        ib_host="127.0.0.1",
        ib_port=7497,
        ib_client_id=1,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        rapidapi_key=None,
        rapidapi_host="example.com",
    )
    output = write_data_collection_summary(settings)
    text = output.read_text(encoding="utf-8")
    assert "Data Collection Summary" in text
    assert "Database present: False" in text
