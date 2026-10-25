import pytest

from data_collection.alpha_factors import build_alpha101


def test_build_alpha101_requires_existing_price_database(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Run IB collection first"):
        build_alpha101(
            tmp_path / "missing.db",
            tmp_path / "alphas.db",
            skip_version_check=True,
        )
