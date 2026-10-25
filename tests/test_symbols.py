from data_collection.symbols import (
    canonical_symbol,
    ib_contract_symbol,
    sqlite_table_name,
)


def test_canonical_symbol_replaces_class_share_dot() -> None:
    assert canonical_symbol(" brk.b ") == "BRK-B"


def test_ib_contract_symbol_uses_space_for_class_share() -> None:
    assert ib_contract_symbol("BRK-B") == "BRK B"


def test_sqlite_table_name_is_safe() -> None:
    assert sqlite_table_name("prices", "BRK-B") == "prices_BRK_B"
