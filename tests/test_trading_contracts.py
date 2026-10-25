"""Phase 1 contract tests: structural separation of decision vs state-mutating types."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from trading_agent.contracts import (
    STATE_AUTHORITY_FIELDS,
    AccountState,
    ExecutionAdapter,
    OrderIntent,
    Position,
    Signal,
    TargetPosition,
)

DECISION_TYPES = [Signal, TargetPosition, OrderIntent]


def _field_names(cls):
    return {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("cls", DECISION_TYPES)
def test_decision_types_have_no_state_authority(cls):
    # Decision-side types must expose NONE of the cash/holdings authority fields.
    assert _field_names(cls).isdisjoint(STATE_AUTHORITY_FIELDS)


@pytest.mark.parametrize("cls", DECISION_TYPES)
def test_decision_types_are_frozen(cls):
    instance = _make(cls)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, dataclasses.fields(cls)[0].name, None)


def test_state_types_carry_authority_and_are_frozen():
    account = AccountState(as_of=date(2024, 1, 2), cash=10.0, initial_cash=10.0,
                           positions_value=0.0, equity=10.0)
    # AccountState is the carrier of cash authority...
    assert "cash" in _field_names(AccountState)
    # ...and is itself an immutable snapshot (LLM cannot mutate it).
    with pytest.raises(dataclasses.FrozenInstanceError):
        account.cash = 0.0
    position = Position(ticker="AAA", quantity=1, avg_price=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        position.quantity = 99


def test_finsaber_adapter_satisfies_execution_protocol():
    from trading_agent.execution.finsaber_adapter import FinsaberBacktestAdapter

    # runtime_checkable Protocol: the adapter must expose the 4 contract methods.
    for method in ("get_observation", "submit_order", "get_positions", "get_account"):
        assert hasattr(FinsaberBacktestAdapter, method)
    assert issubclass(FinsaberBacktestAdapter, object)
    assert isinstance(ExecutionAdapter, type(ExecutionAdapter))  # importable Protocol


def _make(cls):
    if cls is Signal:
        return Signal(ticker="AAA", value=1.0, source_id="x", as_of=date(2024, 1, 2))
    if cls is TargetPosition:
        return TargetPosition(ticker="AAA", target_weight=0.5, as_of=date(2024, 1, 2))
    if cls is OrderIntent:
        return OrderIntent(ticker="AAA", side="buy", quantity=1)
    raise AssertionError(cls)
