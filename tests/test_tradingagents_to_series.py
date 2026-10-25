"""The decision parser must read every label the agent actually uses.

Measured over 2332 recorded decisions TradingAgents used five labels: Hold 71.4%,
Overweight 17.6%, Underweight 3.7%, Buy 1.3%, Sell 0.2%. The first version of this parser
knew only Buy, Sell and Hold, so it dropped 496 calls into "said nothing" and described an
agent that took 36 positions instead of 532. These tests pin all five.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tradingagents_to_series",
    Path(__file__).resolve().parents[1] / "scripts" / "tradingagents_to_series.py",
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


@pytest.mark.parametrize("text,expected", [
    ("Buy", "LONG"),
    ("Overweight", "LONG"),
    ("Sell", "FLAT"),
    ("Underweight", "FLAT"),
    ("Hold", "HOLD"),
])
def test_every_label_the_agent_actually_emits(text, expected):
    assert mod.parse_decision(text) == expected


def test_case_and_whitespace_do_not_matter():
    assert mod.parse_decision("  overweight  ") == "LONG"
    assert mod.parse_decision("UNDERWEIGHT") == "FLAT"


def test_underweight_is_not_read_as_a_bare_verb():
    """The substring trap: 'underweight' must not fall through to something else."""
    assert mod.parse_decision("Underweight") == "FLAT"
    assert mod.parse_decision("Overweight") == "LONG"


def test_the_last_label_wins_because_the_call_comes_last():
    assert mod.parse_decision("Some argue buy, others sell. Final: Hold") == "HOLD"
    assert mod.parse_decision("Hold for now, but I recommend Overweight") == "LONG"


def test_nothing_decisive_returns_none_so_the_position_is_maintained():
    assert mod.parse_decision("") is None
    assert mod.parse_decision(None) is None
    assert mod.parse_decision("The outlook is uncertain.") is None


def test_a_word_containing_a_label_does_not_count():
    assert mod.parse_decision("rebuying") is None
    assert mod.parse_decision("holder") is None


def test_the_long_and_flat_word_lists_are_disjoint():
    assert not set(mod.LONG_WORDS) & set(mod.FLAT_WORDS)
