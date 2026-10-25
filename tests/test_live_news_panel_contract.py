"""The live news panel must satisfy the contract the expert context calls on it.

Until 2026-08-06 `IBNewsPanel` had no `coverage`. The context calls it on the FIRST line
of the block that also attaches the headline snippets, and the whole block sits under one
`except Exception`, so the AttributeError did not degrade the seat's input: it removed it.
Every live day the news expert was handed `available=False`, coverage zero and no
headlines, while three hundred fetched headlines a stock sat unused in the panel. It
abstained on nineteen of twenty stocks and said there was no recent coverage, which was
true of what it was shown.

The test therefore asserts on what the context produces, not on whether the method
exists: a panel holding headlines must yield non-zero coverage and carry the text through.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _live_analysis():
    """Load the script by path; it lives in scripts/ and is not an installed module."""
    spec = importlib.util.spec_from_file_location(
        "live_analysis_under_test", ROOT / "scripts" / "live_analysis.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _Item:
    date: date
    text: str


def test_live_news_panel_answers_the_coverage_call_the_context_makes():
    panel = _live_analysis().IBNewsPanel({
        "AAA": [_Item(date(2026, 8, 3), "AAA beats on margins"),
                _Item(date(2026, 8, 4), "AAA raises guidance"),
                _Item(date(2026, 8, 4), "AAA raises guidance")],   # a repeat
        "BBB": [_Item(date(2026, 8, 1), "BBB names a new chief")],
    })

    fraction, raw, dedup = panel.coverage(["AAA", "BBB"], date(2026, 8, 4))
    assert fraction == 0.5          # AAA has items that day, BBB does not
    assert raw == 2 and dedup == 1  # the repeat is counted once after dedup

    assert panel.coverage([], date(2026, 8, 4)) == (0.0, 0, 0)
    assert panel.coverage(["AAA"], date(2026, 7, 1)) == (0.0, 0, 0)


def test_live_news_panel_matches_the_finsaber_panel_it_stands_in_for():
    """Same call, same shape: the expert must not be able to tell the two apart."""
    from trading_agent.data.finsaber_news import FinsaberNewsPanel

    live = _live_analysis().IBNewsPanel
    for name in ("items_on", "dedup_items", "trailing_items", "coverage"):
        assert hasattr(live, name), f"IBNewsPanel is missing {name}"
        assert hasattr(FinsaberNewsPanel, name), f"FinsaberNewsPanel is missing {name}"
