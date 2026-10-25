"""The branch that submits orders must never be laxer than the branch that previews.

Two safety checks existed only in the dry-run path, so a preview refused to run while
the real submission carried on. Both were found on 2026-08-03, the morning the forward
run was due to start:

1. An empty market proxy. Without SPY the state falls back to a default, every stock
   takes it, and a state-armed stock never reaches the cell it was deployed for.
2. An empty bar history. IB answers an over-long day-unit duration with an empty frame
   rather than an error, so the decision would be made on no data.

These are source-level checks because the two branches are long and share no seam that
can be exercised without a gateway.
"""
from __future__ import annotations

import re
from pathlib import Path

from trading_agent import cli

SRC = Path(cli.__file__).read_text(encoding="utf-8")


def _auto_decide_blocks() -> list[str]:
    """The body of each auto-decide block that requests bars and builds orders.

    The paper command also has auto-decide blocks that only fill in the manifest;
    those request nothing and order nothing, so they are not the ones under test.
    """
    blocks = []
    for m in re.finditer(r"if args\.auto_decide:", SRC):
        tail = SRC[m.end():]
        stop = tail.find("auto_decide_order_intents(")
        if stop < 0:
            continue
        block = tail[:stop]
        # A decision block reaches its order call without opening another one first.
        if "if args.auto_decide:" in block or "request_daily_bars" not in block:
            continue
        blocks.append(block)
    return blocks


def test_both_paper_branches_exist():
    assert len(_auto_decide_blocks()) == 2, (
        "expected a dry-run branch and a connected branch in the paper command"
    )


def test_every_branch_refuses_an_empty_market_proxy():
    for i, block in enumerate(_auto_decide_blocks()):
        assert "returned no bars" in block and "proxy" in block.lower(), (
            f"auto-decide branch {i} does not refuse an empty market proxy; without it "
            "the state silently falls back to a default"
        )


def test_every_branch_refuses_an_empty_bar_history():
    for i, block in enumerate(_auto_decide_blocks()):
        assert "IB returned no bars for" in block, (
            f"auto-decide branch {i} accepts an empty bar history; IB answers an "
            "over-long duration with an empty frame rather than an error"
        )


def test_the_empty_frame_test_degrades_safely():
    """`len(frame)` raises on an object with no __len__; `.empty` defaults to blocking."""
    for i, block in enumerate(_auto_decide_blocks()):
        assert 'getattr(frame, "empty", True)' in block, (
            f"auto-decide branch {i} tests emptiness in a way that can raise instead "
            "of blocking when the connector returns an unexpected object"
        )


def test_every_branch_reads_the_account_before_ordering():
    for i, block in enumerate(_auto_decide_blocks()):
        assert "_held_quantities(" in block, (
            f"auto-decide branch {i} does not read current positions, so a repeated "
            "long re-buys and a stock turned flat is never exited"
        )


def test_every_order_call_passes_the_held_positions():
    for call in re.findall(r"auto_decide_order_intents\((.*?)\n\s*\)\)", SRC, re.S):
        assert "held=held" in call, (
            "an order call omits held positions and falls back to buy-only behaviour: "
            + call.strip()[:120]
        )
