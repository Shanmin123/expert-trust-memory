"""The forward run must cover exactly what the thesis pre-registers.

Table tab:forward fixes, before the run, the arm and the weights each stock should
retrieve. On 2026-08-02 the runner covered eight names and included TSLA, which was
held out of the random draw, while the table listed all twenty. A pre-registration
that the runner does not follow is not a pre-registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DRAW = {"ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT"}


def _plan():
    from paper_forward_run import PLAN
    return PLAN


def test_the_plan_covers_the_whole_draw():
    names = {p[0] for p in _plan()}
    assert names == DRAW, (
        f"missing {sorted(DRAW - names)}, extra {sorted(names - DRAW)}"
    )


def test_no_held_out_name_is_traded():
    # TSLA is a development/reference name held out of the draw; trading it here would
    # misdescribe the universe as random.
    assert "TSLA" not in {p[0] for p in _plan()}


def test_client_ids_are_unique():
    ids = [p[2] for p in _plan()]
    assert len(set(ids)) == len(ids), f"duplicate client ids: {ids}"


def test_each_stock_takes_the_arm_the_thesis_registers():
    """ADSK deploys its state memory; every other drawn stock deploys global."""
    arms = {p[0]: p[1] for p in _plan()}
    assert arms["ADSK"] == "state"
    assert {t for t, a in arms.items() if a == "state"} == {"ADSK"}


def test_the_snapshots_the_plan_names_exist():
    mem = ROOT / "frozen_memory"
    missing = [f"{n}__{a}.json" for n, a, *_ in _plan()
               if not (mem / f"{n}__{a}.json").exists()]
    assert not missing, f"missing from frozen_memory/: {missing}"
