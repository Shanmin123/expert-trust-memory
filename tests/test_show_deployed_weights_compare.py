"""--compare checks a forward run against the frozen snapshots.

The claim the forward demonstration rests on is that the weights the memory
learned are the weights that were deployed. This is the check that says so, so
it has to read the field the run actually writes, and it has to compare a
state-scope stock against the state the run was in rather than the state the
stock certified.
"""

from __future__ import annotations

import json

import pytest

from scripts import show_deployed_weights as tool


def write_report(directory, symbol, arm, weights, market_state):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{symbol}__{arm}.json").write_text(
        json.dumps({
            "market_state": market_state,
            "memory_applied": {"weights": weights, "market_state": market_state},
            "aggregation": {"weights": weights},
        }),
        encoding="utf-8",
    )


def deployed(symbol, arm, state):
    values, _ = tool.weights(symbol, arm, state)
    return {name: values[name] for name in tool.ORDER}


def run(capsys, directory):
    """The comparison section only. The snapshot table above it prints a line
    per stock too, so a search over the whole output finds that one first."""
    code = tool.main(["--compare", str(directory)])
    out = capsys.readouterr().out
    marker = "Comparing against"
    return code, out[out.index(marker):] if marker in out else out


def line_for(out, symbol):
    return next(l for l in out.splitlines() if l.strip().startswith(symbol))


def test_the_deployed_weights_are_read_from_the_field_the_run_writes(tmp_path,
                                                                     capsys):
    """The forward run records them under memory_applied. Looking only at the
    older manifest field reported every stock as having no weights at all."""
    for symbol, arm, state in tool.PLAN:
        live = "up_stress" if arm == "state" else state
        write_report(tmp_path, symbol, arm, deployed(symbol, arm, live), live)
    code, out = run(capsys, tmp_path)
    assert "carries no deployed weights" not in out
    assert out.count("MATCH") == len(tool.PLAN)
    assert "DIFFERS" not in out
    assert code == 0


def test_a_state_scope_stock_is_compared_against_the_state_the_run_was_in(
    tmp_path, capsys
):
    """ADSK certified in down_stress and holds equal weights everywhere else.
    A run that never entered down_stress should read as a match, not as a
    disagreement with the cell it certified."""
    stateful = [row for row in tool.PLAN if row[1] == "state"]
    assert stateful, "no state-scope stock in the plan"
    for symbol, arm, state in tool.PLAN:
        live = "up_stress" if arm == "state" else state
        write_report(tmp_path, symbol, arm, deployed(symbol, arm, live), live)
    _, out = run(capsys, tmp_path)
    for symbol, _arm, certified in stateful:
        assert certified != "up_stress", "pick a stock certified elsewhere"
        line = line_for(out, symbol)
        assert "MATCH" in line, line
        assert "0.333" in line, "the run was not in the certified state"


def test_a_deployed_weight_that_disagrees_is_reported(tmp_path, capsys):
    symbol, arm, state = tool.PLAN[0]
    wrong = dict(deployed(symbol, arm, state))
    wrong[tool.ORDER[0]] += 0.05
    write_report(tmp_path, symbol, arm, wrong, state)
    for other, other_arm, other_state in tool.PLAN[1:]:
        live = "up_stress" if other_arm == "state" else other_state
        write_report(tmp_path, other, other_arm,
                     deployed(other, other_arm, live), live)
    _, out = run(capsys, tmp_path)
    assert "DIFFERS" in out
    assert out.count("MATCH") == len(tool.PLAN) - 1


def test_a_missing_report_is_named_rather_than_skipped(tmp_path, capsys):
    code, out = run(capsys, tmp_path)
    assert "NO REPORT" in out
    assert code == 1, "a run with nothing to compare must not read as a pass"


@pytest.mark.parametrize("field", ["memory_applied", "aggregation"])
def test_either_recorded_shape_is_accepted(tmp_path, capsys, field):
    symbol, arm, state = tool.PLAN[0]
    values = deployed(symbol, arm, state)
    payload = {"market_state": state, field: {"weights": values}}
    (tmp_path / f"{symbol}__{arm}.json").write_text(json.dumps(payload),
                                                    encoding="utf-8")
    _, out = run(capsys, tmp_path)
    line = line_for(out, symbol)
    assert "MATCH" in line, line
