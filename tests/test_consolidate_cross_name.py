"""Tests for the cross-name consolidation (robust to missing reports / arms)."""

from __future__ import annotations

import json

from scripts import consolidate_cross_name as ccn


def _report(tmp_path, name, arms):
    payload = {"arms": {k: {"metrics": v} for k, v in arms.items()}}
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _m(sharpe, ret=0.05, dsr=0.1):
    return {
        "total_return": ret, "sharpe": sharpe, "deflated_sharpe_ratio": dsr,
        "max_drawdown": 0.1, "turnover": 10.0, "participation_rate": 0.2,
    }


def test_table_computes_means_and_tolerates_missing(tmp_path):
    # TSLA has all four arms; NFLX lacks the online arm; AMZN report absent.
    _report(tmp_path, "TSLA", {
        "equal_weight": _m(-0.3), "global_memory": _m(-0.4),
        "market_state_memory": _m(0.3), "market_state_memory_online": _m(0.5),
    })
    _report(tmp_path, "NFLX", {
        "equal_weight": _m(-0.6), "global_memory": _m(-0.5),
        "market_state_memory": _m(-0.2),
    })
    lines, sharpe_by_arm = ccn._table(("TSLA", "NFLX", "AMZN", "MSFT"), tmp_path)
    text = "\n".join(lines)
    assert "no report yet" in text                      # AMZN + MSFT absent
    assert "(absent)" in text                           # NFLX online arm missing
    # Means computed only over present values.
    assert sharpe_by_arm["state(frozen)"] == [0.3, -0.2]
    assert sharpe_by_arm["state(online)"] == [0.5]


def test_main_writes_markdown(tmp_path, monkeypatch, capsys):
    _report(tmp_path, "TSLA", {
        "equal_weight": _m(-0.3), "market_state_memory": _m(0.3),
        "market_state_memory_online": _m(0.5),
    })
    out = tmp_path / "cross_name.md"
    rc = ccn.main(["--report-dir", str(tmp_path), "--output", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Reference set" in md and "Random set" in md
    assert "learned trust vs equal" in md
    assert "self-improving vs frozen" in md
