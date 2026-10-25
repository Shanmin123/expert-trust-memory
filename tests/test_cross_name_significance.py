"""Tests for the cross-name significance tests."""

from __future__ import annotations

import json

from scripts import cross_name_significance as cns


def test_paired_test_flags_a_consistent_positive_effect():
    # Every name favours the arm by a similar margin -> small p, positive CI.
    deltas = [0.30, 0.28, 0.35, 0.31, 0.29, 0.33, 0.27, 0.32]
    r = cns.paired_arm_test(deltas)
    assert r["n"] == 8
    assert r["n_positive"] == 8
    assert r["mean_delta"] > 0.25
    assert r["p_value"] is not None and r["p_value"] < 0.05
    assert r["significant_at_05"] is True
    lo, hi = r["bootstrap_ci95"]
    assert lo > 0                                   # CI excludes zero


def test_paired_test_reports_nonsignificant_for_mixed_effect():
    # Mixed signs, small n -> not significant.
    deltas = [0.3, -0.4, 0.1, -0.2, 0.05]
    r = cns.paired_arm_test(deltas)
    assert r["significant_at_05"] is False
    lo, hi = r["bootstrap_ci95"]
    assert lo < 0 < hi                              # CI straddles zero


def test_sign_test_fallback_matches_when_scipy_missing(monkeypatch):
    # Force the scipy import to fail; the sign-test fallback must still return a p.
    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    p, method = cns._wilcoxon_p([0.3, 0.2, 0.4, 0.1, 0.35])
    assert method == "sign-test"
    assert p is not None and p < 0.10               # 5/5 positive -> 2*(1/32)


def test_arm_deltas_reads_paired_metrics(tmp_path):
    def _w(name, state, equal):
        (tmp_path / f"{name}.json").write_text(json.dumps({"arms": {
            "market_state_memory": {"metrics": {"sharpe": state}},
            "equal_weight": {"metrics": {"sharpe": equal}},
        }}), encoding="utf-8")

    _w("TSLA", 0.34, -0.32)
    _w("NFLX", -0.21, -0.67)
    import pytest
    deltas = cns.arm_deltas(tmp_path, ("TSLA", "NFLX", "AMZN"),
                            "market_state_memory", "equal_weight")
    assert deltas == pytest.approx([0.66, 0.46])    # AMZN absent -> skipped
