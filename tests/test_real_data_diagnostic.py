"""Real-data diagnostic report tests.

These tests intentionally avoid performance assertions. The diagnostic answers
whether the system is alive enough for a main experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_agent.experiments import real_data_diagnostic


def test_payload_reports_liveness_margins_states_news_and_reserved_periods():
    loop_result = type("Loop", (), {})()
    loop_result.confidence_diagnostics = {
        "technical": {
            "n": 10,
            "real_judgment_count": 9,
            "real_judgment_rate": 0.9,
            "safe_default_count": 1,
            "safe_default_reasons": {"data_unavailable": 1, "parse_error": 0, "unknown": 0},
        },
        "factor": {
            "n": 10,
            "real_judgment_count": 8,
            "real_judgment_rate": 0.8,
            "safe_default_count": 2,
            "safe_default_reasons": {"data_unavailable": 0, "parse_error": 2, "unknown": 0},
        },
        "news": {
            "n": 10,
            "real_judgment_count": 8,
            "real_judgment_rate": 0.8,
            "safe_default_count": 2,
            "safe_default_reasons": {"data_unavailable": 2, "parse_error": 0, "unknown": 0},
        },
        "fundamental": {
            "n": 10,
            "real_judgment_count": 10,
            "real_judgment_rate": 1.0,
            "safe_default_count": 0,
            "safe_default_reasons": {"data_unavailable": 0, "parse_error": 0, "unknown": 0},
        },
    }
    loop_result.expert_view_records = {
        "technical": [
            {"market_state": "up_stress", "direction": "long", "confidence": 0.71, "safe_default": False},
            {"market_state": "down_calm", "direction": "flat", "confidence": 0.31, "safe_default": False},
        ],
        "factor": [{"market_state": "up_stress", "direction": "flat", "confidence": 0.42, "safe_default": False}],
        "news": [{"market_state": "down_stress", "direction": "long", "confidence": 0.62, "safe_default": False}],
        "fundamental": [{"market_state": "up_stress", "direction": "long", "confidence": 0.52, "safe_default": False}],
    }
    loop_result.equal_weight_score_records = [
        {"score": 0.21, "market_state": "up_stress", "direction": "flat"},
        {"score": 0.29, "market_state": "down_calm", "direction": "flat"},
        {"score": 0.41, "market_state": "down_stress", "direction": "long"},
    ]
    loop_result.context_records = [
        {
            "technical": {"by_ticker": {"AAA": {"available": False, "unavailable_reason": "insufficient_trailing_history"}}},
            "news": {
                "raw_item_count": 3,
                "dedup_item_count": 2,
                "nlp_model_name": "stub-news",
                "nlp_model_revision": "r1",
                "by_ticker": {"AAA": {"available": True}, "BBB": {"available": False}},
            },
        }
    ]
    loop_result.episodes = [
        type("Episode", (), {"market_state": "up_stress", "action": "long"})(),
        type("Episode", (), {"market_state": "down_calm", "action": "flat"})(),
        type("Episode", (), {"market_state": "down_stress", "action": "long"})(),
    ]
    loop_result.market_states = [
        ("2020-01-02", "up_stress"),
        ("2020-01-03", "down_calm"),
        ("2020-01-06", "down_stress"),
    ]

    payload = real_data_diagnostic.build_diagnostic_payload(
        loop_result,
        manifest={
            "mode": "DIAGNOSTIC",
            "diagnostic_window": {"start": "2017-01-01", "end": "2019-12-31"},
            "reserved_test_periods": [
                {"name": "recent_holdout", "start": "2024-01-01", "end": "2025-12-31"}
            ],
            "decision_thresholds": [0.2, 0.3, 0.4],
        },
    )

    assert payload["mode"] == "DIAGNOSTIC"
    assert payload["manifest"]["reserved_test_periods"][0]["name"] == "recent_holdout"
    assert payload["expert_liveness"]["technical"]["pass"] is True
    assert payload["confidence_distributions"]["technical"]["all_histogram"]["0.7-0.8"] == 1
    assert payload["direction_mix"]["technical"]["by_state"]["up_stress"]["long_fraction"] == 1.0
    assert payload["score_margin"]["thresholds"]["0.3000"]["within_0.10_fraction"] > 0
    assert payload["score_margin"]["quantiles"] == {
        "p10": pytest.approx(0.226),
        "p25": pytest.approx(0.25),
        "p50": pytest.approx(0.29),
        "p75": pytest.approx(0.35),
        "p90": pytest.approx(0.386),
    }
    assert payload["market_state_coverage"]["states_seen"] == 3
    assert payload["expert_state_evidence"]["up_stress"]["technical"]["non_default_evidence_count"] == 1
    assert payload["expert_state_evidence"]["up_stress"]["technical"]["passes_min_support"] is False
    assert payload["news_health"]["raw_item_count"] == 3
    assert payload["news_health"]["ticker_day_coverage_fraction"] == 0.5
    assert payload["technical_unavailable"]["insufficient_trailing_history"] == 1
    assert payload["pass_gates"]["all_experts_live"] is True
    assert payload["pass_gates"]["expert_state_min_support"] is False
    assert "metrics" not in payload


def test_script_writes_diagnostic_report_without_performance_summary(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    script = real_data_diagnostic

    class Selection:
        tickers = ["T1", "T2", "T3", "T4"]
        label = "synthetic"

    monkeypatch.setattr(script, "select_fixed_universe", lambda *args, **kwargs: Selection())
    monkeypatch.setattr(script, "load_panel", lambda *args, **kwargs: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    monkeypatch.setattr(script, "build_expert_signal_sources", lambda **kwargs: script.ExpertSignalSources())

    output = tmp_path / "real_data_diagnostic"
    rc = script.main([
        "--data-root", "unused",
        "--start", mem_backtest_config.date_from.isoformat(),
        "--end", mem_backtest_config.date_to.isoformat(),
        "--names", "4",
        "--reserved-test-period", "recent_holdout:2024-01-01:2025-12-31",
        "--reserved-test-period", "covid_crisis:2020-02-01:2020-06-30",
        "--reflect-every", "10",
        "--reflection-window", "7",
        "--reflection-min-support", "1",
        "--vol-window", "5",
        "--percentile-window", "20",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert payload["mode"] == "DIAGNOSTIC"
    assert len(payload["manifest"]["reserved_test_periods"]) == 2
    assert payload["manifest"]["expert_transport"] == "heuristic"
    assert payload["manifest"]["expert_temperature"] == 0.0
    assert payload["manifest"]["codex_sdk_available"] in (True, False)
    assert payload["manifest"]["signal_only"] is True
    assert payload["manifest"]["performance_metrics_computed"] is False
    assert payload["manifest"]["window_role"] == "diagnostic_or_validation_only"
    assert payload["manifest"]["news_sentiment_enabled"] is False
    assert "Expert Liveness" in markdown
    assert "Confidence Distributions" in markdown
    assert "Direction Mix" in markdown
    assert "Score Margin" in markdown
    assert "Technical Unavailable" in markdown
    assert "News Health" in markdown
    assert "Returns" not in markdown
    assert "Sharpe" not in markdown
    assert "Drawdown" not in markdown
    assert "turnover" not in markdown.lower()
    assert "equity_curve" not in json.dumps(payload)


def test_script_constrains_universe_to_factor_artifact_tickers(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    script = real_data_diagnostic
    captured = {}

    class Selection:
        tickers = ["AAA", "BBB"]
        label = "factor-constrained"

    def fake_select(*args, **kwargs):
        captured["allowed_tickers"] = tuple(kwargs.get("allowed_tickers") or ())
        return Selection()

    monkeypatch.setattr(script, "available_tickers_from_sqlite_qlib_factors", lambda path: ("AAA", "BBB"))
    monkeypatch.setattr(script, "select_fixed_universe", fake_select)
    monkeypatch.setattr(script, "load_panel", lambda *args, **kwargs: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    monkeypatch.setattr(script, "build_expert_signal_sources", lambda **kwargs: script.ExpertSignalSources())

    rc = script.main([
        "--data-root", "unused",
        "--start", mem_backtest_config.date_from.isoformat(),
        "--end", mem_backtest_config.date_to.isoformat(),
        "--names", "4",
        "--alpha158-factors-db", str(tmp_path / "factors.sqlite"),
        "--reflect-every", "10",
        "--reflection-window", "7",
        "--reflection-min-support", "1",
        "--vol-window", "5",
        "--percentile-window", "20",
        "--output", str(tmp_path / "diagnostic"),
    ])

    assert rc == 0
    assert captured["allowed_tickers"] == ("AAA", "BBB")


def test_preflight_only_does_not_build_experts_or_signal_sources(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    script = real_data_diagnostic

    class Selection:
        tickers = ["T1", "T2", "T3", "T4"]
        label = "synthetic"

    monkeypatch.setattr(script, "select_fixed_universe", lambda *args, **kwargs: Selection())
    monkeypatch.setattr(script, "load_panel", lambda *args, **kwargs: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    monkeypatch.setattr(script, "build_experts", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no experts")))
    monkeypatch.setattr(script, "build_expert_signal_sources", lambda **kwargs: (_ for _ in ()).throw(AssertionError("no sources")))

    output = tmp_path / "window_preflight"
    rc = script.main([
        "--data-root", "unused",
        "--start", mem_backtest_config.date_from.isoformat(),
        "--end", mem_backtest_config.date_to.isoformat(),
        "--names", "4",
        "--preflight-only",
        "--reserved-test-period", "recent_holdout:2024-01-01:2025-12-31",
        "--vol-window", "5",
        "--percentile-window", "20",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["mode"] == "DIAGNOSTIC_WINDOW_PREFLIGHT"
    assert payload["window_preflight"]["window_role"] == "diagnostic_or_validation_only"
    assert "expert_liveness" not in payload


def test_script_rejects_diagnostic_window_overlapping_reserved_period(tmp_path):
    with pytest.raises(ValueError, match="reserved test period"):
        real_data_diagnostic.main([
            "--data-root", "unused",
            "--start", "2024-01-01",
            "--end", "2024-12-31",
            "--reserved-test-period", "recent_holdout:2024-01-01:2025-12-31",
            "--output", str(tmp_path / "x"),
        ])


def test_diagnostic_parser_accepts_codex_sdk_backend():
    parser = real_data_diagnostic.build_parser()

    args = parser.parse_args([
        "--data-root", "unused",
        "--start", "2022-01-03",
        "--end", "2022-03-31",
        "--expert-backend", "codex_sdk",
    ])

    assert args.expert_backend == "codex_sdk"


def test_signal_only_result_payload_does_not_require_backtest_object():
    result = real_data_diagnostic.SignalOnlyDiagnosticResult(
        n_dates=3,
        n_active_dates=3,
        market_states=[
            ("2024-01-02", "up_stress"),
            ("2024-01-03", "down_calm"),
            ("2024-01-04", "down_stress"),
        ],
        confidence_diagnostics={
            expert: {
                "n": 15,
                "real_judgment_count": 15,
                "real_judgment_rate": 1.0,
                "safe_default_count": 0,
                "safe_default_reasons": {
                    "data_unavailable": 0,
                    "parse_error": 0,
                    "timeout": 0,
                    "unknown": 0,
                },
            }
            for expert in ("technical", "factor", "news", "fundamental")
        },
        expert_view_records={
            expert: [
                {"market_state": state, "direction": "long", "confidence": 0.7, "safe_default": False}
                for state in ("up_stress", "down_calm", "down_stress")
                for _ in range(5)
            ]
            for expert in ("technical", "factor", "news", "fundamental")
        },
        equal_weight_score_records=[
            {"score": score, "market_state": "up_stress", "direction": "flat"}
            for score in (0.2, 0.3, 0.4)
        ],
        context_records=[],
        episode_counts_by_state_action={
            "up_stress": {"long": 5, "flat": 0},
            "down_calm": {"long": 5, "flat": 0},
            "down_stress": {"long": 5, "flat": 0},
        },
        parallel_serial_check={"enabled": False, "passed": None, "comparisons": []},
    )

    payload = real_data_diagnostic.build_diagnostic_payload(
        result,
        manifest={
            "mode": "DIAGNOSTIC",
            "signal_only": True,
            "performance_metrics_computed": False,
            "reflection_min_support": 5,
            "diagnostic_window": {"start": "2024-01-02", "end": "2024-01-04"},
            "reserved_test_periods": [],
            "decision_thresholds": [0.2, 0.3, 0.4],
        },
    )

    assert payload["pass_gates"]["overall"] is True
    assert "backtest" not in payload
    assert "equity_curve" not in json.dumps(payload)


def test_window_preflight_reports_support_and_rejects_reserved_overlap():
    with pytest.raises(ValueError, match="reserved test period"):
        real_data_diagnostic.build_window_preflight(
            market_states=[("2024-01-02", "up_stress")],
            tickers=("AAA", "BBB"),
            min_support=5,
            start=real_data_diagnostic._parse_date("2024-01-02"),
            end=real_data_diagnostic._parse_date("2024-01-31"),
            reserved_periods=[{"name": "reserved", "start": "2024-01-01", "end": "2024-12-31"}],
        )

    report = real_data_diagnostic.build_window_preflight(
        market_states=[
            ("2022-01-03", "up_stress"),
            ("2022-01-04", "down_calm"),
            ("2022-01-05", "down_stress"),
        ],
        tickers=("AAA", "BBB"),
        min_support=5,
        start=real_data_diagnostic._parse_date("2022-01-03"),
        end=real_data_diagnostic._parse_date("2022-01-31"),
        reserved_periods=[],
    )

    assert report["window_pass"] is False
    assert report["recommendation"] == "expand_window_do_not_lower_min_support"
    assert report["state_support"]["up_stress"]["estimated_per_name_observations"] == 2


def test_parallel_serial_check_compares_structural_invariants():
    serial = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("long", 0.7, False, None),
            "BBB": real_data_diagnostic._view_signature("flat", 0.2, False, None),
        }
    }
    parallel = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("long", 0.7, False, None),
            "BBB": real_data_diagnostic._view_signature("flat", 0.2, False, None),
        }
    }

    check = real_data_diagnostic.compare_parallel_serial_views(
        [{"date": "2024-01-02", "serial": serial, "parallel": parallel}]
    )

    assert check["enabled"] is True
    assert check["passed"] is True


def test_parallel_serial_check_audits_direction_confidence_without_blocking():
    serial = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("long", 0.7, False, None),
        }
    }
    parallel = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("flat", 0.2, False, None),
        }
    }

    check = real_data_diagnostic.compare_parallel_serial_views(
        [{"date": "2024-01-02", "serial": serial, "parallel": parallel}]
    )

    assert check["passed"] is True
    assert check["direction_confidence_matched"] is False
    assert check["comparisons"][0]["structural_matched"] is True
    assert check["comparisons"][0]["direction_confidence_mismatches"] == 1


def test_parallel_serial_check_blocks_structural_mismatch():
    serial = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("long", 0.7, False, None),
            "BBB": real_data_diagnostic._view_signature("flat", 0.0, True, "parse_error"),
        }
    }
    parallel = {
        "technical": {
            "AAA": real_data_diagnostic._view_signature("long", 0.7, False, None),
        }
    }

    check = real_data_diagnostic.compare_parallel_serial_views(
        [{"date": "2024-01-02", "serial": serial, "parallel": parallel}]
    )

    assert check["passed"] is False
    assert check["comparisons"][0]["structural_matched"] is False
    assert check["comparisons"][0]["ticker_coverage_mismatches"] == 1


def test_diagnostic_progress_printer_reports_dates_and_experts(capsys):
    callback = real_data_diagnostic._progress_printer(enabled=True)

    callback({"event": "date_start", "date": "2024-01-02"})
    callback({"event": "expert_start", "date": "2024-01-02", "expert_id": "technical"})
    callback({"event": "expert_done", "date": "2024-01-02", "expert_id": "technical", "n_tickers": 5})
    callback({"event": "date_done", "date": "2024-01-02", "market_state": "up_stress", "active": True, "long_count": 2, "n_tickers": 5})

    out = capsys.readouterr().out
    assert "[diagnostic] date_start 2024-01-02" in out
    assert "[diagnostic] expert_start 2024-01-02 technical" in out
    assert "[diagnostic] expert_done 2024-01-02 technical n_tickers=5" in out
    assert "[diagnostic] date_done 2024-01-02 state=up_stress active=True long=2/5" in out
