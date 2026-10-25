import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiment_protocols" / "main_walk_forward.json"


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _overlaps(left: dict, right: dict) -> bool:
    return _date(left["start"]) <= _date(right["end"]) and _date(right["start"]) <= _date(left["end"])


def test_main_protocol_reserves_test_periods_before_main_experiment() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    reserved = protocol["reserved_test_periods"]

    assert [period["name"] for period in reserved] == [
        "gfc_crisis_test",
        "covid_crisis_test",
        "recent_holdout_test",
    ]
    assert protocol["protocol_invariants"]["embargo_trading_days"] >= 5
    assert protocol["protocol_invariants"]["minimum_non_diagnostic_embargo_trading_days"] == 5
    assert protocol["protocol_invariants"]["test_windows_used_once"] is True


def test_diagnostic_windows_do_not_overlap_reserved_test_periods() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    for diagnostic in protocol["diagnostic_windows"]:
        assert diagnostic["prohibited_as_test"] is True
        for reserved in protocol["reserved_test_periods"]:
            assert not _overlaps(diagnostic, reserved), (diagnostic["name"], reserved["name"])


def test_factor_artifact_span_covers_reserved_tests_and_diagnostic_window() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    span = protocol["data_scope"]["factor_artifact_span"]

    windows = protocol["reserved_test_periods"] + protocol["diagnostic_windows"]
    for window in windows:
        assert _date(span["start"]) <= _date(window["start"])
        assert _date(window["end"]) <= _date(span["end"])


def test_main_protocol_pins_backend_roles() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    backend = protocol["backend_decision_required"]

    assert backend["expert_backend_for_main"] == "openai"
    assert backend["reflection_backend_for_main"] == "deterministic_primary_with_llm_ablation"
    assert backend["llm_reflection_role"] == "ablation_or_extension"


def test_main_protocol_preregisters_transaction_cost_levels() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    costs = protocol["transaction_cost_protocol"]

    assert costs["model"] == "bps_per_side_on_traded_notional"
    assert costs["headline_bps_per_side"] == 10.0
    assert costs["sensitivity_bps_per_side"] == 25.0
    assert costs["arm_identical"] is True
    assert costs["sensitivity_reports_required"] is True


def test_main_protocol_records_optional_indicator_decision() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    decision = protocol["technical_indicator_decision"]

    assert decision["decision"] == "skip_for_main_experiment"
    assert {"RSI", "MACD", "moving_average_gap", "bollinger_percent_b"} == set(
        decision["optional_indicators"]
    )
    assert "Momentum, reversal, and low-volatility" in decision["rationale"]
