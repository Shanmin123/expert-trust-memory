"""Tests for the E0 LLM expert reliability diagnostic."""

from __future__ import annotations

from datetime import date

import pytest

from trading_agent.experiments.expert_reliability_diagnostic import (
    ViewOutcome,
    calibration_by_state,
    directional_behavior_by_state,
    expert_reliability_report,
    expert_signed_confidence_correlation,
    view_direction_summary,
    view_outcomes_from_episodes,
)
from trading_agent.memory.expert_error import ExpertErrorRecord
from trading_agent.memory.expert_reflection import ATTRIBUTION_COMPONENTS, AttributedEpisode


def _outcome(expert, state, direction, conf, correct, day=1, ticker="AAA"):
    return ViewOutcome(
        decision_date=f"2022-01-0{day}",
        ticker=ticker,
        market_state=state,
        expert_id=expert,
        direction=direction,
        confidence=conf,
        realized_next_return=(0.01 if correct else -0.01),
        direction_correct=correct,
    )


def test_signed_confidence_is_zero_for_flat():
    assert _outcome("t", "up_stress", "long", 0.7, True).signed_confidence == 0.7
    assert _outcome("t", "up_stress", "flat", 0.7, True).signed_confidence == 0.0


def test_calibration_reports_overconfidence_gap():
    # Confidence 0.9 but only half correct -> positive overconfidence gap.
    outcomes = [
        _outcome("technical", "down_stress", "long", 0.9, correct, day=d)
        for d, correct in enumerate([True, False, True, False], start=1)
    ]
    cal = calibration_by_state(outcomes)["down_stress"]["technical"]
    assert cal["n"] == 4
    assert cal["mean_confidence"] == pytest.approx(0.9)
    assert cal["empirical_accuracy"] == pytest.approx(0.5)
    assert cal["overconfidence_gap"] == pytest.approx(0.4)
    assert cal["bins"][-1]["lo"] == pytest.approx(0.8)


def test_calibration_separates_market_states():
    outcomes = [
        _outcome("technical", "up_calm", "long", 0.6, True, day=1),
        _outcome("technical", "down_stress", "long", 0.6, False, day=2),
    ]
    cal = calibration_by_state(outcomes)
    assert cal["up_calm"]["technical"]["empirical_accuracy"] == pytest.approx(1.0)
    assert cal["down_stress"]["technical"]["empirical_accuracy"] == pytest.approx(0.0)


def test_directional_behavior_reports_long_rate_by_state():
    outcomes = [
        _outcome("news", "up_stress", "long", 0.5, True, day=1),
        _outcome("news", "up_stress", "flat", 0.3, True, day=2),
        _outcome("news", "up_stress", "flat", 0.2, True, day=3),
    ]
    beh = directional_behavior_by_state(outcomes)["up_stress"]["news"]
    assert beh["n"] == 3
    assert beh["long_rate"] == pytest.approx(1 / 3)


def test_correlation_identical_and_opposite_experts():
    outcomes = []
    for d in range(1, 6):
        # 'a' and 'copy' identical signed confidence; 'anti' goes flat when a is long.
        a_long = d % 2 == 0
        outcomes.append(_outcome("a", "up_stress", "long" if a_long else "flat", 0.8, True, day=d))
        outcomes.append(_outcome("copy", "up_stress", "long" if a_long else "flat", 0.8, True, day=d))
        outcomes.append(_outcome("anti", "up_stress", "flat" if a_long else "long", 0.8, True, day=d))
    corr = expert_signed_confidence_correlation(outcomes)
    assert corr["experts"] == ["a", "anti", "copy"]
    assert corr["pairs"]["a|copy"] == pytest.approx(1.0)
    assert corr["pairs"]["a|anti"] == pytest.approx(-1.0)


def test_view_outcomes_from_episodes_reads_expert_errors():
    def _err(expert, correct):
        return ExpertErrorRecord(
            expert_id=expert,
            market_state="up_stress",
            direction="long",
            confidence=0.7,
            weight_used=0.25,
            realized_next_return=0.01 if correct else -0.01,
            direction_correct=correct,
            error_type="long_correct" if correct else "long_wrong",
            error_score=0.0,
            overconfidence_loss=0.0,
            missed_opportunity_loss=0.0,
            data_quality="medium",
        )

    episode = AttributedEpisode(
        decision_date=date(2022, 1, 3),
        market_state="up_stress",
        attribution={c: 0.0 for c in ATTRIBUTION_COMPONENTS},
        signal_attribution={},
        expert_errors=(_err("technical", True), _err("news", False)),
        action="long",
        ticker="AAA",
    )
    outcomes = view_outcomes_from_episodes([episode])
    assert {o.expert_id for o in outcomes} == {"technical", "news"}
    assert all(o.decision_date == "2022-01-03" and o.ticker == "AAA" for o in outcomes)


def test_buy_hold_sell_summary_maps_target_transitions():
    from datetime import date as _date

    from trading_agent.experiments.expert_reliability_diagnostic import buy_hold_sell_summary
    from trading_agent.memory.expert_reflection import ATTRIBUTION_COMPONENTS, AttributedEpisode

    def _ep(day, action, ticker="AAA"):
        return AttributedEpisode(
            decision_date=_date(2022, 1, day),
            market_state="up_stress",
            attribution={c: 0.0 for c in ATTRIBUTION_COMPONENTS},
            signal_attribution={},
            action=action,
            ticker=ticker,
        )

    # flat, long (BUY), long (HOLD), flat (SELL), flat (HOLD)
    episodes = [
        _ep(3, "flat"), _ep(4, "long"), _ep(5, "long"), _ep(6, "flat"), _ep(7, "flat"),
    ]
    s = buy_hold_sell_summary(episodes)
    assert s["buy"] == 1
    assert s["sell"] == 1
    assert s["hold"] == 3
    assert s["n"] == 5
    assert s["hold_rate"] == pytest.approx(0.6)


def test_expert_reliability_report_over_real_loop(regime_dataset, mem_backtest_config):
    from trading_agent.alpha.market_state_classifier import MarketStateConfig
    from trading_agent.experts import FactorExpert, FundamentalExpert, NewsExpert, TechnicalExpert
    from trading_agent.experts.backends import HeuristicExpertBackend
    from trading_agent.runner import run_expert_loop

    experts = [
        TechnicalExpert(HeuristicExpertBackend()),
        FactorExpert(HeuristicExpertBackend()),
        NewsExpert(HeuristicExpertBackend()),
        FundamentalExpert(HeuristicExpertBackend()),
    ]
    result = run_expert_loop(
        regime_dataset, mem_backtest_config, experts,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        allow_reflection=False,
    )
    report = expert_reliability_report(result.episodes)
    assert report["n_view_outcomes"] > 0
    assert report["calibration_by_state"]
    assert set(report["expert_correlation"]["experts"]) <= {
        "technical", "factor", "news", "fundamental",
    }
    # The raw-view direction summary runs over the same loop output.
    directions = view_direction_summary(result.expert_view_records)
    assert directions
    for summary in directions.values():
        assert summary["all_states"]["n"] > 0


def test_view_direction_summary_counts_neutral_and_excludes_safe_defaults():
    records = {
        "news": [
            {"market_state": "up_calm", "direction": "long", "safe_default": False},
            {"market_state": "up_calm", "direction": "neutral", "safe_default": False},
            {"market_state": "up_calm", "direction": "neutral", "safe_default": False},
            # Safe default carries a placeholder "flat" direction: it is a
            # failure, not a judgment, and must NOT count as a flat vote.
            {"market_state": "up_calm", "direction": "flat", "safe_default": True},
            {"market_state": "down_stress", "direction": "flat", "safe_default": False},
        ],
    }
    out = view_direction_summary(records)

    cell = out["news"]["by_state"]["up_calm"]
    assert (cell["long"], cell["neutral"], cell["flat"]) == (1, 2, 0)
    assert cell["safe_default"] == 1 and cell["n"] == 4
    assert cell["neutral_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert cell["spoken_rate"] == pytest.approx(1 / 4)

    total = out["news"]["all_states"]
    assert total["n"] == 5 and total["flat"] == 1 and total["safe_default"] == 1
    assert total["neutral_rate"] == pytest.approx(0.5)   # 2 of 4 judgments
    assert total["spoken_rate"] == pytest.approx(2 / 5)  # (long+flat) / all views
