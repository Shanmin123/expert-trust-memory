"""Tests for expert-level error analysis used by reflection."""

from __future__ import annotations

import pytest

from trading_agent.contracts import ExpertView
from trading_agent.memory.expert_error import (
    ExpertErrorRecord,
    build_expert_error_records,
    summarize_expert_errors,
)


def _view(
    expert_id: str,
    direction: str,
    confidence: float,
    *,
    data_quality: str = "medium",
    risk_flags: tuple[str, ...] = (),
) -> ExpertView:
    return ExpertView(
        expert_id=expert_id,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="test",
        evidence=("test evidence",),
        risk_flags=risk_flags,
        data_quality=data_quality,  # type: ignore[arg-type]
    )


def test_long_correct_has_positive_error_score():
    records = build_expert_error_records(
        market_state="up_stress",
        views={"technical": _view("technical", "long", 0.8)},
        weights={"technical": 0.25},
        realized_next_return=0.02,
    )

    rec = records[0]
    assert rec.direction_correct is True
    assert rec.error_type == "long_correct"
    assert rec.error_score == pytest.approx(0.016)
    assert rec.overconfidence_loss == 0.0
    assert rec.missed_opportunity_loss == 0.0


def test_long_wrong_has_confidence_weighted_overconfidence_loss():
    records = build_expert_error_records(
        market_state="up_stress",
        views={"news": _view("news", "long", 0.9, risk_flags=("sparse_text",))},
        weights={"news": 0.25},
        realized_next_return=-0.03,
    )

    rec = records[0]
    assert rec.direction_correct is False
    assert rec.error_type == "long_wrong"
    assert rec.error_score == pytest.approx(-0.027)
    assert rec.overconfidence_loss == pytest.approx(0.027)
    assert rec.risk_flags == ("sparse_text",)


def test_flat_defensive_correct_is_rewarded():
    records = build_expert_error_records(
        market_state="down_stress",
        views={"fundamental": _view("fundamental", "flat", 0.7)},
        weights={"fundamental": 0.25},
        realized_next_return=-0.04,
    )

    rec = records[0]
    assert rec.direction_correct is True
    assert rec.error_type == "flat_defensive_correct"
    assert rec.error_score == pytest.approx(0.028)


def test_flat_missed_opportunity_is_penalized_by_confidence():
    records = build_expert_error_records(
        market_state="up_calm",
        views={"factor": _view("factor", "flat", 0.6)},
        weights={"factor": 0.25},
        realized_next_return=0.05,
    )

    rec = records[0]
    assert rec.direction_correct is False
    assert rec.error_type == "flat_missed_opportunity"
    assert rec.error_score == pytest.approx(-0.03)
    assert rec.missed_opportunity_loss == pytest.approx(0.03)


def test_confidence_increases_error_penalty():
    low, high = build_expert_error_records(
        market_state="up_stress",
        views={
            "technical": _view("technical", "long", 0.2),
            "news": _view("news", "long", 0.9),
        },
        weights={"technical": 0.25, "news": 0.25},
        realized_next_return=-0.02,
    )

    assert abs(high.error_score) > abs(low.error_score)
    assert high.overconfidence_loss > low.overconfidence_loss


def test_summary_groups_by_market_state_and_expert_without_trading_authority():
    records = [
        ExpertErrorRecord(
            expert_id="news",
            market_state="down_stress",
            direction="long",
            confidence=0.9,
            weight_used=0.25,
            realized_next_return=-0.02,
            direction_correct=False,
            error_type="long_wrong",
            error_score=-0.018,
            overconfidence_loss=0.018,
            missed_opportunity_loss=0.0,
            data_quality="low",
            risk_flags=("sparse_text",),
            evidence=("low coverage",),
        ),
        ExpertErrorRecord(
            expert_id="news",
            market_state="down_stress",
            direction="flat",
            confidence=0.5,
            weight_used=0.25,
            realized_next_return=-0.01,
            direction_correct=True,
            error_type="flat_defensive_correct",
            error_score=0.005,
            overconfidence_loss=0.0,
            missed_opportunity_loss=0.0,
            data_quality="low",
            risk_flags=("sparse_text",),
            evidence=("low coverage",),
        ),
    ]

    summary = summarize_expert_errors(records)

    news = summary["down_stress"]["news"]
    assert news.n == 2
    assert news.accuracy == pytest.approx(0.5)
    assert news.mean_error_score == pytest.approx((-0.018 + 0.005) / 2)
    assert news.data_quality_counts == {"low": 2}
    assert news.top_risk_flags == ("sparse_text",)
