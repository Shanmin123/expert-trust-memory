"""Expert-level error analysis for reflection.

The functions here turn audit-only ExpertView fields plus next-bar realized
returns into structured, point-in-time error records. Reflection can learn from
these records, but they never become direct trading authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from trading_agent.contracts import ExpertView
from trading_agent.experts.base_expert import is_safe_default_view


@dataclass(frozen=True)
class ExpertErrorRecord:
    expert_id: str
    market_state: str
    direction: str
    confidence: float
    weight_used: float
    realized_next_return: float
    direction_correct: bool
    error_type: str
    error_score: float
    overconfidence_loss: float
    missed_opportunity_loss: float
    data_quality: str
    risk_flags: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpertErrorSummary:
    expert_id: str
    market_state: str
    n: int
    accuracy: float
    mean_error_score: float
    mean_confidence: float
    overconfidence_loss: float
    missed_opportunity_loss: float
    data_quality_counts: Mapping[str, int]
    top_risk_flags: tuple[str, ...] = ()


def _record_for_view(
    market_state: str,
    view: ExpertView,
    weight_used: float,
    realized_next_return: float,
) -> ExpertErrorRecord:
    confidence = max(0.0, min(1.0, float(view.confidence)))
    realized = float(realized_next_return)
    magnitude = abs(realized)
    if view.direction == "long" and realized > 0:
        direction_correct = True
        error_type = "long_correct"
        error_score = confidence * realized
        overconfidence_loss = 0.0
        missed_opportunity_loss = 0.0
    elif view.direction == "long":
        direction_correct = False
        error_type = "long_wrong"
        error_score = -confidence * magnitude
        overconfidence_loss = confidence * magnitude
        missed_opportunity_loss = 0.0
    elif realized <= 0:
        direction_correct = True
        error_type = "flat_defensive_correct"
        error_score = confidence * magnitude
        overconfidence_loss = 0.0
        missed_opportunity_loss = 0.0
    else:
        direction_correct = False
        error_type = "flat_missed_opportunity"
        error_score = -confidence * realized
        overconfidence_loss = 0.0
        missed_opportunity_loss = confidence * realized

    return ExpertErrorRecord(
        expert_id=view.expert_id,
        market_state=market_state,
        direction=view.direction,
        confidence=confidence,
        weight_used=float(weight_used),
        realized_next_return=realized,
        direction_correct=direction_correct,
        error_type=error_type,
        error_score=error_score,
        overconfidence_loss=overconfidence_loss,
        missed_opportunity_loss=missed_opportunity_loss,
        data_quality=view.data_quality,
        risk_flags=tuple(view.risk_flags),
        evidence=tuple(view.evidence),
    )


def build_expert_error_records(
    *,
    market_state: str,
    views: Mapping[str, ExpertView],
    weights: Mapping[str, float],
    realized_next_return: float,
) -> list[ExpertErrorRecord]:
    return [
        _record_for_view(
            market_state=market_state,
            view=view,
            weight_used=float(weights.get(expert_id, 0.0)),
            realized_next_return=realized_next_return,
        )
        for expert_id, view in views.items()
        # Safe defaults are fallbacks, deliberate neutrals are abstentions (design D3):
        # neither is a directional judgment, so neither belongs in the error records.
        if not is_safe_default_view(view) and view.direction != "neutral"
    ]


def summarize_expert_errors(
    records: Sequence[ExpertErrorRecord],
) -> dict[str, dict[str, ExpertErrorSummary]]:
    grouped: dict[tuple[str, str], list[ExpertErrorRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.market_state, record.expert_id)].append(record)

    out: dict[str, dict[str, ExpertErrorSummary]] = defaultdict(dict)
    for (market_state, expert_id), vals in sorted(grouped.items()):
        n = len(vals)
        if n == 0:
            continue
        risk_counts = Counter(flag for val in vals for flag in val.risk_flags)
        data_quality_counts = Counter(val.data_quality for val in vals)
        out[market_state][expert_id] = ExpertErrorSummary(
            expert_id=expert_id,
            market_state=market_state,
            n=n,
            accuracy=sum(1 for val in vals if val.direction_correct) / n,
            mean_error_score=fmean(val.error_score for val in vals),
            mean_confidence=fmean(val.confidence for val in vals),
            overconfidence_loss=fmean(val.overconfidence_loss for val in vals),
            missed_opportunity_loss=fmean(val.missed_opportunity_loss for val in vals),
            data_quality_counts=dict(data_quality_counts),
            top_risk_flags=tuple(flag for flag, _ in risk_counts.most_common(3)),
        )
    return {state: dict(experts) for state, experts in out.items()}


def expert_error_summary_to_dict(
    summary: Mapping[str, Mapping[str, ExpertErrorSummary]],
) -> dict[str, dict[str, dict]]:
    return {
        state: {
            expert_id: {
                "n": item.n,
                "accuracy": round(item.accuracy, 6),
                "mean_error_score": round(item.mean_error_score, 6),
                "mean_confidence": round(item.mean_confidence, 6),
                "overconfidence_loss": round(item.overconfidence_loss, 6),
                "missed_opportunity_loss": round(item.missed_opportunity_loss, 6),
                "data_quality_counts": dict(item.data_quality_counts),
                "top_risk_flags": list(item.top_risk_flags),
            }
            for expert_id, item in experts.items()
        }
        for state, experts in summary.items()
    }
