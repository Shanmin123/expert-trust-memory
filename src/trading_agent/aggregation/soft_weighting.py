"""Soft-weighted aggregation of the expert views into a position decision.

Pipeline role:
  memory.get_weights(market_state) -> weights
  aggregate(views, weights)        -> AggregationResult
  runner reads result.direction and passes it to the risk engine.

Weight rules (delegated to ``project_to_floored_simplex``, the single source of
truth for learned expert weights):
  1. Any expert not present in the input dict receives weight 0.
  2. Weights are projected onto the floored simplex: every weight >= WEIGHT_FLOOR
     (0.05) and the four sum to exactly 1. This is an iterative clamp-renorm, not a
     single pass, so the floor genuinely holds after normalisation.
  3. If all input weights were 0 (empty dict), the budget splits equally (0.25 each).

Score and decision (Chapter 3, Eq. aggregate; design D3 ternary vote):
  signed_view(view) = +1.0  if view.direction == "long"
                     = -1.0  if view.direction == "flat"    (bearish vote)
  "neutral" ABSTAINS: a deliberate no-edge judgment leaves the vote entirely
  (sleeping-experts semantics), exactly like missing views and safe defaults.
  score = sum_{e in SPEAKERS} ( w'[e] * signed_view(views[e]) )   in [-1, 1]
  direction: STRICT boundaries (frozen a priori 2026-07-17, before any D3 data;
  round-2 review concurrence). State changes require strict net evidence:
    enter long  iff score >  threshold
    exit long   iff score <  exit_threshold   (when HOLD semantics are on)
    score exactly AT a boundary MAINTAINS the current position.
  D3 creates an atom at exactly 0 (one neutral + a 1-1 split renormalises to
  score 0); under the old non-strict rule its meaning flipped with the tuned
  threshold (entered at theta=0 via the entry branch, exited at theta=0.25).
  A zero net spoken edge is evidence for nothing, so it changes nothing --
  the smooth extension of the no-speaker rule below.
where SPEAKERS excludes missing views, safe-default fallbacks AND deliberate
neutrals, and w' renormalises the floored weights over the speakers
(w'[e] = w[e] / sum speakers). With equal weights this reproduces proportional
vote counting -- an abstaining expert neither supports nor opposes the trade.
If NO expert speaks: `aggregate` stays flat (no prior-position context), while
`aggregate_by_ticker` MAINTAINS the previous position (silence is not a sell
signal; entries and exits both require at least one spoken vote).
The aggregation reads only the direction of each view, as a +/-1 vote; the
stated confidence is not used (it is audit/reporting context only).
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Mapping

from trading_agent.contracts import (
    ExpertDirection,
    ExpertView,
    is_abstention_view,
    is_safe_default_view,
)
from trading_agent.aggregation.weights import WEIGHT_FLOOR, project_to_floored_simplex

# Design D (frozen 2026-07-17, architecture.md §0): TradingAgents-aligned THREE experts.
# technical = deterministic TradingView rating; news = TA news-analyst style (text + FRED
# macro); fundamental = TA fundamentals-analyst style (SEC statements + filing text).
# factor was dropped (cross-sectional mismatch; z-score prompting unsound).
EXPERT_IDS: tuple[str, ...] = ("technical", "news", "fundamental")
_N_EXPERTS: int = len(EXPERT_IDS)
_EQUAL_WEIGHT: float = 1.0 / _N_EXPERTS


@dataclasses.dataclass(frozen=True)
class AggregationResult:
    """Output of ``aggregate``: the combined position decision for one rebalance date."""

    direction: ExpertDirection
    score: float                    # weighted signed-vote sum in [-1, 1]
    weights_used: Mapping[str, float]  # floored + normalised weights actually applied


@dataclasses.dataclass(frozen=True)
class TickerDecision:
    """One ticker's weighted expert decision."""

    direction: ExpertDirection
    score: float


@dataclasses.dataclass(frozen=True)
class PerTickerAggregationResult:
    """Per-name aggregation output for one decision date."""

    by_ticker: Mapping[str, TickerDecision]
    weights_used: Mapping[str, float]


def normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Project the expert weights onto the floored simplex.

    Missing experts are treated as weight 0. The result has every weight
    >= WEIGHT_FLOOR and sums to 1. An empty input dict produces equal weights
    (0.25 each). Delegates to ``project_to_floored_simplex`` so all callers use
    exactly the same floor/normalisation rule.
    """
    raw = {eid: float(weights.get(eid, 0.0)) for eid in EXPERT_IDS}
    return project_to_floored_simplex(raw, WEIGHT_FLOOR)


def _available_score(
    views_by_eid: Mapping[str, ExpertView | None], norm: Mapping[str, float],
    *, confidence_weighted: bool = False, soft_abstain_weight: float = 0.0,
) -> float | None:
    """Weighted signed-vote score over AVAILABLE experts, weights renormalised.

    Missing views and safe-default fallbacks are not judgments: they drop out
    and the remaining experts' weights are renormalised to sum to 1, so an
    expert without data neither supports nor opposes the trade. Returns None
    when NO expert is available (no evidence: the decision must be flat
    regardless of the threshold).

    With ``confidence_weighted`` each vote is scaled by the expert's stated
    confidence in [0,1], so a high-confidence minority can outweigh a
    low-confidence majority. This changes only the committee DECISION, never the
    attribution (which reads each expert's own +/-1 direction), so the learned
    memory is unaffected. It is an opt-in experiment on the otherwise-discarded
    confidence channel, not the sealed default.
    """
    available = {
        eid: view
        for eid, view in views_by_eid.items()
        if view is not None and not is_abstention_view(view)
    }
    if not available:
        # Soft abstention never conjures a vote out of silence: with NO full speaker
        # the date stays a no-evidence date (silence maintains), regardless of any
        # forced calls. This is what separates it from the forced-binary replay.
        return None
    # Soft abstention (opt-in): a deliberate NEUTRAL that carries a forced_direction is
    # a judgment ("no edge, but if forced: X"); it may TILT a spoken committee at
    # ``soft_abstain_weight`` of a vote. Safe defaults are not judgments and never enter.
    soft: dict[str, ExpertView] = {}
    if soft_abstain_weight > 0.0:
        soft = {
            eid: view
            for eid, view in views_by_eid.items()
            if view is not None and view.direction == "neutral"
            and not is_safe_default_view(view)
            and view.forced_direction in ("long", "flat")
        }
    weight_budget = (
        sum(norm[eid] for eid in available)
        + soft_abstain_weight * sum(norm[eid] for eid in soft)
    )
    if weight_budget <= 0:
        return None
    score = 0.0
    for eid, view in available.items():
        signed_view = 1.0 if view.direction == "long" else -1.0
        if confidence_weighted:
            signed_view *= max(0.0, min(1.0, float(view.confidence)))
        score += (norm[eid] / weight_budget) * signed_view
    for eid, view in soft.items():
        forced = 1.0 if view.forced_direction == "long" else -1.0
        score += (soft_abstain_weight * norm[eid] / weight_budget) * forced
    return score


def aggregate(
    views: Mapping[str, ExpertView],
    weights: Mapping[str, float],
    threshold: float = 0.0,
) -> AggregationResult:
    """Combine the available ExpertViews into a single position decision.

    Args:
        views:     {expert_id: ExpertView}. Missing or safe-default experts drop
                   out and the remaining weights are renormalised.
        weights:   {expert_id: float}. Floored to WEIGHT_FLOOR and renormalised.
        threshold: score must be STRICTLY above this to go long (default 0.5).
                   This stateless form has no position to maintain, so a score
                   exactly at the threshold stays flat.

    Returns:
        AggregationResult with direction, score, and the normalised weights used.
    """
    norm = normalise_weights(weights)
    maybe_score = _available_score({eid: views.get(eid) for eid in EXPERT_IDS}, norm)
    if maybe_score is None:
        return AggregationResult(direction="flat", score=0.0, weights_used=norm)
    direction: ExpertDirection = "long" if maybe_score > threshold else "flat"
    return AggregationResult(direction=direction, score=maybe_score, weights_used=norm)


def aggregate_by_ticker(
    views_by_expert: Mapping[str, Mapping[str, ExpertView]],
    weights: Mapping[str, float],
    *,
    tickers: list[str] | tuple[str, ...],
    threshold: float = 0.0,
    exit_threshold: float | None = 0.0,
    prev_directions: Mapping[str, str] | None = None,
    silence_policy: Literal["carry", "flat"] = "carry",
    confidence_weighted: bool = False,
    soft_abstain_weight: float = 0.0,
) -> PerTickerAggregationResult:
    """Combine the available expert-view maps into independent per-ticker decisions.

    STRICT boundaries with HOLD semantics (frozen a priori 2026-07-17): when
    ``exit_threshold`` is set and the ticker was long on the previous decision
    date, the position is held while the score stays AT OR ABOVE
    ``exit_threshold`` and exits only STRICTLY below it; a flat ticker enters
    only STRICTLY above the entry ``threshold``. A score exactly at a boundary
    therefore maintains the current position -- zero net spoken edge is
    evidence for nothing (D3 makes score==0 a real atom: one neutral plus a
    1-1 split). Enter on strict net support, exit on strict net opposition;
    this mirrors the BUY/HOLD/SELL action space of the reference systems
    instead of re-voting the entry every day. A no-speaker date (every expert
    missing, safe-default or neutral) MAINTAINS the previous position: silence
    is not a sell signal, and entries/exits both require at least one spoken
    vote (design D3). With ``exit_threshold=None`` (the daily re-vote
    ablation) there is no held state: the ticker is long iff the score is
    strictly above ``threshold``. ``silence_policy="flat"`` is the ablation
    replay switch (round-2 review): a no-speaker date goes to cash instead of
    carrying the position, isolating the effect of the carry rule itself.
    """
    norm = normalise_weights(weights)
    prev = prev_directions or {}
    by_ticker: dict[str, TickerDecision] = {}
    for ticker in tickers:
        maybe_score = _available_score(
            {eid: views_by_expert.get(eid, {}).get(ticker) for eid in EXPERT_IDS}, norm,
            confidence_weighted=confidence_weighted,
            soft_abstain_weight=soft_abstain_weight,
        )
        if maybe_score is None:
            if silence_policy == "carry":
                held = "long" if prev.get(ticker) == "long" else "flat"
            else:
                held = "flat"
            by_ticker[ticker] = TickerDecision(direction=held, score=0.0)
            continue
        was_long = prev.get(ticker) == "long"
        if was_long and exit_threshold is not None:
            # STRICT exit: leave only on net evidence against the position.
            direction: ExpertDirection = "long" if maybe_score >= exit_threshold else "flat"
        else:
            # STRICT entry (and the exit_threshold=None daily re-vote ablation).
            direction = "long" if maybe_score > threshold else "flat"
        by_ticker[ticker] = TickerDecision(direction=direction, score=maybe_score)
    return PerTickerAggregationResult(by_ticker=by_ticker, weights_used=norm)
