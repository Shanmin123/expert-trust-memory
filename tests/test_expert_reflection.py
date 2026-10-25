"""Tests for attribution-fed expert reflection."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from trading_agent.memory.expert_error import ExpertErrorRecord
from trading_agent.memory.expert_reflection import (
    ATTRIBUTION_COMPONENTS,
    AttributedEpisode,
    DeterministicExpertReflection,
    LLMExpertReflection,
)
from trading_agent.memory.expert_weights import ExpertWeightEntry, WEIGHT_DELTA_MAX, WEIGHT_DELTA_MIN

CREATED = date(2024, 6, 1)


def _err(expert_id, market_state, score, day=1):
    direction_correct = score > 0
    return ExpertErrorRecord(
        expert_id=expert_id,
        market_state=market_state,
        direction="long" if score < 0 else "flat",
        confidence=1.0,
        weight_used=0.25,
        realized_next_return=score,
        direction_correct=direction_correct,
        error_type="flat_defensive_correct" if direction_correct else "long_wrong",
        error_score=score,
        overconfidence_loss=abs(score) if score < 0 else 0.0,
        missed_opportunity_loss=0.0,
        data_quality="medium",
        risk_flags=(),
        evidence=(f"day {day}",),
    )


def _episode(market_state, signal_attr, day=1, expert_errors=()):
    return AttributedEpisode(
        decision_date=date(2024, 5, day),
        market_state=market_state,
        attribution={c: 0.0 for c in ATTRIBUTION_COMPONENTS},
        signal_attribution=signal_attr,
        expert_errors=tuple(expert_errors),
        action="long",
    )


def test_attributed_episode_rejects_unknown_market_state():
    with pytest.raises(ValueError):
        _episode("bull", {"technical": 0.1})


def test_attributed_episode_requires_all_components():
    with pytest.raises(ValueError):
        AttributedEpisode(
            decision_date=CREATED,
            market_state="up_stress",
            attribution={"signal": 0.1},
            signal_attribution={"technical": 0.1},
        )


def test_deterministic_shifts_toward_more_reliable_expert():
    # Contribution scorecard: technical is above the state mean, factor below. The symmetric
    # rule up-weights technical and down-weights factor.
    episodes = [
        _episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d)
        for d in range(1, 6)
    ]
    backend = DeterministicExpertReflection(min_support=3)
    out = backend.reflect(episodes, CREATED, participation_rate=0.8)
    by_expert = {e.expert_id: e for e in out}
    assert by_expert["technical"].weight_delta > 0
    assert by_expert["fundamental"].weight_delta < 0
    assert 0.0 < by_expert["technical"].weight_delta <= WEIGHT_DELTA_MAX
    assert by_expert["technical"].status == "learned"


def test_deterministic_shifts_away_from_the_worse_expert():
    # Reads the contribution scorecard (g = v*r), not confidence-based error records.
    episodes = [
        _episode("down_stress", {"technical": 0.02, "news": -0.04}, day=d)
        for d in range(1, 6)
    ]
    out = DeterministicExpertReflection(min_support=5, step_scale=2.0).reflect(
        episodes, CREATED, participation_rate=0.8
    )
    by_expert = {entry.expert_id: entry for entry in out}
    assert by_expert["news"].weight_delta < 0            # below state mean -> down-weight
    assert by_expert["news"].weight_delta >= WEIGHT_DELTA_MIN
    assert by_expert["technical"].weight_delta > 0       # above state mean -> up-weight


def test_deterministic_reflection_is_independent_of_participation_rate():
    # GRID-INVARIANCE GUARD: participation depends on the realised decisions (hence on the
    # tuned temperature and threshold), so reflection must not read it, or the trained memory
    # would differ across the tuning grid.
    episodes = [
        _episode("down_stress", {"technical": 0.02, "news": -0.04}, day=d)
        for d in range(1, 6)
    ]
    backend = DeterministicExpertReflection(min_support=5, step_scale=2.0)

    low = backend.reflect(episodes, CREATED, participation_rate=0.1)
    high = backend.reflect(episodes, CREATED, participation_rate=0.8)

    key = lambda out: sorted((e.expert_id, e.market_state, round(e.weight_delta, 9)) for e in out)
    assert key(low) == key(high)
    assert any(e.weight_delta < 0 for e in low)   # the below-mean expert is down-weighted


def test_min_support_counts_expert_state_non_default_evidence_not_episode_count():
    episodes = [
        _episode(
            "down_stress",
            {"news": -0.05},
            day=d,
            expert_errors=(_err("news", "down_stress", -0.05, d),) if d <= 2 else (),
        )
        for d in range(1, 8)
    ]

    out = DeterministicExpertReflection(min_support=5, step_scale=2.0).reflect(
        episodes, CREATED, participation_rate=0.8
    )

    assert out == []


def test_deterministic_delta_is_clipped_to_bound():
    episodes = [
        _episode("down_stress", {"technical": 1.0, "news": -1.0}, day=d) for d in range(1, 6)
    ]
    out = DeterministicExpertReflection(min_support=3, step_scale=10.0).reflect(
        episodes, CREATED, participation_rate=0.8
    )
    by_expert = {e.expert_id: e for e in out}
    assert by_expert["technical"].weight_delta == pytest.approx(WEIGHT_DELTA_MAX)
    assert by_expert["news"].weight_delta == pytest.approx(WEIGHT_DELTA_MIN)


def test_deterministic_emits_nothing_below_support():
    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in (1, 2)]
    out = DeterministicExpertReflection(min_support=3).reflect(episodes, CREATED, 0.8)
    assert out == []


def test_deterministic_emits_nothing_when_no_clear_leader():
    episodes = [_episode("up_stress", {"technical": 0.03, "fundamental": 0.03}, day=d) for d in range(1, 6)]
    out = DeterministicExpertReflection(min_support=3).reflect(episodes, CREATED, 0.8)
    assert out == []


def test_deterministic_tilts_relative_to_state_mean_even_when_all_negative():
    # In an all-losing state the least-bad expert is still up-weighted relative to the worst.
    episodes = [_episode("up_stress", {"technical": -0.01, "fundamental": -0.05}, day=d) for d in range(1, 6)]
    out = DeterministicExpertReflection(min_support=3).reflect(episodes, CREATED, 0.8)
    by_expert = {e.expert_id: e for e in out}
    assert by_expert["technical"].weight_delta > 0
    assert by_expert["fundamental"].weight_delta < 0


def test_deterministic_handles_multiple_states():
    episodes = (
        [_episode("up_calm", {"fundamental": 0.06, "technical": -0.01}, day=d) for d in range(1, 6)]
        + [_episode("down_stress", {"news": 0.07, "technical": -0.02}, day=d) for d in range(1, 6)]
    )
    out = DeterministicExpertReflection(min_support=3).reflect(episodes, CREATED, 0.8)
    positive = {p.market_state: p.expert_id for p in out if p.weight_delta > 0}
    assert positive == {"up_calm": "fundamental", "down_stress": "news"}


def test_trial_log_uses_proposal_fields():
    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in range(1, 6)]
    backend = DeterministicExpertReflection(min_support=3)
    backend.reflect(episodes, CREATED, 0.8)
    assert backend.log.trials == 2          # technical up, factor down
    assert backend.log.proposed == 2
    ids = [r["proposal_id"] for r in backend.log.records]
    assert any(i.startswith("det::technical::up_stress") for i in ids)


def _llm(response: str, raises: bool = False):
    def backend(system_prompt: str, user_prompt: str) -> str:
        if raises:
            raise RuntimeError("backend down")
        return response
    return backend


_NO_SLEEP = lambda _s: None   # keep retry-backoff tests instant


def test_llm_parses_structured_json():
    resp = json.dumps({"proposals": [
        {"expert_id": "news", "market_state": "down_calm", "weight_delta": 0.08, "rationale": "x"},
    ]})
    out = LLMExpertReflection(_llm(resp)).reflect([], CREATED, 0.7)
    assert len(out) == 1
    assert out[0].expert_id == "news"
    assert out[0].market_state == "down_calm"
    assert out[0].weight_delta == pytest.approx(0.08)
    assert out[0].status == "learned"


def test_llm_clips_out_of_bound_delta():
    resp = json.dumps({"proposals": [
        {"expert_id": "technical", "market_state": "up_stress", "weight_delta": 0.5},
        {"expert_id": "fundamental", "market_state": "up_stress", "weight_delta": -0.5},
    ]})
    out = LLMExpertReflection(_llm(resp)).reflect([], CREATED, 0.7)
    assert out[0].weight_delta == pytest.approx(WEIGHT_DELTA_MAX)
    assert out[1].weight_delta == pytest.approx(WEIGHT_DELTA_MIN)


def test_llm_malformed_json_falls_back_to_deterministic():
    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in range(1, 6)]
    refl = LLMExpertReflection(_llm("not json"), sleeper=_NO_SLEEP)
    out = refl.reflect(episodes, CREATED, 0.8)
    assert refl.fallback_count == 1
    assert {e.expert_id for e in out} == {"technical", "fundamental"}   # symmetric deterministic rule


def test_llm_backend_exception_falls_back():
    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in range(1, 6)]
    refl = LLMExpertReflection(_llm("", raises=True), sleeper=_NO_SLEEP)
    out = refl.reflect(episodes, CREATED, 0.8)
    assert refl.fallback_count == 1
    assert len(out) == 2


def test_llm_unknown_expert_falls_back():
    resp = json.dumps({"proposals": [
        {"expert_id": "sentiment", "market_state": "up_stress", "weight_delta": 0.05},
    ]})
    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in range(1, 6)]
    refl = LLMExpertReflection(_llm(resp), sleeper=_NO_SLEEP)
    out = refl.reflect(episodes, CREATED, 0.8)
    assert refl.fallback_count == 1
    assert len(out) == 2


def test_llm_retries_transient_failure_before_falling_back():
    # A rate-limit blip on the first call should be retried, not silently degraded
    # to the deterministic rule (which corrupts the reflection comparison).
    calls = {"n": 0}
    good = json.dumps({"proposals": [
        {"expert_id": "news", "market_state": "down_calm", "weight_delta": 0.05},
    ]})

    def flaky(system_prompt, user_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limit")
        return good

    refl = LLMExpertReflection(flaky, max_retries=2, sleeper=_NO_SLEEP)
    out = refl.reflect([], CREATED, 0.7)
    assert calls["n"] == 2                 # retried once, then succeeded
    assert refl.retry_count == 1
    assert refl.fallback_count == 0        # NO fallback: the LLM proposal is used
    assert out[0].expert_id == "news"


def test_llm_falls_back_only_after_exhausting_retries():
    calls = {"n": 0}

    def always_down(system_prompt, user_prompt):
        calls["n"] += 1
        raise RuntimeError("persistent outage")

    episodes = [_episode("up_stress", {"technical": 0.05, "fundamental": -0.02}, day=d) for d in range(1, 6)]
    refl = LLMExpertReflection(always_down, max_retries=2, sleeper=_NO_SLEEP)
    out = refl.reflect(episodes, CREATED, 0.8)
    assert calls["n"] == 3                 # initial + 2 retries
    assert refl.retry_count == 2
    assert refl.fallback_count == 1        # only then the deterministic fallback
    assert len(out) == 2


def test_llm_prompt_excludes_participation_for_grid_invariance():
    # The reflection prompt must not carry the participation rate (a tuned-parameter-
    # dependent, portfolio-level quantity), so the LLM's delta is invariant across the
    # tuning grid and the trained memory can be replayed without retraining.
    captured = {}

    def backend(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        return json.dumps({"proposals": []})

    LLMExpertReflection(backend).reflect([], CREATED, 0.15)
    assert "0.15" not in captured["system"]
    assert "participation" not in captured["system"].lower()


def test_llm_prompt_attributed_only_with_deterministic_lessons_payload():
    """The system prompt is attribution-only (3 experts, no dead lessons instruction —
    lessons are computed deterministically and ride in the user payload)."""
    captured = {}
    episodes = [
        _episode("up_stress", {"technical": 0.04, "fundamental": -0.01}, day=d)
        for d in range(1, 6)
    ]

    def backend(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return json.dumps({"proposals": []})

    LLMExpertReflection(backend).reflect(episodes, CREATED, 0.7)

    assert "three experts" in captured["system"]
    assert "compact lessons" not in captured["system"]
    assert "counterfactual" not in captured["system"].lower()
    assert "never raw portfolio PnL" in captured["system"]
    assert "confidence" in captured["system"].lower()   # prompt states confidence is NOT used
    payload = json.loads(captured["user"])
    assert "scorecard_by_state" in payload
    assert "expert_error_summary_by_state" not in payload   # confidence-based channel removed
    assert "recent_lessons_by_state" in payload
    assert payload["recent_lessons_by_state"]["up_stress"][0].startswith("up_stress:")
    # scorecard carries mean contribution + hit rate + count, no confidence.
    tech = payload["scorecard_by_state"]["up_stress"]["technical"]
    assert set(tech) == {"mean_contribution", "hit_rate", "count"}


def test_reflection_source_never_references_raw_pnl():
    import trading_agent.memory.expert_reflection as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "raw_pnl" not in src
    assert "raw_return" not in src


def test_attribution_components_are_the_five_way_decomposition():
    assert ATTRIBUTION_COMPONENTS == ("market", "style", "signal", "sizing", "execution")


def test_scale_free_deterministic_rule_saturates_the_bound():
    # Fixed-mode deltas on daily-return scales are ~1e-2 (functionally a no-op);
    # scale_free rescales by the largest deviation so the best/worst experts get
    # near-bound deltas, commensurate with what the LLM emits.
    episodes = [
        _episode("up_stress", {"technical": 0.004, "fundamental": -0.004, "news": 0.001}, day=d)
        for d in range(1, 6)
    ]
    fixed = DeterministicExpertReflection(min_support=3, step_mode="fixed")
    scaled = DeterministicExpertReflection(min_support=3, step_mode="scale_free")
    fixed_by = {e.expert_id: e.weight_delta for e in fixed.reflect(episodes, CREATED, 0.8)}
    scaled_by = {e.expert_id: e.weight_delta for e in scaled.reflect(episodes, CREATED, 0.8)}
    assert abs(fixed_by["technical"]) < 0.02                      # no-op scale
    # The largest deviator saturates the bound; others scale proportionally.
    assert scaled_by["fundamental"] == pytest.approx(WEIGHT_DELTA_MIN)
    assert scaled_by["technical"] > 0.05
    assert abs(scaled_by["news"]) < 0.05                          # interior expert stays interior


def test_scale_free_rule_rejects_unknown_mode():
    with pytest.raises(ValueError):
        DeterministicExpertReflection(step_mode="bogus")
