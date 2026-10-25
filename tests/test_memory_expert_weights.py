"""Tests for learned expert-weight memory."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from trading_agent.aggregation.weights import WEIGHT_FLOOR
from trading_agent.memory.expert_weights import (
    EXPERT_IDS,
    GLOBAL_MARKET_STATE,
    MARKET_STATES,
    WEIGHT_DELTA_MAX,
    WEIGHT_DELTA_MIN,
    ExpertWeightEntry,
    ExpertWeightStore,
    equal_weights,
    get_weights,
)
import trading_agent.memory.expert_weights as expert_weights

# Equal weight under the CURRENT expert set (design D: three experts -> 1/3). Tests assert
# against this constant instead of a hardcoded value so the expert-set size is not baked in.
EQ = 1.0 / len(EXPERT_IDS)


def _entry(expert_id, market_state, delta, *, status="learned", entry_id=None):
    return ExpertWeightEntry(
        entry_id=entry_id or f"{expert_id}-{market_state}-{delta}",
        expert_id=expert_id,
        market_state=market_state,
        weight_delta=delta,
        status=status,
        created_at=date(2024, 1, 1),
        learned_at=date(2024, 1, 1) if status == "learned" else None,
    )


def _assert_valid_weights(w):
    assert set(w.keys()) == set(EXPERT_IDS)
    assert all(v >= WEIGHT_FLOOR - 1e-12 for v in w.values()), w
    assert sum(w.values()) == pytest.approx(1.0)


def test_get_weights_empty_returns_equal_weights():
    for state in MARKET_STATES:
        w = get_weights(state, [])
        _assert_valid_weights(w)
        for v in w.values():
            assert v == pytest.approx(EQ)


def test_get_weights_single_entry_shifts_and_projects():
    w = get_weights("up_stress", [_entry("technical", "up_stress", 0.10)])
    _assert_valid_weights(w)
    assert w["technical"] > EQ
    assert w["news"] < EQ


def test_get_weights_ignores_other_market_state():
    w = get_weights("up_stress", [_entry("technical", "down_stress", 0.10)])
    for v in w.values():
        assert v == pytest.approx(EQ)


def test_get_weights_ignores_deprecated_entries():
    w = get_weights("up_stress", [_entry("technical", "up_stress", 0.10, status="deprecated")])
    for v in w.values():
        assert v == pytest.approx(EQ)


def test_get_weights_applies_global_entries_when_enabled():
    w = get_weights("down_stress", [_entry("news", GLOBAL_MARKET_STATE, 0.10)])
    _assert_valid_weights(w)
    assert w["news"] > EQ


def test_get_weights_can_ignore_global_entries_for_state_only_arm():
    w = get_weights(
        "down_stress",
        [_entry("news", GLOBAL_MARKET_STATE, 0.10)],
        include_global=False,
    )
    for v in w.values():
        assert v == pytest.approx(EQ)


def test_get_weights_accumulates_and_opposing_deltas_can_cancel():
    entries = [
        _entry("technical", "up_stress", 0.10, entry_id="up"),
        _entry("technical", "up_stress", -0.10, entry_id="down"),
    ]
    w = get_weights("up_stress", entries)
    for v in w.values():
        assert v == pytest.approx(EQ)


def test_get_weights_clips_cumulative_delta_per_expert_state_key():
    # Three max-bound entries for the same (expert, state) key must not exceed
    # the effect of a single max-bound entry: the summed delta is clipped to
    # the documented [-0.10, 0.10] bound before projection.
    stacked = [
        _entry("technical", "up_stress", 0.10, entry_id=f"up-{i}") for i in range(3)
    ]
    single = [_entry("technical", "up_stress", 0.10, entry_id="up-once")]
    w_stacked = get_weights("up_stress", stacked)
    w_single = get_weights("up_stress", single)
    _assert_valid_weights(w_stacked)
    for expert_id in EXPERT_IDS:
        assert w_stacked[expert_id] == pytest.approx(w_single[expert_id])


def test_get_weights_unbounded_toggle_lets_cumulative_delta_exceed_bound():
    # Three max-bound entries for one key. Bounded caps the sum at 0.10; the E3
    # unbounded ablation lets the effective adjustment stack past the bound.
    stacked = [
        _entry("technical", "up_stress", 0.10, entry_id=f"up-{i}") for i in range(3)
    ]
    bounded = get_weights("up_stress", stacked, clip_cumulative=True)
    unbounded = get_weights("up_stress", stacked, clip_cumulative=False)
    _assert_valid_weights(bounded)
    _assert_valid_weights(unbounded)
    # Unbounded pushes technical strictly higher (0.30 pre-projection vs 0.10).
    assert unbounded["technical"] > bounded["technical"]


def test_store_bounded_and_unbounded_reads_do_not_cross_contaminate():
    store = ExpertWeightStore()
    for i in range(3):
        store.add_learned_adjustment(_entry("technical", "up_stress", 0.10, entry_id=f"up-{i}"))
    bounded = store.get_weights("up_stress", clip_cumulative=True)
    unbounded = store.get_weights("up_stress", clip_cumulative=False)
    # Cache is keyed by the effective policy, so neither read poisons the other.
    assert store.get_weights("up_stress", clip_cumulative=True) == bounded
    assert store.get_weights("up_stress", clip_cumulative=False) == unbounded
    assert unbounded["technical"] > bounded["technical"]


def test_store_construction_sets_default_bounding_policy():
    bounded_store = ExpertWeightStore(clip_cumulative=True)
    unbounded_store = ExpertWeightStore(clip_cumulative=False)
    for i in range(3):
        e = _entry("technical", "up_stress", 0.10, entry_id=f"up-{i}")
        bounded_store.add_learned_adjustment(e)
        unbounded_store.add_learned_adjustment(
            _entry("technical", "up_stress", 0.10, entry_id=f"up-{i}")
        )
    # Each store's own get_weights uses the policy set at construction.
    assert unbounded_store.get_weights("up_stress")["technical"] > (
        bounded_store.get_weights("up_stress")["technical"]
    )


def test_get_weights_clips_state_and_global_deltas_independently():
    entries = [
        _entry("technical", "up_stress", 0.10, entry_id="state-1"),
        _entry("technical", "up_stress", 0.10, entry_id="state-2"),
        _entry("technical", GLOBAL_MARKET_STATE, 0.10, entry_id="global-1"),
        _entry("technical", GLOBAL_MARKET_STATE, 0.10, entry_id="global-2"),
    ]
    w = get_weights("up_stress", entries)
    expected = get_weights("up_stress", [
        _entry("technical", "up_stress", 0.10, entry_id="state-once"),
        _entry("technical", GLOBAL_MARKET_STATE, 0.10, entry_id="global-once"),
    ])
    _assert_valid_weights(w)
    for expert_id in EXPERT_IDS:
        assert w[expert_id] == pytest.approx(expected[expert_id])


def test_weight_delta_above_bound_is_clipped_at_write_time():
    entry = ExpertWeightEntry(
        entry_id="x", expert_id="technical", market_state="up_stress", weight_delta=0.15
    )
    assert entry.weight_delta == pytest.approx(WEIGHT_DELTA_MAX)


def test_weight_delta_below_bound_is_clipped_at_write_time():
    entry = ExpertWeightEntry(
        entry_id="x", expert_id="news", market_state="down_stress", weight_delta=-0.15
    )
    assert entry.weight_delta == pytest.approx(WEIGHT_DELTA_MIN)


def test_clipping_logs_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        ExpertWeightEntry(
            entry_id="x", expert_id="technical", market_state="up_stress", weight_delta=0.15
        )
    assert any("clipped" in rec.getMessage().lower() for rec in caplog.records)


def test_unknown_expert_id_raises():
    with pytest.raises(ValueError):
        ExpertWeightEntry(entry_id="x", expert_id="sentiment", market_state="up_stress",
                          weight_delta=0.0)


def test_unknown_market_state_raises():
    with pytest.raises(ValueError):
        ExpertWeightEntry(entry_id="x", expert_id="technical", market_state="bull",
                          weight_delta=0.0)


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        _entry("technical", "up_stress", 0.0, status="pending")


def test_store_add_learned_adjustment_affects_weights():
    store = ExpertWeightStore()
    store.add_learned_adjustment(_entry("technical", "up_stress", 0.10, entry_id="t"))
    w = store.get_weights("up_stress")
    assert store.count_learned() == 1
    assert w["technical"] > EQ
    _assert_valid_weights(w)
    store.close()


def test_store_deprecated_adjustment_is_ignored():
    store = ExpertWeightStore()
    store.add_learned_adjustment(_entry("technical", "down_stress", 0.10, entry_id="c"))
    store.deprecate("c")
    w = store.get_weights("down_stress")
    for v in w.values():
        assert v == pytest.approx(EQ)
    store.close()


def test_store_weight_cache_reuses_parsed_learned_entries_and_invalidates(monkeypatch):
    store = ExpertWeightStore()
    store.add_learned_adjustment(_entry("technical", "up_stress", 0.10, entry_id="t"))

    calls = 0
    original_pd = expert_weights._pd

    def counting_pd(value):
        nonlocal calls
        calls += 1
        return original_pd(value)

    monkeypatch.setattr(expert_weights, "_pd", counting_pd)

    first = store.get_weights("up_stress")
    first_parse_calls = calls
    second = store.get_weights("up_stress")
    assert second == first
    assert calls == first_parse_calls

    store.add_learned_adjustment(_entry("news", "up_stress", 0.10, entry_id="f"))
    after_add = store.get_weights("up_stress")
    assert calls > first_parse_calls
    assert after_add["news"] > first["news"]

    calls_after_add = calls
    store.deprecate("f")
    after_deprecate = store.get_weights("up_stress")
    assert calls > calls_after_add
    assert after_deprecate == first
    store.close()


def test_store_weight_cache_does_not_leak_across_market_states():
    store = ExpertWeightStore()
    store.add_learned_adjustment(_entry("technical", "up_stress", 0.10, entry_id="n"))
    store.add_learned_adjustment(_entry("news", "down_stress", 0.10, entry_id="c"))

    normal = store.get_weights("up_stress")
    crisis = store.get_weights("down_stress")

    assert normal["technical"] > normal["news"]
    assert crisis["news"] > crisis["technical"]

    store.add_learned_adjustment(_entry("news", "up_stress", 0.10, entry_id="n2"))
    normal_after = store.get_weights("up_stress")
    crisis_after = store.get_weights("down_stress")

    assert normal_after["news"] > normal["news"]
    assert crisis_after == crisis
    store.close()


def test_store_freeze_prevents_writes_and_keeps_snapshot_stable():
    store = ExpertWeightStore()
    store.add_learned_adjustment(_entry("news", "up_stress", 0.05, entry_id="f"))
    before = store.snapshot()
    store.freeze(date(2024, 2, 1))
    with pytest.raises(RuntimeError):
        store.add_learned_adjustment(_entry("news", "up_stress", 0.05, entry_id="n"))
    with pytest.raises(RuntimeError):
        store.deprecate("f")
    assert store.snapshot() == before
    store.close()


def test_equal_weights_helper():
    w = equal_weights()
    assert set(w.keys()) == set(EXPERT_IDS)
    for v in w.values():
        assert v == pytest.approx(EQ)
    assert sum(w.values()) == pytest.approx(1.0)


