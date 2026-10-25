"""Tests for project_to_floored_simplex — the shared floor/normalisation primitive.

Invariants verified for every non-degenerate input:
  * all output weights >= floor (the property a single clamp-renorm pass fails)
  * output weights sum to exactly 1
  * keys are preserved

The headline case is a dominant weight (0.9) whose presence drags the other
weights below the floor under naive single-pass renorm; the iterative projection
must pin them at the floor instead.
"""

from __future__ import annotations

import pytest

from trading_agent.aggregation.weights import (
    WEIGHT_FLOOR,
    project_to_floored_simplex,
)


def _assert_valid(result, keys, floor=WEIGHT_FLOOR):
    assert set(result.keys()) == set(keys)
    assert all(v >= floor - 1e-12 for v in result.values()), result
    assert sum(result.values()) == pytest.approx(1.0)


FOUR = ("technical", "factor", "news", "fundamental")


# ---------------------------------------------------------------------------
# The headline dominant-weight case
# ---------------------------------------------------------------------------

def test_dominant_weight_others_stay_at_floor():
    # Single-pass clamp-renorm would leave the three others at 0.05/1.05 ≈ 0.0476,
    # below the floor. The iterative projection must pin them at exactly 0.05.
    raw = {"technical": 0.9, "factor": 0.0, "news": 0.0, "fundamental": 0.0}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    assert result["technical"] == pytest.approx(0.85)
    for eid in ("factor", "news", "fundamental"):
        assert result[eid] == pytest.approx(WEIGHT_FLOOR)


def test_extreme_dominant_weight_converges_above_floor():
    raw = {"technical": 0.999, "factor": 0.0005, "news": 0.0003, "fundamental": 0.0002}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    # technical takes 1 - 3*floor = 0.85, the rest pinned at the floor.
    assert result["technical"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Degenerate / equal inputs
# ---------------------------------------------------------------------------

def test_all_zero_splits_equally():
    result = project_to_floored_simplex({k: 0.0 for k in FOUR}, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    for v in result.values():
        assert v == pytest.approx(0.25)


def test_all_equal_unchanged():
    result = project_to_floored_simplex({k: 0.25 for k in FOUR}, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    for v in result.values():
        assert v == pytest.approx(0.25)


def test_tiny_equal_weights_split_equally():
    result = project_to_floored_simplex({k: 1e-6 for k in FOUR}, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    for v in result.values():
        assert v == pytest.approx(0.25)


def test_negative_inputs_treated_as_zero():
    raw = {"technical": -5.0, "factor": -1.0, "news": 0.0, "fundamental": 0.0}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    for v in result.values():
        assert v == pytest.approx(0.25)


def test_already_floored_unchanged():
    raw = {"technical": 0.7, "factor": 0.1, "news": 0.1, "fundamental": 0.1}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    # All allocations already >= floor, so proportions are preserved.
    assert result["technical"] == pytest.approx(0.7)
    assert result["factor"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Renormalisation of un-normalised but feasible inputs
# ---------------------------------------------------------------------------

def test_unnormalised_inputs_renormalised():
    raw = {"technical": 7.0, "factor": 1.0, "news": 1.0, "fundamental": 1.0}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    assert result["technical"] == pytest.approx(0.7)


def test_two_dominant_two_tiny():
    raw = {"technical": 0.49, "factor": 0.49, "news": 0.01, "fundamental": 0.01}
    result = project_to_floored_simplex(raw, WEIGHT_FLOOR)
    _assert_valid(result, FOUR)
    # news/fundamental would be 0.01 < floor, so they pin at 0.05 each;
    # the remaining 0.90 splits evenly between technical and factor → 0.45 each.
    assert result["news"] == pytest.approx(WEIGHT_FLOOR)
    assert result["fundamental"] == pytest.approx(WEIGHT_FLOOR)
    assert result["technical"] == pytest.approx(0.45)
    assert result["factor"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Floor parameter variations and feasibility guard
# ---------------------------------------------------------------------------

def test_custom_floor_respected():
    raw = {"technical": 0.97, "factor": 0.01, "news": 0.01, "fundamental": 0.01}
    result = project_to_floored_simplex(raw, floor=0.1)
    _assert_valid(result, FOUR, floor=0.1)
    assert result["technical"] == pytest.approx(0.7)  # 1 - 3*0.1


def test_zero_floor_is_plain_renormalisation():
    raw = {"technical": 0.9, "factor": 0.0, "news": 0.0, "fundamental": 0.0}
    result = project_to_floored_simplex(raw, floor=0.0)
    assert result["technical"] == pytest.approx(1.0)
    assert sum(result.values()) == pytest.approx(1.0)


def test_floor_exactly_feasible_boundary():
    # n * floor == 1 exactly: every weight must equal the floor.
    result = project_to_floored_simplex({k: 1.0 for k in FOUR}, floor=0.25)
    _assert_valid(result, FOUR, floor=0.25)
    for v in result.values():
        assert v == pytest.approx(0.25)


def test_infeasible_floor_raises():
    with pytest.raises(ValueError):
        project_to_floored_simplex({k: 1.0 for k in FOUR}, floor=0.3)  # 4*0.3 = 1.2 > 1


def test_negative_floor_raises():
    with pytest.raises(ValueError):
        project_to_floored_simplex({k: 1.0 for k in FOUR}, floor=-0.01)


def test_empty_input_returns_empty():
    assert project_to_floored_simplex({}, WEIGHT_FLOOR) == {}


# ---------------------------------------------------------------------------
# Convergence stress: random vectors always satisfy the invariants
# ---------------------------------------------------------------------------

def test_random_vectors_satisfy_invariants():
    import random

    rng = random.Random(20260625)
    for _ in range(500):
        n = rng.randint(2, 8)
        floor = rng.choice([0.0, 0.01, 0.05, 0.1])
        if n * floor > 1.0:
            continue
        raw = {f"e{i}": rng.random() * rng.choice([1, 10, 0.001]) for i in range(n)}
        # occasionally zero everything out
        if rng.random() < 0.1:
            raw = {k: 0.0 for k in raw}
        result = project_to_floored_simplex(raw, floor)
        assert set(result.keys()) == set(raw.keys())
        assert all(v >= floor - 1e-12 for v in result.values()), (raw, floor, result)
        assert sum(result.values()) == pytest.approx(1.0)
