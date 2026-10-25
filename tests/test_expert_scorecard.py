"""Unit tests for the scorecard-based expert-trust memory (Fable learning layer)."""

from __future__ import annotations

import math

import pytest

from trading_agent.aggregation.soft_weighting import EXPERT_IDS
from trading_agent.aggregation.weights import WEIGHT_FLOOR
from trading_agent.memory.expert_scorecard import (
    DEFAULT_MIN_SUPPORT,
    GLOBAL_MARKET_STATE,
    ExpertScorecardStore,
    Scorecard,
    equal_weights,
)

_STATE = "down_stress"


def _observe(store, state, contrib, n):
    for _ in range(n):
        store.observe(state, contrib)


# ---------------------------------------------------------------------------
# Scorecard EW mean/variance
# ---------------------------------------------------------------------------

def test_scorecard_first_observation_seeds_mean_and_zero_variance():
    sc = Scorecard()
    sc.observe(0.02)
    assert sc.count == 1
    assert sc.ew_mean == pytest.approx(0.02)
    assert sc.ew_var == pytest.approx(0.0)
    assert sc.hits == 1


def test_scorecard_ew_mean_moves_toward_new_values_and_var_nonnegative():
    sc = Scorecard()
    alpha = 0.5
    sc.observe(0.0, alpha=alpha)
    sc.observe(0.10, alpha=alpha)   # mean -> 0.05
    assert sc.ew_mean == pytest.approx(0.05)
    assert sc.ew_var >= 0.0
    assert sc.hit_rate == pytest.approx(0.5)   # one positive of two


def test_score_is_mean_minus_variance_penalty():
    sc = Scorecard(ew_mean=0.03, ew_var=0.004)
    assert sc.score == pytest.approx(0.03 - 1.0 * 0.004)


# ---------------------------------------------------------------------------
# Retrieval: equal-weight fallback, tilt, floor
# ---------------------------------------------------------------------------

def test_under_min_support_returns_equal_weights():
    store = ExpertScorecardStore(min_support=5)
    _observe(store, _STATE, {e: 0.01 for e in EXPERT_IDS}, n=4)   # 4 < 5
    assert store.get_weights(_STATE) == equal_weights()


def test_scorecard_tilts_toward_the_better_expert_and_respects_floor():
    store = ExpertScorecardStore(min_support=5)
    best = EXPERT_IDS[0]
    contrib = {e: (0.02 if e == best else -0.01) for e in EXPERT_IDS}
    _observe(store, _STATE, contrib, n=20)
    w = store.get_weights(_STATE)
    assert w[best] == max(w.values())
    assert w[best] > 1.0 / 3.0                # tilted above equal
    assert all(v >= WEIGHT_FLOOR - 1e-9 for v in w.values())   # floor respected
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # floor caps concentration at 1 - (n_experts-1)*floor = 0.90 for three experts
    assert max(w.values()) <= 0.90 + 1e-9


def test_all_equal_scores_give_equal_weights():
    store = ExpertScorecardStore(min_support=5)
    _observe(store, _STATE, {e: 0.01 for e in EXPERT_IDS}, n=10)
    w = store.get_weights(_STATE)
    for e in EXPERT_IDS:
        assert w[e] == pytest.approx(1.0 / 3.0)


def test_typical_bp_scale_produces_differentiated_weights():
    # SANITY: daily counterfactual contributions are a few basis points. At the grid-centre
    # tau=0.001 the softmax must produce a real tilt, not a degenerate equal 1/3 split
    # (which tau=0.01 would have given). This guards the dimensional scale of tau.
    from trading_agent.memory.expert_scorecard import SOFTMAX_TEMPERATURE
    assert SOFTMAX_TEMPERATURE == 0.001
    store = ExpertScorecardStore(min_support=5, standardize=False,
                                 temperature=SOFTMAX_TEMPERATURE)   # the raw ablation
    per = {EXPERT_IDS[0]: 0.0005, EXPERT_IDS[1]: -0.0002,   # +5 bp / -2 bp
           EXPERT_IDS[2]: -0.0003}                          # -3 bp
    _observe(store, _STATE, per, n=25)
    w = store.get_weights(_STATE)
    print("bp-scale weights:", {e: round(w[e], 4) for e in EXPERT_IDS})
    assert w[EXPERT_IDS[0]] == max(w.values())
    assert w[EXPERT_IDS[0]] > 0.40
    assert max(w.values()) - min(w.values()) > 0.10        # genuinely differentiated


# ---------------------------------------------------------------------------
# Reflection delta: applied vs numeric-only
# ---------------------------------------------------------------------------

def test_reflection_delta_shifts_weight_and_is_clipped():
    store = ExpertScorecardStore(min_support=5)
    _observe(store, _STATE, {e: 0.005 for e in EXPERT_IDS}, n=10)   # all equal baseline
    target = EXPERT_IDS[1]
    store.set_reflection_delta(_STATE, target, 0.10)
    w = store.get_weights(_STATE)
    assert w[target] == max(w.values())        # delta breaks the tie toward target
    # a delta beyond the bound is clipped to 0.10.
    store.set_reflection_delta(_STATE, target, 5.0)
    from trading_agent.memory.expert_scorecard import REFLECTION_DELTA_BOUND
    assert store._cards[(_STATE, target)].reflection_delta == pytest.approx(REFLECTION_DELTA_BOUND)


def test_numeric_only_store_ignores_reflection_delta():
    store = ExpertScorecardStore(min_support=5, use_reflection=False)
    _observe(store, _STATE, {e: 0.005 for e in EXPERT_IDS}, n=10)
    store.set_reflection_delta(_STATE, EXPERT_IDS[1], 0.10)
    w = store.get_weights(_STATE)
    for e in EXPERT_IDS:
        assert w[e] == pytest.approx(1.0 / 3.0)  # delta ignored -> stays equal


# ---------------------------------------------------------------------------
# Scope: state vs global
# ---------------------------------------------------------------------------

def test_global_scope_pools_states_and_returns_same_weights_everywhere():
    store = ExpertScorecardStore(scope="global", min_support=5)
    best = EXPERT_IDS[2]
    contrib = {e: (0.02 if e == best else -0.01) for e in EXPERT_IDS}
    _observe(store, "up_calm", contrib, n=6)
    _observe(store, "down_stress", contrib, n=6)
    w_up = store.get_weights("up_calm")
    w_down = store.get_weights("down_stress")
    assert w_up == w_down                        # global -> state-independent
    assert w_up[best] == max(w_up.values())


def test_state_scope_learns_per_state():
    store = ExpertScorecardStore(scope="state", min_support=5)
    _observe(store, "up_calm", {e: (0.02 if e == EXPERT_IDS[0] else -0.01) for e in EXPERT_IDS}, n=6)
    # a different, unlearned state falls back to equal weights.
    assert store.get_weights("down_stress") == equal_weights()
    assert store.get_weights("up_calm")[EXPERT_IDS[0]] > 1.0 / 3.0


# ---------------------------------------------------------------------------
# Freeze + snapshot round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scope", ["state", "global"])
def test_freeze_blocks_writes_for_both_arms(scope):
    # The frozen test window must block accumulation for the global (pooled) arm exactly
    # as for the state arm -- pooling is not a licence to keep learning past the freeze.
    store = ExpertScorecardStore(scope=scope, min_support=1)
    store.observe(_STATE, {e: 0.01 for e in EXPERT_IDS})
    store.freeze()
    with pytest.raises(RuntimeError):
        store.observe(_STATE, {e: 0.01 for e in EXPERT_IDS})
    with pytest.raises(RuntimeError):
        store.set_reflection_delta(_STATE, EXPERT_IDS[0], 0.05)


def test_production_default_thresholds():
    from trading_agent.memory.expert_scorecard import DEFAULT_MIN_SUPPORT, SCORECARD_HALFLIFE
    assert DEFAULT_MIN_SUPPORT == 20        # tilt threshold matches the dissertation
    assert SCORECARD_HALFLIFE == 63
    assert ExpertScorecardStore()._min_support == 20


def test_snapshot_round_trip_reproduces_weights():
    store = ExpertScorecardStore(min_support=5)
    _observe(store, _STATE, {e: (0.02 if e == EXPERT_IDS[-1] else -0.01) for e in EXPERT_IDS}, n=15)
    store.set_reflection_delta(_STATE, EXPERT_IDS[-1], 0.05)
    before = store.get_weights(_STATE)
    restored = ExpertScorecardStore.from_snapshot(store.snapshot())
    assert restored.get_weights(_STATE) == pytest.approx(before)
    assert restored.scope == "state"


def test_counts_for_metrics():
    store = ExpertScorecardStore(min_support=5)
    _observe(store, "up_calm", {e: 0.01 for e in EXPERT_IDS}, n=6)
    _observe(store, "down_calm", {e: 0.01 for e in EXPERT_IDS}, n=3)   # under support
    store.set_reflection_delta("up_calm", EXPERT_IDS[0], 0.05)
    assert store.learned_state_count() == 1        # only up_calm clears min_support
    assert store.reflection_delta_count() == 1
    assert store.support("up_calm") == 6


def test_standardized_retrieval_gives_dimensionless_tau():
    # Daily-return-scale moments (~1e-3): with tau=1 the raw store is stuck at
    # near-equal weights, while the standardized store tilts decisively.
    raw = ExpertScorecardStore(scope="state", temperature=1.0, min_support=5,
                               standardize=False)
    std = ExpertScorecardStore(scope="state", temperature=1.0, min_support=5, standardize=True)
    for store in (raw, std):
        for i in range(30):
            wiggle = 0.002 if i % 2 == 0 else 0.0   # nonzero variance, like real returns
            store.observe("up_calm", {"technical": 0.002 + wiggle, "news": -0.002 - wiggle,
                                      "fundamental": 0.0005 * (1 if i % 2 else -1)})
    w_raw = raw.get_weights("up_calm")
    w_std = std.get_weights("up_calm")
    assert abs(w_raw["technical"] - 1.0 / 3.0) < 0.01     # raw z ~ 1e-3: no tilt
    assert w_std["technical"] > 0.40                       # standardized: real tilt
    assert w_std["technical"] > w_std["news"]


def test_standardize_flag_survives_snapshot_roundtrip():
    store = ExpertScorecardStore(scope="state", standardize=True, shrink_to_global=True)
    for _ in range(25):
        store.observe("up_calm", {"technical": 0.002, "news": -0.001,
                                  "fundamental": 0.001})
    clone = ExpertScorecardStore.from_snapshot(store.snapshot(), temperature=1.0)
    assert clone.get_weights("up_calm") == store.__class__.from_snapshot(
        store.snapshot(), temperature=1.0).get_weights("up_calm")
    assert clone.snapshot()["standardize"] is True
    assert clone.snapshot()["shrink_to_global"] is True


def test_shrinkage_uses_pooled_evidence_for_an_unseen_state():
    # Plenty of pooled evidence, none in down_stress: the shrunk store returns
    # global-shaped weights instead of the equal fallback.
    store = ExpertScorecardStore(scope="state", min_support=20, standardize=True,
                                 shrink_to_global=True, temperature=1.0)
    for i in range(40):
        wiggle = 0.003 if i % 2 == 0 else 0.0
        store.observe("up_calm", {"technical": 0.003 + wiggle, "news": -0.003 - wiggle,
                                  "fundamental": 0.001 * (1 if i % 2 else -1)})
    w = store.get_weights("down_stress")     # zero down_stress episodes
    assert w["technical"] > 0.36             # pooled tilt, not 1/3
    plain = ExpertScorecardStore(scope="state", min_support=20, temperature=1.0)
    for i in range(40):
        wiggle = 0.003 if i % 2 == 0 else 0.0
        plain.observe("up_calm", {"technical": 0.003 + wiggle, "news": -0.003 - wiggle,
                                  "fundamental": 0.001 * (1 if i % 2 else -1)})
    assert plain.get_weights("down_stress") == equal_weights()   # old behaviour


def test_shrinkage_lambda_moves_toward_state_evidence_as_support_grows():
    store = ExpertScorecardStore(scope="state", min_support=20, standardize=True,
                                 shrink_to_global=True, temperature=1.0)
    # Global evidence favours technical; down_stress evidence contradicts it.
    for i in range(200):
        wiggle = 0.002 if i % 2 == 0 else 0.0
        store.observe("up_calm", {"technical": 0.003 + wiggle, "news": -0.003 - wiggle,
                                  "fundamental": 0.001 * (1 if i % 2 else -1)})
    for i in range(200):
        wiggle = 0.002 if i % 2 == 0 else 0.0
        store.observe("down_stress", {"technical": -0.003 - wiggle, "news": 0.003 + wiggle,
                                      "fundamental": 0.001 * (1 if i % 2 else -1)})
    w = store.get_weights("down_stress")
    assert w["news"] > w["technical"]        # local evidence dominates at high support


def test_reflection_delta_is_set_semantics_not_accumulation():
    # Audit 2026-07-18: each reflection round OVERWRITES the cell's delta (SET),
    # so the frozen value is the LAST window's proposal — pinned here so any
    # future switch to accumulation is a deliberate, tested decision.
    store = ExpertScorecardStore(scope="state")
    store.set_reflection_delta(_STATE, EXPERT_IDS[0], 0.10)
    store.set_reflection_delta(_STATE, EXPERT_IDS[0], -0.03)
    entries = {(e.market_state, e.expert_id): e.weight_delta for e in store.learned()}
    assert entries == {(_STATE, EXPERT_IDS[0]): -0.03}   # overwritten, not 0.07


def test_certified_only_retrieval_never_deviates_without_certification():
    # Never-harm rule (2026-07-19): without certified cells, retrieval is equal
    # weights even when the scorecard carries strong (uncertified) statistics.
    store = ExpertScorecardStore(scope="state", certified_only=True)
    for _ in range(40):
        store.observe(_STATE, {EXPERT_IDS[0]: 0.02, EXPERT_IDS[1]: -0.02, EXPERT_IDS[2]: 0.0})
    w = store.get_weights(_STATE)
    assert all(v == pytest.approx(1.0 / len(EXPERT_IDS)) for v in w.values())

    # Certify ONE cell: only that expert's evidence may tilt the weights.
    store.set_certified_cells({(_STATE, EXPERT_IDS[0])})
    w2 = store.get_weights(_STATE)
    assert w2[EXPERT_IDS[0]] > 1.0 / len(EXPERT_IDS)
    assert w2[EXPERT_IDS[1]] == pytest.approx(w2[EXPERT_IDS[2]])  # uncertified: symmetric


def test_certified_only_roundtrips_through_snapshot():
    store = ExpertScorecardStore(scope="state", certified_only=True)
    for _ in range(40):
        store.observe(_STATE, {EXPERT_IDS[0]: 0.02, EXPERT_IDS[1]: -0.02, EXPERT_IDS[2]: 0.0})
    store.set_certified_cells({(_STATE, EXPERT_IDS[0])})
    snap = store.snapshot()
    assert snap["certified_only"] is True
    assert snap["certified_cells"] == [[_STATE, EXPERT_IDS[0]]]

    rebuilt = ExpertScorecardStore.from_snapshot(snap)
    assert rebuilt.get_weights(_STATE) == store.get_weights(_STATE)


def test_certified_only_off_is_byte_identical_to_before():
    # Default (flag off) must not change existing behaviour even when a
    # certified set happens to be recorded.
    a = ExpertScorecardStore(scope="state")
    b = ExpertScorecardStore(scope="state")
    for _ in range(40):
        for s in (a, b):
            s.observe(_STATE, {EXPERT_IDS[0]: 0.02, EXPERT_IDS[1]: -0.02, EXPERT_IDS[2]: 0.0})
    b.set_certified_cells({(_STATE, EXPERT_IDS[2])})
    assert a.get_weights(_STATE) == b.get_weights(_STATE)


def test_reflection_delta_saturation_scoped_to_low_tau():
    """CORRECTED SCOPE (2026-07-21): the 2026-07-20 'provably inert' claim held only at
    the saturated snapshot temperature (tau=0.001, the constructor default), NOT at the
    selected tau=0.5 of the sealed runs. This test keeps the true half: at tau=0.001 the
    z-scale (~tens) dominates the +-0.10 bound, so the delta cannot move a weight."""
    store_with = ExpertScorecardStore(scope="global", certified_only=True, standardize=True)
    store_without = ExpertScorecardStore(scope="global", certified_only=True,
                                         standardize=True, use_reflection=False)
    # a clearly separated certified cell set (fundamental best, technical worst)
    contrib = {EXPERT_IDS[0]: 0.03, EXPERT_IDS[1]: 0.0, EXPERT_IDS[2]: -0.03}
    for _ in range(40):
        store_with.observe(GLOBAL_MARKET_STATE, contrib)
        store_without.observe(GLOBAL_MARKET_STATE, contrib)
    certified = {(GLOBAL_MARKET_STATE, e) for e in EXPERT_IDS}
    store_with.set_certified_cells(certified)
    store_without.set_certified_cells(certified)
    # apply reflection deltas of BOTH signs, including one that contradicts the mean
    for e, d in zip(EXPERT_IDS, (0.10, -0.10, 0.10)):
        store_with.set_reflection_delta(GLOBAL_MARKET_STATE, e, d)
    w_with = store_with.get_weights(GLOBAL_MARKET_STATE)
    w_without = store_without.get_weights(GLOBAL_MARKET_STATE)
    for e in EXPERT_IDS:
        assert w_with[e] == w_without[e], f"delta moved {e}: {w_with[e]} != {w_without[e]}"


def test_reflection_delta_is_effective_at_selected_tau():
    """SEALED FACT (2026-07-21): at the sealed runs' selected temperature (tau=0.5,
    standardized retrieval) the bounded delta IS live — it shifts certified-cell weights.
    Uses BIIB's real frozen global moments so the check is deterministic and reproduces
    the artefact's 0.041 shift, locking out any reintroduced 'provably inert' claim."""
    # BIIB global memory (reports/final/BIIB__designD3_q_h21cert.json), all three cells
    # verified; deltas as written by the sealed maker.
    biib = [
        ("fundamental", 0.007104404024921661, 0.006391331467488197, +0.10),
        ("news", 0.009871437414064172, 0.010621429541695163, -0.04973229756970285),
        ("technical", -0.008416827770587597, 0.006305141482833536, -0.10),
    ]
    snap = {
        "scope": "global", "certified_only": True, "standardize": True,
        "certified_cells": [["global", e] for e, *_ in biib],
        "cards": [
            {"market_state": "global", "expert_id": e, "ew_mean": mean,
             "ew_var": var, "count": 3000, "hits": 1500, "reflection_delta": d}
            for e, mean, var, d in biib
        ],
    }
    s_with = ExpertScorecardStore.from_snapshot(snap, temperature=0.5)
    s_without = ExpertScorecardStore.from_snapshot(snap, temperature=0.5)
    s_without._use_reflection = False
    w_with = s_with.get_weights(GLOBAL_MARKET_STATE)
    w_without = s_without.get_weights(GLOBAL_MARKET_STATE)
    moved = max(abs(w_with[e] - w_without[e]) for e in ("fundamental", "news", "technical"))
    assert 0.03 < moved < 0.06, f"expected ~0.041 shift at tau=0.5, got {moved}"


def test_frozen_snapshots_deltas_live_at_selected_temperature():
    """SEALED FACT (2026-07-21, supersedes the retracted inertness check): on the real
    frozen certified snapshots, retrieval at the run's SELECTED temperature (0.5) carries
    the bounded delta live — the global-arm weights differ between delta-on and delta-off.
    The old check read the snapshot's stored tau=0.001, where saturation masked the delta.
    Skips cleanly if the artifacts are absent (they are gitignored)."""
    import json
    from pathlib import Path

    cases = [
        ("reports/final/BIIB__designD3_q_h21cert.json", "global_memory"),
        ("reports/final/ECL__sealed.json", "global_memory"),
    ]
    checked = 0
    for path, arm in cases:
        p = Path(path)
        if not p.exists():
            continue
        report = json.loads(p.read_text(encoding="utf-8"))
        snap = report["arm_memory_snapshots"][arm]
        tau = float(report.get("selected_temperature", 0.5))
        s_with = ExpertScorecardStore.from_snapshot(snap, temperature=tau)
        s_without = ExpertScorecardStore.from_snapshot(snap, temperature=tau)
        s_without._use_reflection = False
        states = {c[0] for c in snap.get("certified_cells", [])} or {GLOBAL_MARKET_STATE}
        for st in states:
            w_with = s_with.get_weights(st)
            w_without = s_without.get_weights(st)
            moved = max(abs(w_with[e] - w_without[e]) for e in EXPERT_IDS)
            assert moved > 0.005, f"{path} {st}: delta unexpectedly inert (max {moved})"
        checked += 1
    if checked == 0:
        pytest.skip("frozen certified snapshots not present (gitignored artifacts)")
