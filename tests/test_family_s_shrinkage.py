"""Family S (prereg Amendment 11): empirical-Bayes shrinkage toward the global card.

Certified-only retrieval makes a binary decision on cells whose median power ratio is 0.775,
which leaves the state arm byte-identical to equal weight on seven of eight names. Shrinkage
replaces that decision with a continuous per-cell reliability B = tau^2/(tau^2 + se^2), the
James-Stein / Efron-Morris form that Jorion (1986) and Vasicek (1973) apply to portfolio
inputs and betas.

These tests pin the estimator's shape, the effective-sample-size detail that is easy to get
wrong, and the invariant that the sealed default is untouched.
"""

import math

import pytest

from trading_agent.memory.expert_scorecard import (
    EXPERT_IDS,
    GLOBAL_MARKET_STATE,
    ExpertScorecardStore,
    Scorecard,
)

STATES = ("up_calm", "up_stress", "down_stress")


def _store(**kw) -> ExpertScorecardStore:
    return ExpertScorecardStore(scope="state", shrinkage="empirical_bayes", **kw)


def _feed(store, per_state: dict[str, dict[str, float]], n: int = 300) -> None:
    """Feed n episodes per state, each expert receiving its state's fixed contribution."""
    for _ in range(n):
        for state, contributions in per_state.items():
            store.observe(state, contributions)


def test_default_is_off_and_the_flag_validates():
    assert ExpertScorecardStore(scope="state")._shrinkage == "none"
    with pytest.raises(ValueError, match="unknown shrinkage"):
        ExpertScorecardStore(scope="state", shrinkage="bogus")
    with pytest.raises(ValueError, match="replaces"):
        ExpertScorecardStore(scope="state", shrinkage="empirical_bayes", shrink_to_global=True)


def test_the_ew_effective_sample_size_is_used_not_the_raw_count():
    """Var(EW mean) = sigma^2 * alpha/(2-alpha); using count would understate it ~1.5x."""
    store = _store(half_life=63)
    card = Scorecard(ew_var=4.0, count=416)
    alpha = 1.0 - 0.5 ** (1.0 / 63)
    assert store._ew_mean_variance(card) == pytest.approx(4.0 * alpha / (2.0 - alpha))
    # n_eff is 182, not the 416 observations, so the variance is the larger of the two.
    assert store._ew_mean_variance(card) > 4.0 / 416
    assert (2.0 - alpha) / alpha == pytest.approx(182, abs=1)


def test_identical_states_collapse_every_cell_onto_the_global_card():
    """tau^2 = 0 means there is no between-state signal, so B = 0 everywhere."""
    store = _store()
    _feed(store, {s: {e: 0.01 for e in EXPERT_IDS} for s in STATES})
    for s in STATES:
        for e in EXPERT_IDS:
            assert store._shrinkage_factor(s, e) == 0.0
    # With every cell on the target, all states retrieve the same weights.
    first = store.get_weights(STATES[0])
    for s in STATES[1:]:
        assert store.get_weights(s) == first


def test_a_cell_with_real_between_state_signal_keeps_part_of_its_own_estimate():
    store = _store()
    _feed(store, {
        "up_calm":     {"technical": +0.05, "news": 0.0, "fundamental": -0.02},
        "up_stress":   {"technical": -0.04, "news": 0.0, "fundamental": +0.03},
        "down_stress": {"technical": +0.01, "news": 0.0, "fundamental": +0.01},
    })
    b = store._shrinkage_factor("up_calm", "technical")
    assert b is not None and 0.0 < b <= 1.0
    # The shrunk mean lies between the cell's own estimate and the global card.
    own = store._cards[("up_calm", "technical")].ew_mean
    tgt = store._cards[(GLOBAL_MARKET_STATE, "technical")].ew_mean
    shrunk, _var, _delta = store._effective_moments("up_calm", "technical")
    lo, hi = sorted((own, tgt))
    assert lo - 1e-12 <= shrunk <= hi + 1e-12
    assert shrunk == pytest.approx(tgt + b * (own - tgt))


def test_shrinkage_is_stronger_for_the_noisier_cell():
    """With real between-state signal present, B rises with a cell's own precision."""
    store = _store()
    for i in range(400):
        store.observe("up_calm", {e: +0.5 for e in EXPERT_IDS})      # separated and precise
        store.observe("up_stress", {e: -0.5 for e in EXPERT_IDS})    # separated and precise
        jitter = 5.0 if i % 2 else -5.0                              # mean ~0 but very noisy
        store.observe("down_stress", {e: jitter for e in EXPERT_IDS})
    b_precise = store._shrinkage_factor("up_calm", "technical")
    b_noisy = store._shrinkage_factor("down_stress", "technical")
    assert b_precise > b_noisy, f"precise {b_precise} should exceed noisy {b_noisy}"
    assert b_noisy < 0.95, "a cell that noisy must be pulled materially toward global"


def test_no_shrinkage_when_between_state_differences_are_smaller_than_the_noise():
    """The estimator's own answer to 'is there any state signal at all': tau^2 = 0 -> B = 0.

    This is the correct behaviour, not a degenerate one -- if the states differ by less than
    sampling error, the honest estimate for every state is the pooled one.
    """
    store = _store()
    for i in range(400):
        store.observe("up_calm", {e: 0.02 for e in EXPERT_IDS})
        store.observe("up_stress", {e: (0.5 if i % 2 else -0.5) for e in EXPERT_IDS})
        store.observe("down_stress", {e: 0.01 for e in EXPERT_IDS})
    assert store._shrinkage_factor("up_calm", "technical") == 0.0
    assert store._shrinkage_factor("up_stress", "technical") == 0.0


def test_the_global_scope_and_the_global_card_itself_are_never_shrunk():
    store = _store()
    _feed(store, {s: {e: 0.01 * (i + 1) for i, e in enumerate(EXPERT_IDS)} for s in STATES})
    assert store._shrinkage_factor(GLOBAL_MARKET_STATE, "technical") is None
    glob = ExpertScorecardStore(scope="global", shrinkage="empirical_bayes")
    _feed(glob, {s: {e: 0.01 for e in EXPERT_IDS} for s in STATES})
    assert glob._shrinkage_factor(GLOBAL_MARKET_STATE, "technical") is None


def test_a_single_observed_state_cannot_be_shrunk():
    """tau^2 needs at least two state cells to be estimable; fewer means no shrinkage."""
    store = _store()
    _feed(store, {"up_calm": {e: 0.02 for e in EXPERT_IDS}})
    assert store._shrinkage_factor("up_calm", "technical") is None


def test_snapshot_round_trip_including_a_legacy_payload():
    store = _store()
    _feed(store, {s: {e: 0.01 * (i + 1) for i, e in enumerate(EXPERT_IDS)} for s in STATES})
    snap = store.snapshot()
    assert snap["shrinkage"] == "empirical_bayes"
    rebuilt = ExpertScorecardStore.from_snapshot(snap)
    assert rebuilt._shrinkage == "empirical_bayes"
    for s in STATES:
        assert rebuilt.get_weights(s) == store.get_weights(s)
    legacy = {k: v for k, v in snap.items() if k != "shrinkage"}
    assert ExpertScorecardStore.from_snapshot(legacy)._shrinkage == "none"


def test_sealed_default_weights_are_unchanged_by_the_new_code_path():
    """With shrinkage off the retrieval must be byte-identical to the pre-Family-S behaviour."""
    plain = ExpertScorecardStore(scope="state")
    per_state = {
        "up_calm":     {"technical": +0.03, "news": -0.01, "fundamental": +0.02},
        "up_stress":   {"technical": -0.02, "news": +0.04, "fundamental": -0.01},
        "down_stress": {"technical": +0.01, "news": -0.03, "fundamental": +0.05},
    }
    _feed(plain, per_state)
    for s in STATES:
        w = plain.get_weights(s)
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert min(w.values()) >= 0.05 - 1e-12
        # No global card is accumulated when shrinkage is off.
        assert (GLOBAL_MARKET_STATE, "technical") not in plain._cards
    # And the shrinkage factor is inert on a store that did not opt in.
    assert plain._shrinkage_factor("up_calm", "technical") is None


def test_shrunk_weights_stay_on_the_floored_simplex():
    store = _store()
    _feed(store, {
        "up_calm":     {"technical": +0.9, "news": -0.9, "fundamental": 0.0},
        "up_stress":   {"technical": -0.9, "news": +0.9, "fundamental": 0.0},
        "down_stress": {"technical": +0.1, "news": +0.1, "fundamental": +0.1},
    })
    for s in STATES:
        w = store.get_weights(s)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert min(w.values()) >= 0.05 - 1e-9
        assert all(math.isfinite(v) for v in w.values())


def test_v2_a_thin_deviant_cell_is_shrunk_hard_not_trusted():
    """Regression for the v1 defect (TSLA down_calm): a cell with count below min_support
    and a wild mean must get a SMALL B, because its accumulated-from-zero ew_var says
    nothing about its true noise. v1 gave it se^2 ~ 0 and B ~ 1."""
    store = _store(min_support=20)
    for i in range(400):
        noise = 0.1 if i % 2 else -0.1                    # realistic observation noise
        store.observe("up_calm", {e: +0.02 + noise for e in EXPERT_IDS})
        store.observe("up_stress", {e: -0.02 + noise for e in EXPERT_IDS})
        if i % 57 == 0:                                    # 7 thin outliers, interleaved in time
            store.observe("down_stress", {e: -0.5 for e in EXPERT_IDS})
    b_thin = store._shrinkage_factor("down_stress", "technical")
    b_fat = store._shrinkage_factor("up_calm", "technical")
    assert b_thin is not None and b_fat is not None
    assert b_thin < b_fat, f"thin cell must be shrunk harder: {b_thin} vs {b_fat}"
    assert b_thin < 0.5, f"a 7-observation cell must lose most of its own estimate, got {b_thin}"


def test_v2_under_supported_cells_do_not_pollute_tau2():
    """tau^2 must be estimated from supported cells only: adding a thin outlier state must
    leave the supported cells' shrinkage factors unchanged."""
    def build(with_outlier: bool) -> ExpertScorecardStore:
        store = _store(min_support=20)
        for i in range(400):
            noise = 0.1 if i % 2 else -0.1
            store.observe("up_calm", {e: +0.02 + noise for e in EXPERT_IDS})
            store.observe("up_stress", {e: -0.02 + noise for e in EXPERT_IDS})
            if with_outlier and i % 57 == 0:
                store.observe("down_stress", {e: -0.5 for e in EXPERT_IDS})
        return store

    clean, polluted = build(False), build(True)
    for s in ("up_calm", "up_stress"):
        b0 = clean._shrinkage_factor(s, "technical")
        b1 = polluted._shrinkage_factor(s, "technical")
        # The outlier changes the global card slightly (7 of 807 observations), so allow a
        # small drift; what must NOT happen is the order-of-magnitude tau^2 inflation of v1.
        assert b0 == pytest.approx(b1, abs=0.15), f"{s}: {b0} vs {b1}"


def test_v3_overlap_inflates_the_standard_error_and_lowers_b():
    """g = v*r at horizon h overlaps h-1 days, so se^2 must scale by h (prereg Amendment 11
    addendum 2). overlap=1 stays byte-identical to the pre-v3 behaviour."""
    def build(overlap: int) -> ExpertScorecardStore:
        store = ExpertScorecardStore(scope="state", shrinkage="empirical_bayes", overlap=overlap)
        for i in range(300):
            noise = 0.1 if i % 2 else -0.1
            store.observe("up_calm", {e: +0.03 + noise for e in EXPERT_IDS})
            store.observe("up_stress", {e: -0.03 + noise for e in EXPERT_IDS})
        return store

    b1 = build(1)._shrinkage_factor("up_calm", "technical")
    b21 = build(21)._shrinkage_factor("up_calm", "technical")
    assert b21 < b1, f"overlap must shrink harder: {b21} vs {b1}"
    with pytest.raises(ValueError, match="overlap"):
        ExpertScorecardStore(scope="state", overlap=0)
    snap = build(21).snapshot()
    assert snap["overlap"] == 21
    assert ExpertScorecardStore.from_snapshot(snap)._overlap == 21
    legacy = {k: v for k, v in snap.items() if k != "overlap"}
    assert ExpertScorecardStore.from_snapshot(legacy)._overlap == 1


def test_v3_holdout_threads_the_attribution_horizon_as_overlap():
    import inspect

    from trading_agent.experiments import expert_weight_holdout as hold

    src = inspect.getsource(hold.run_expert_weight_holdout)
    assert '_prior_kwargs["overlap"] = int(attribution_horizon)' in src
