"""Small-scale validation of the verification pass (architecture.md sec.11 rigor caveat 2).

Verification promotes a reflection delta only for (expert, market_state) cells whose
counterfactual skill is statistically verified (moving-block bootstrap, autocorrelation-robust)
after multiplicity control across cells. These tests prove it discriminates real skill from
noise, controls false discoveries, computes verification coverage, and leaves the default
(verification off) reflection byte-identical.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from trading_agent.memory.expert_reflection import (
    AttributedEpisode,
    DeterministicExpertReflection,
    apply_terminal_gate,
    marginal_series,
    terminal_verify,
    verify_reliable_cells,
)

_ATTR0 = {"market": 0.0, "style": 0.0, "signal": 0.0, "sizing": 0.0, "execution": 0.0}


def _series(rng, mean, std, n=40):
    return [rng.gauss(mean, std) for _ in range(n)]


def test_verify_accepts_real_skill_rejects_noise():
    rng = random.Random(1)
    by_state = {
        # reliably positive directional skill -> should verify
        "down_stress": {
            "technical": _series(rng, 0.012, 0.006),
            # pure noise, centred on zero -> should NOT verify
            "news": _series(rng, 0.0, 0.03),
            "financials": _series(rng, 0.0, 0.03),
            "disclosure": _series(rng, 0.0, 0.03),
        },
    }
    verified, info = verify_reliable_cells(by_state)
    assert ("down_stress", "technical") in verified
    assert ("down_stress", "news") not in verified
    assert info["cells_tested"] == 4
    assert info["cells_verified"] == 1
    assert info["verification_coverage"] == pytest.approx(0.25)


def test_fdr_controls_false_discoveries_across_many_noise_cells():
    # 16 pure-noise cells: with FDR at 0.10, we expect near-zero verified (well under a
    # per-cell 5% t-test's expected ~1-2 false positives).
    rng = random.Random(7)
    by_state = {
        st: {f"e{i}": _series(rng, 0.0, 0.03) for i in range(4)}
        for st in ("up_calm", "up_stress", "down_calm", "down_stress")
    }
    verified, info = verify_reliable_cells(by_state)
    assert info["cells_tested"] == 16
    assert info["verification_coverage"] <= 0.125  # <= 2/16, FDR keeps it low


def test_below_min_support_cells_are_not_tested():
    rng = random.Random(3)
    by_state = {"down_stress": {"technical": _series(rng, 0.05, 0.001, n=10)}}  # n<20
    verified, info = verify_reliable_cells(by_state, min_support=20)
    assert info["cells_tested"] == 0
    assert verified == set()


def test_gate_is_deterministic_given_seed():
    rng = random.Random(5)
    by_state = {"down_stress": {"technical": _series(rng, 0.01, 0.008),
                                "news": _series(rng, 0.0, 0.03)}}
    a, _ = verify_reliable_cells(by_state, seed=42)
    b, _ = verify_reliable_cells(by_state, seed=42)
    assert a == b


def _episodes():
    """40 down_stress episodes: technical reliably +, news pure noise."""
    rng = random.Random(11)
    eps = []
    for i in range(40):
        eps.append(AttributedEpisode(
            decision_date=date(2022, 1, 1),
            market_state="down_stress",
            attribution=dict(_ATTR0),
            signal_attribution={"technical": rng.gauss(0.012, 0.006),
                                "news": rng.gauss(0.0, 0.03)},
        ))
    return eps


def test_gate_off_default_still_proposes_for_both_experts():
    eps = _episodes()
    refl = DeterministicExpertReflection(min_support=3, step_mode="scale_free")
    out = refl.reflect(eps, created_at=date(2022, 2, 1), participation_rate=0.5)
    experts = {e.expert_id for e in out}
    assert experts == {"technical", "news"}          # both get a (relative) delta
    assert refl.verification_coverage is None          # verification never ran


def test_gate_on_writes_only_verified_cells():
    eps = _episodes()
    refl = DeterministicExpertReflection(min_support=3, step_mode="scale_free",
                                         verify_gate=True, verify_min_support=20)
    out = refl.reflect(eps, created_at=date(2022, 2, 1), participation_rate=0.5)
    experts = {e.expert_id for e in out}
    assert experts == {"technical"}                    # news failed verification -> dropped
    assert refl.verification_coverage == pytest.approx(0.5)  # 1 of 2 cells verified
    assert ("down_stress", "technical") in refl.verified_cells


# --- TERMINAL verification (protocol repair: m-proxy estimand, synchronized blocks, Holm) --

def _ep(day, state, contribs):
    return AttributedEpisode(
        decision_date=date(2022, 1, 1) + __import__("datetime").timedelta(days=day),
        market_state=state,
        attribution=dict(_ATTR0),
        signal_attribution=contribs,
    )


def test_marginal_series_two_expert_math():
    # N=2: m_a = (g_a - g_b)/2, m_b = (g_b - g_a)/2 (zero-sum by construction).
    eps = [_ep(0, "down_stress", {"technical": 0.03, "news": -0.01})]
    series = marginal_series(eps)
    assert series[("down_stress", "technical")][0][1] == pytest.approx((0.03 - (-0.01)) / 2)
    assert series[("down_stress", "news")][0][1] == pytest.approx((-0.01 - 0.03) / 2)


def test_marginal_series_skips_single_expert_episodes_and_pools_states():
    eps = [
        _ep(0, "down_stress", {"technical": 0.02}),                     # no relative signal
        _ep(1, "up_calm", {"technical": 0.02, "news": 0.0}),
        _ep(2, "down_calm", {"technical": 0.01, "news": -0.01}),
    ]
    series = marginal_series(eps)
    assert ("down_stress", "technical") not in series
    pooled = marginal_series(eps, pool_states=True)
    assert len(pooled[("global", "technical")]) == 2  # both multi-expert episodes pooled


def _terminal_episodes(n=48):
    """down_stress episodes: technical persistently beats the committee; two noise experts."""
    rng = random.Random(23)
    eps = []
    for i in range(n):
        eps.append(_ep(i, "down_stress", {
            "technical": 0.012 + rng.gauss(0, 0.004),
            "news": rng.gauss(0, 0.004),
            "fundamental": rng.gauss(0, 0.004),
        }))
    return eps


def test_terminal_verify_accepts_relative_skill_rejects_noise():
    verified, report = terminal_verify(_terminal_episodes(), min_support=20, n_boot=500)
    assert ("down_stress", "technical") in verified
    assert report["cells_tested"] == 3
    assert 0.0 < report["verification_coverage"] <= 1.0
    assert report["per_cell"]["down_stress|technical"]["verified"] is True
    assert "mde_80pct" in report["per_cell"]["down_stress|technical"]


def test_terminal_verify_report_carries_holm_receipt_fields():
    # Round-2 review receipt: the artifact must be self-contained — adjusted p
    # + rank per cell, omitted-cell list, verified/planned denominator, and the
    # se/mde semantics note (se is IID; inference is the MBB bootstrap p).
    verified, report = terminal_verify(_terminal_episodes(), min_support=20, n_boot=500)
    ranks = set()
    for name, cell in report["per_cell"].items():
        assert cell["p_holm"] >= cell["p"] - 1e-12, name          # adjustment never shrinks p
        assert 0.0 < cell["p_holm"] <= 1.0
        assert (cell["p_holm"] <= report["alpha"]) == cell["verified"], name
        ranks.add(cell["holm_rank"])
    assert ranks == {1, 2, 3}
    assert report["cells_omitted_below_support"] == []
    assert report["verified_over_planned"] == pytest.approx(
        report["cells_verified"] / report["cells_planned"]
    )
    assert "moving-block" in report["se_note"]


def test_terminal_verify_holm_controls_noise_cells():
    rng = random.Random(9)
    eps = []
    day = 0
    for st in ("up_calm", "up_stress", "down_calm", "down_stress"):
        for i in range(30):
            eps.append(_ep(day, st, {
                "technical": rng.gauss(0, 0.02),
                "news": rng.gauss(0, 0.02),
                "fundamental": rng.gauss(0, 0.02),
            }))
            day += 1
    verified, report = terminal_verify(eps, min_support=20, n_boot=500)
    assert report["cells_tested"] == 12  # 4 states x 3 experts (design D)
    assert len(verified) <= 1  # FWER at 0.10 over pure noise


def test_delta_sign_receipt_flags_recency_vs_fulltrain_tension():
    # Settled 2026-07-18: terminal clearing is SIGN-BLIND by design; the receipt makes the
    # time-horizon tension auditable. A verified cell whose delta contradicts the
    # full-train m sign is KEPT and flagged sign_agreement=False.
    from trading_agent.memory.expert_reflection import apply_terminal_gate, delta_sign_receipt
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    store = ExpertScorecardStore(scope="state")
    store.set_reflection_delta("down_stress", "technical", -0.10)   # contradicts m>0
    store.set_reflection_delta("down_stress", "fundamental", 0.05)  # agrees with m>0
    store.set_reflection_delta("up_calm", "news", 0.08)             # unverified cell

    report = {"per_cell": {
        "down_stress|technical": {"mean_m": 1.2e-3, "verified": True},
        "down_stress|fundamental": {"mean_m": 0.5e-3, "verified": True},
        "up_calm|news": {"mean_m": -0.1e-3, "verified": False},
    }}
    receipt = delta_sign_receipt(store, report)
    assert receipt["down_stress|technical"] == {
        "delta": -0.10, "verified": True, "verified_mean_m": 1.2e-3,
        "sign_agreement": False,
    }
    assert receipt["down_stress|fundamental"]["sign_agreement"] is True
    assert receipt["up_calm|news"]["verified"] is False

    # Sign-blind clearing keeps BOTH verified deltas (including the contradicting
    # one) and clears only the unverified cell.
    cleared = apply_terminal_gate(
        store, {("down_stress", "technical"), ("down_stress", "fundamental")}
    )
    assert cleared == 1
    kept = {(e.market_state, e.expert_id): e.weight_delta for e in store.learned()}
    assert kept == {
        ("down_stress", "technical"): -0.10,
        ("down_stress", "fundamental"): 0.05,
    }


def test_apply_terminal_gate_clears_unverified_keeps_verified():
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore
    store = ExpertScorecardStore(scope="state")
    store.set_reflection_delta("down_stress", "technical", 0.08)
    store.set_reflection_delta("down_stress", "news", -0.05)
    store.set_reflection_delta("up_calm", "fundamental", 0.03)
    cleared = apply_terminal_gate(store, {("down_stress", "technical")})
    assert cleared == 2
    kept = {(e.market_state, e.expert_id): e.weight_delta for e in store.learned()}
    assert kept == {("down_stress", "technical"): pytest.approx(0.08)}


def test_terminal_verify_deterministic_given_seed():
    eps = _terminal_episodes(30)
    a, _ = terminal_verify(eps, min_support=20, n_boot=300, seed=7)
    b, _ = terminal_verify(eps, min_support=20, n_boot=300, seed=7)
    assert a == b


def test_fdr_correction_is_at_least_as_permissive_as_holm():
    # Redesign knob: Benjamini-Hochberg FDR controls the false discovery rate instead of
    # the familywise error rate, so its verified set always contains Holm's (BH's critical
    # values dominate Holm's). The correction is recorded in the report.
    rng = random.Random(101)
    eps = []
    day = 0
    graded = {"down_stress": 0.012, "up_stress": 0.009, "up_calm": 0.007, "down_calm": 0.0}
    for st, mean in graded.items():
        for _i in range(40):
            eps.append(_ep(day, st, {
                "technical": mean + rng.gauss(0, 0.004),
                "news": rng.gauss(0, 0.004),
                "fundamental": rng.gauss(0, 0.004),
            }))
            day += 1
    holm_v, holm_r = terminal_verify(eps, min_support=20, n_boot=500, correction="holm")
    fdr_v, fdr_r = terminal_verify(eps, min_support=20, n_boot=500, correction="fdr")
    assert holm_v <= fdr_v                 # FDR verifies a superset of Holm's cells
    assert holm_r["correction"] == "holm"
    assert fdr_r["correction"] == "fdr"


def test_terminal_verify_rejects_unknown_correction():
    with pytest.raises(ValueError, match="unknown correction"):
        terminal_verify(_terminal_episodes(30), min_support=20, n_boot=100, correction="bogus")


def test_persists_across_halves_flags_sign_flip():
    from datetime import timedelta
    from trading_agent.memory.expert_reflection import _persists_across_halves
    d0 = date(2020, 1, 1)
    split = d0 + timedelta(days=9)
    flip = {d0 + timedelta(days=i): [0.02 if i < 10 else -0.02] for i in range(20)}
    assert _persists_across_halves(flip, split) is False          # sign flips between halves
    steady = {d0 + timedelta(days=i): [0.01] for i in range(20)}
    assert _persists_across_halves(steady, split) is True         # same sign throughout
    thin = {d0 + timedelta(days=i): [0.01] for i in range(3)}
    assert _persists_across_halves(thin, split) is False          # too few to confirm


def test_min_effect_floor_drops_small_but_significant_cell():
    # technical's marginal mean here is ~0.004; a 0.01 effect-size floor rejects it even
    # though it is statistically significant (tiny-but-consistent signal).
    eps = _terminal_episodes()
    base_v, _ = terminal_verify(eps, min_support=20, n_boot=500)
    floor_v, floor_r = terminal_verify(eps, min_support=20, n_boot=500, min_effect=0.01)
    assert ("down_stress", "technical") in base_v
    assert ("down_stress", "technical") not in floor_v
    assert floor_r["min_effect"] == 0.01
    assert floor_r["require_persistence"] is False


def test_oos_confirmation_drops_cell_that_flips_out_of_sample():
    from datetime import timedelta
    eps = _terminal_episodes()   # (down_stress, technical) verifies in-sample, mean_m > 0
    base_v, _ = terminal_verify(eps, min_support=20, n_boot=500)
    assert ("down_stress", "technical") in base_v
    d0 = date(2024, 1, 2)
    flip = {("down_stress", "technical"): [(d0 + timedelta(days=i), -0.01) for i in range(10)]}
    v_flip, rep = terminal_verify(eps, min_support=20, n_boot=500, oos_series=flip)
    assert ("down_stress", "technical") not in v_flip     # opposite sign out-of-sample -> dropped
    assert rep["oos_confirmed"] is True
    ok = {("down_stress", "technical"): [(d0 + timedelta(days=i), 0.01) for i in range(10)]}
    v_ok, _ = terminal_verify(eps, min_support=20, n_boot=500, oos_series=ok)
    assert ("down_stress", "technical") in v_ok           # same sign out-of-sample -> kept


def test_kfold_confirms_a_fold_stable_cell_and_reports_receipt():
    # A cell whose relative skill holds throughout training confirms in every fold.
    eps = _terminal_episodes()
    v, rep = terminal_verify(eps, min_support=20, n_boot=500, kfold=5)
    assert ("down_stress", "technical") in v
    assert rep["kfold"] == 5
    assert rep["kfold_signs"]["down_stress|technical"] == [1, 1, 1, 1, 1]


def test_kfold_drops_a_cell_whose_skill_flips_late_in_training():
    # Technical beats the committee for the first 60% of days, then reliably loses.
    # The full-train mean is still positive (in-sample significant) but the last two
    # folds disagree -> 3/5 agreement < 4 -> dropped. This is the failure shape the
    # thin-window confirmation could not test on all-train data.
    rng = random.Random(31)
    eps = []
    for i in range(60):
        eps.append(_ep(i, "down_stress", {
            "technical": 0.03 + rng.gauss(0, 0.003), "news": rng.gauss(0, 0.003),
        }))
    for i in range(60, 100):
        eps.append(_ep(i, "down_stress", {
            "technical": -0.005 + rng.gauss(0, 0.003), "news": rng.gauss(0, 0.003),
        }))
    base_v, _ = terminal_verify(eps, min_support=20, n_boot=500)
    kf_v, rep = terminal_verify(eps, min_support=20, n_boot=500, kfold=5)
    assert ("down_stress", "technical") in base_v          # in-sample significant overall
    assert ("down_stress", "technical") not in kf_v        # but not fold-stable
    signs = rep["kfold_signs"]["down_stress|technical"]
    assert signs[:3] == [1, 1, 1] and signs[3:] == [-1, -1]


def test_kfold_allows_one_noisy_fold():
    # 4/5 agreement passes at the pre-registered min_agree = k-1.
    rng = random.Random(37)
    eps = []
    for i in range(100):
        # one middle stretch (days 40-59, exactly fold 3) flips sign; rest stable +
        mean = -0.015 if 40 <= i < 60 else 0.012
        eps.append(_ep(i, "down_stress", {
            "technical": mean + rng.gauss(0, 0.003), "news": rng.gauss(0, 0.003),
        }))
    v, rep = terminal_verify(eps, min_support=20, n_boot=500, kfold=5)
    signs = rep["kfold_signs"].get("down_stress|technical")
    assert signs is not None and signs.count(1) == 4 and signs.count(-1) == 1
    assert ("down_stress", "technical") in v


def test_kfold_off_by_default_leaves_report_and_selection_unchanged():
    eps = _terminal_episodes()
    v0, r0 = terminal_verify(eps, min_support=20, n_boot=500)
    assert r0["kfold"] == 0 and "kfold_signs" not in r0
    v5, _ = terminal_verify(eps, min_support=20, n_boot=500, kfold=5)
    assert v0 == v5   # stable fixture: selection identical with the filter on


def test_oos_confirmation_drops_cell_with_no_held_out_data():
    eps = _terminal_episodes()
    v, _ = terminal_verify(eps, min_support=20, n_boot=500, oos_series={})   # cannot confirm
    assert v == set()


def test_block_len_override_reaches_the_report_and_defaults_for_h1():
    # Audit fix 2026-07-19: h-day overlapping counterfactual returns need blocks
    # spanning the dependence length; the override must reach the bootstrap and
    # the report receipt. h=1 (no override) keeps the a-priori n^(1/3) rule.
    from trading_agent.memory.expert_reflection import terminal_verify

    import random as _r
    from datetime import timedelta
    rng = _r.Random(7)
    episodes = [
        AttributedEpisode(
            decision_date=date(2022, 1, 3) + timedelta(days=i),
            market_state="down_stress",
            attribution=dict(_ATTR0),
            signal_attribution={"technical": rng.gauss(0.0, 0.01),
                                "news": rng.gauss(0.0, 0.01)},
        )
        for i in range(120)
    ]
    _, rep_default = terminal_verify(episodes)
    _, rep_42 = terminal_verify(episodes, block_len=42)
    assert rep_default["block_len"] != 42          # default n**(1/3) rule
    assert rep_42["block_len"] == 42               # override honored
