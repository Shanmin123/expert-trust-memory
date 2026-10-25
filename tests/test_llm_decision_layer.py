"""Tests for the LLM-decision-layer comparison arm (experiments/llm_decision_layer.py).

Includes regression cases for every hole the 2026-07-20 external review found: flat-day
return validation, NaN rejection, duplicate dates, middle-gap omission, state-scoped
skill exposure, cache policy re-keying, and free-text skill injection.
"""
from __future__ import annotations

import json
import math

import pytest

from trading_agent.experiments.llm_decision_layer import (
    EXPORT_HEADER_KEY,
    PromptCache,
    SYSTEM_PROMPT,
    build_user_prompt,
    check_export_binding,
    derive_certified_skills,
    equal_committee_decisions,
    exposures_from_decisions,
    load_daily_views,
    load_producer_skills,
    metrics_from_series,
    net_returns,
    paired_mbb,
    parity_check,
    parse_decision,
    run_llm_decision_arm,
    series_from_exposures,
    skills_digest,
)


def _day(date="2024-07-09", tech="long", news="neutral", fund="flat",
         safe=(), reasoning="momentum positive", state="up_calm"):
    views = {}
    for expert_id, direction in (("technical", tech), ("news", news), ("fundamental", fund)):
        if direction is None:
            continue
        views[expert_id] = {
            "direction": direction,
            "confidence": 0.6,
            "safe_default": expert_id in safe,
            "reasoning": reasoning,
        }
    return {"date": date, "ticker": "ECL", "market_state": state, "views": views}


# --- prompt -----------------------------------------------------------------

def test_prompt_is_deterministic_and_orders_experts_canonically():
    day = _day()
    p1, p2 = build_user_prompt(day), build_user_prompt(day)
    assert p1 == p2
    assert p1.index("technical:") < p1.index("news:") < p1.index("fundamental:")
    assert "buy" in p1 and "sell" in p1 and "hold" in p1  # FinMem-style action vocabulary


def test_prompt_includes_skills_only_when_given():
    day = _day()
    bare = build_user_prompt(day)
    skill = (None, "Training-window verified finding: the fundamental expert reliably ADDED.")
    fed = build_user_prompt(day, (skill,))
    assert "Certified skills" not in bare
    assert skill[1] in fed and "Certified skills" in fed
    assert bare == build_user_prompt(day, ())


def test_state_scoped_skill_appears_only_in_its_state():
    """Review finding 6: a down_stress-certified skill must not treat other states."""
    skill = ("down_stress", "the technical expert reliably SUBTRACTED in down_stress.")
    calm_prompt = build_user_prompt(_day(state="up_calm"), (skill,))
    stress_prompt = build_user_prompt(_day(state="down_stress"), (skill,))
    assert skill[1] not in calm_prompt and "Certified skills" not in calm_prompt
    assert skill[1] in stress_prompt


def test_prompt_marks_safe_default_views_as_non_judgments():
    prompt = build_user_prompt(_day(safe=("news",)))
    assert "news: NO VIEW" in prompt
    assert "not a judgment" in prompt


# --- certified-skill derivation (no free text) ------------------------------

def _report(verified_cells, horizon=21):
    per_cell_g, per_cell_s = {}, {}
    for cell, mean_m, p, verified in verified_cells:
        state, _ = cell.split("|", 1)
        rec = {"mean_m": mean_m, "p_holm": p, "verified": verified}
        (per_cell_g if state == "global" else per_cell_s)[cell] = rec
    return {
        "manifest": {"attribution_horizon": horizon, "tickers": ["ECL"],
                     "start": "2011-07-01", "boundary": "2024-06-30", "end": "2025-12-31",
                     "prior_lessons_path": None,
                     "view_corpus": {"path": "reports/corpus/views_ECL__sealed.jsonl"}},
        "verify_gate_report": {
            "global_memory": {"per_cell": per_cell_g},
            "market_state_memory": {"per_cell": per_cell_s},
        },
    }


def test_derive_certified_skills_scopes_states_and_states_horizon_units():
    report = _report([
        ("global|fundamental", +7.8e-3, 0.016, True),
        ("global|technical", -7.9e-3, 0.016, True),
        ("global|news", +1e-4, 0.9, False),
        ("down_stress|technical", -3.6e-2, 0.099, True),
    ])
    skills = derive_certified_skills(report)
    assert len(skills) == 3
    scopes = [s for s, _ in skills]
    assert scopes.count(None) == 2 and "down_stress" in scopes
    for _, text in skills:
        assert "per 21-day attribution horizon" in text  # review finding 10: not "/day"
        assert "/day" not in text
    assert len(derive_certified_skills(_report([("global|news", 1e-4, 0.9, False)]))) == 0
    assert skills_digest(skills) != skills_digest(())


# --- export binding ---------------------------------------------------------

def test_check_export_binding_refuses_lesson_fed_or_mismatched_exports():
    man = {**_report([])["manifest"], "expert_model": "gpt-4.1-mini-2025-04-14",
           "expert_model_revision": "gpt-4.1-mini-2025-04-14", "vote_policy": "ternary"}
    good = {"prior_lessons_path": None, "view_corpus_path": man["view_corpus"]["path"],
            "tickers": ["ECL"], "start": "2011-07-01", "boundary": "2024-06-30",
            "end": "2025-12-31", "expert_model": man["expert_model"],
            "expert_model_revision": man["expert_model_revision"],
            "vote_policy": "ternary"}
    check_export_binding(good, man)
    with pytest.raises(AssertionError):
        check_export_binding({**good, "prior_lessons_path": "reports/lessons/x.json"}, man)
    with pytest.raises(AssertionError):
        check_export_binding({**good, "view_corpus_path": "other.jsonl"}, man)
    with pytest.raises(AssertionError):
        check_export_binding({**good, "tickers": ["TGT"]}, man)
    # review round-2 finding 5: model revision and vote policy are bound too
    with pytest.raises(AssertionError):
        check_export_binding({**good, "expert_model_revision": "other-rev"}, man)
    with pytest.raises(AssertionError):
        check_export_binding({**good, "vote_policy": "binary"}, man)


def test_load_producer_skills_requires_provenance(tmp_path):
    """Review round-2 (finding 4): bare skill lists are refused; provenance required."""
    p = tmp_path / "skills.json"
    p.write_text(json.dumps([["up_calm", "some skill"]]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_producer_skills(p)
    p.write_text(json.dumps({"producer": {}, "skills": [["up_calm", "s"]]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_producer_skills(p)
    p.write_text(json.dumps({
        "producer": {"kind": "llm", "model": "m", "evidence": "per-cell training stats"},
        "skills": [[None, "global skill"], ["down_stress", "state skill"]],
    }), encoding="utf-8")
    producer, skills = load_producer_skills(p)
    assert producer["kind"] == "llm"
    assert skills == ((None, "global skill"), ("down_stress", "state skill"))


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"decision": "buy", "rationale": "x"}', ("buy", False)),
    ('{"decision": "SELL", "rationale": "x"}', ("sell", False)),
    ('Sure! Here you go: {"decision": "hold", "rationale": "mixed"} Thanks.', ("hold", False)),
    ("no json at all", ("hold", True)),
    ('{"decision": "short everything"}', ("hold", True)),
    ("", ("hold", True)),
])
def test_parse_decision(text, expected):
    assert parse_decision(text) == expected


# --- position semantics -----------------------------------------------------

def test_exposures_follow_buy_sell_hold_with_maintain():
    decisions = ["hold", "buy", "hold", "hold", "sell", "hold", "buy"]
    assert exposures_from_decisions(decisions) == [0, 1, 1, 1, 0, 0, 1]


def test_exposures_reject_unknown_decision():
    with pytest.raises(ValueError):
        exposures_from_decisions(["buy", "moon"])


# --- series, costs, metrics -------------------------------------------------

def test_series_and_cost_model_match_portfolio_convention():
    dates = ["d1", "d2", "d3"]
    exposures = [1, 1, 0]
    nxt = {"d1": 0.01, "d2": -0.02, "d3": 0.05}
    series = series_from_exposures(dates, exposures, nxt)
    assert series == [["d1", 0.01, 1.0], ["d2", -0.02, 1.0], ["d3", 0.0, 0.0]]
    net = net_returns(series, cost_bps_per_side=10.0)
    assert net == pytest.approx([0.01 - 1e-3, -0.02, 0.0 - 1e-3])
    m = metrics_from_series(series, 10.0)
    assert m["n_dates"] == 3 and 0 < m["participation_rate"] < 1
    assert m["n_position_changes"] == 2
    assert m["position_changes_per_year"] == pytest.approx(2 * 252 / 3)


def test_series_rejects_missing_or_nan_returns():
    with pytest.raises(KeyError):
        series_from_exposures(["d1"], [1], {})
    with pytest.raises(KeyError):
        series_from_exposures(["d1"], [1], {"d1": float("nan")})


# --- equal-committee reconstruction and parity check ------------------------

def test_equal_committee_semantics_strict_boundaries():
    assert equal_committee_decisions(_day(tech="long", news="long", fund="flat")) == "buy"
    assert equal_committee_decisions(_day(tech="flat", news="neutral", fund="flat")) == "sell"
    assert equal_committee_decisions(_day(tech="long", news="neutral", fund="flat")) == "hold"
    assert equal_committee_decisions(_day(tech="long", news="flat", fund="flat",
                                          safe=("news", "fundamental"))) == "buy"
    assert equal_committee_decisions(_day(tech="long", news="long", fund="long",
                                          safe=("technical", "news", "fundamental"))) == "hold"


def _parity_fixture():
    days = [
        _day(date="d1", tech="long", news="long", fund="flat"),   # buy -> 1
        _day(date="d2", tech="flat", news="flat", fund="long"),   # sell -> 0
        _day(date="d3", tech="long", news="neutral", fund="flat"),  # hold -> stays 0
        _day(date="d4", tech="long", news="long", fund="long"),   # tail beyond canonical
    ]
    nxt = {"d1": 0.02, "d2": 0.01, "d3": -0.01, "d4": 0.005}
    canonical = {
        "equal_weight": [["d1", 0.02, 1.0], ["d2", 0.0, 0.0], ["d3", 0.0, 0.0]],
        "global_memory": [["d1", 0.02, 1.0], ["d2", 0.01, 1.0], ["d3", 0.0, 0.0]],
    }
    return days, canonical, nxt, ["d1", "d2", "d3", "d4"]


def test_parity_check_passes_on_consistent_series():
    days, canonical, nxt, price_dates = _parity_fixture()
    out = parity_check(days, canonical, nxt, price_dates=price_dates)
    assert out["checked_days"] == 3 and out["mismatches"] == 0
    # d2 return validated via the global arm even though equal was flat there
    assert out["long_rows_return_checked"] == 3


def test_parity_check_catches_wrong_return_on_equal_flat_day():
    """Review finding 1: equal flat on d2 must not hide a wrong return when another
    canonical arm was long there."""
    days, canonical, nxt, price_dates = _parity_fixture()
    bad = dict(nxt)
    bad["d2"] = 0.02  # global arm recorded 0.01
    with pytest.raises(AssertionError):
        parity_check(days, canonical, bad, price_dates=price_dates)


def test_parity_check_rejects_nan_and_implausible_returns():
    days, canonical, nxt, price_dates = _parity_fixture()
    for poison in (float("nan"), 0.9):
        bad = dict(nxt)
        bad["d3"] = poison  # nobody long on d3 -> only the sanity net can catch it
        with pytest.raises(AssertionError):
            parity_check(days, canonical, bad, price_dates=price_dates)


def test_parity_check_rejects_middle_gap_and_duplicates():
    days, canonical, nxt, price_dates = _parity_fixture()
    gapped = {k: [row for row in v if row[0] != "d2"] for k, v in canonical.items()}
    with pytest.raises(AssertionError):
        parity_check(days, gapped, nxt, price_dates=price_dates)  # d1,d3 not a prefix
    dup_days = days + [days[0]]
    with pytest.raises(AssertionError):
        parity_check(dup_days, canonical, nxt, price_dates=price_dates)


def test_parity_check_rejects_non_adjacent_price_dates():
    days, canonical, nxt, _ = _parity_fixture()
    with pytest.raises(AssertionError):
        parity_check(days, canonical, nxt, price_dates=["d1", "d1b", "d2", "d3", "d4"])


def test_parity_check_rejects_exposure_skew():
    days, canonical, nxt, price_dates = _parity_fixture()
    skewed = {**canonical,
              "equal_weight": [["d1", 0.02, 1.0], ["d2", 0.01, 1.0], ["d3", 0.0, 0.0]]}
    with pytest.raises(AssertionError):
        parity_check(days, skewed, nxt, price_dates=price_dates)


def test_parity_check_is_fail_closed_on_malformed_canonical_rows():
    """Review round-2 (new finding 3): fractional exposures, NaN gross, and nonzero
    gross on flat rows must all fail, not slip through the long-row-only check."""
    days, canonical, nxt, price_dates = _parity_fixture()
    frac = {**canonical,
            "global_memory": [["d1", 0.02, 0.5], ["d2", 0.01, 1.0], ["d3", 0.0, 0.0]]}
    with pytest.raises(AssertionError):
        parity_check(days, frac, nxt, price_dates=price_dates)
    nan_gross = {**canonical,
                 "global_memory": [["d1", float("nan"), 0.0], ["d2", 0.01, 1.0], ["d3", 0.0, 0.0]]}
    with pytest.raises(AssertionError):
        parity_check(days, nan_gross, nxt, price_dates=price_dates)
    flat_nonzero = {**canonical,
                    "global_memory": [["d1", 0.02, 1.0], ["d2", 0.01, 1.0], ["d3", 0.003, 0.0]]}
    with pytest.raises(AssertionError):
        parity_check(days, flat_nonzero, nxt, price_dates=price_dates)


# --- paired within-LLM-decision-layer primary ----------------------------------------

def test_paired_mbb_detects_shift_and_handles_zero_heavy_series():
    import random
    rng = random.Random(7)
    null = [rng.gauss(0.0, 1e-3) for _ in range(300)]
    shifted = [d + 5e-4 for d in null]
    p_null = paired_mbb(null, block_len=21, n_boot=800)["p_one_sided"]
    p_shift = paired_mbb(shifted, block_len=21, n_boot=800)["p_one_sided"]
    assert p_shift < 0.05 < p_null
    # zero-heavy (ADSK-like: skill shown only in one state) stays valid, not degenerate
    sparse = [0.0] * 280 + [2e-3] * 20
    out = paired_mbb(sparse, block_len=21, n_boot=800)
    assert out["n_nonzero_diffs"] == 20
    assert 0.0 < out["p_one_sided"] <= 1.0
    # all-zero difference series: mean 0, p must not crash and must not claim an effect
    degenerate = paired_mbb([0.0] * 100, block_len=21, n_boot=200)
    assert degenerate["mean_daily_diff"] == 0.0
    assert degenerate["p_one_sided"] > 0.5


def test_paired_mbb_rejects_bad_inputs():
    with pytest.raises(ValueError):
        paired_mbb([], block_len=7)
    with pytest.raises(ValueError):
        paired_mbb([0.1, 0.2], block_len=3)


def _decision_arm_report(name="ECL", arm="llm_decision_certified_skills", skills_mode="certified",
                       n_skills=2, cache="cache_shared.jsonl", dates=("d1", "d2")):
    series = [[d, 0.01, 1.0] for d in dates]
    return {
        "name": name,
        "arm": arm,
        "manifest": {"skills_mode": skills_mode, "n_skills": n_skills, "cache": cache,
                     "views": "v.jsonl", "canonical_report": "c.json",
                     "cost_bps_per_side": 10.0, "model": "m", "skills_sha": "abc"},
        "daily_arm_series": {arm: series, "equal_weight": series},
    }


def test_paired_analysis_preconditions():
    """Review round-2 (new findings 1-2): the paired primary must refuse mismatched
    names, un-shared caches, wrong arm roles, and differing date vectors."""
    from scripts.llm_decision_paired_analysis import check_pair, paired_diffs

    treated = _decision_arm_report()
    control = _decision_arm_report(arm="llm_decision_no_skills", skills_mode="none", n_skills=0)
    check_pair("ECL", treated, control)
    dates, diffs = paired_diffs(treated, control)
    assert dates == ["d1", "d2"] and diffs == [0.0, 0.0]

    with pytest.raises(AssertionError):
        check_pair("TGT", treated, control)
    with pytest.raises(AssertionError):  # control secretly has skills
        check_pair("ECL", treated, _decision_arm_report(arm="llm_decision_no_skills",
                                                      skills_mode="certified", n_skills=1))
    with pytest.raises(AssertionError):  # treated has none
        check_pair("ECL", control, control)
    with pytest.raises(AssertionError):  # different caches -> different temp-0 draws
        check_pair("ECL", treated, _decision_arm_report(arm="llm_decision_no_skills",
                                                      skills_mode="none", n_skills=0,
                                                      cache="other_cache.jsonl"))
    with pytest.raises(AssertionError):  # date vectors differ
        check_pair("ECL", treated, _decision_arm_report(arm="llm_decision_no_skills",
                                                      skills_mode="none", n_skills=0,
                                                      dates=("d1", "d3")))


# --- cache and arm runner ---------------------------------------------------

def test_cache_roundtrip_and_replay_forbids_spend(tmp_path):
    days = [
        _day(date="d1", tech="long", news="long", fund="flat"),
        _day(date="d2", tech="flat", news="flat", fund="long"),
    ]
    nxt = {"d1": 0.02, "d2": 0.01}
    cache_path = tmp_path / "llm_decision_cache.jsonl"
    calls = []

    def fake_call(system_prompt, user_prompt):
        calls.append(user_prompt)
        return '{"decision": "buy", "rationale": "test"}'

    cache = PromptCache(cache_path, model="test-model")
    run1 = run_llm_decision_arm(days, (), cache, fake_call, nxt, arm="no_skills")
    assert run1.n_api_calls == 2 and run1.n_cache_hits == 0
    assert [row[2] for row in run1.series] == [1.0, 1.0]

    cache2 = PromptCache(cache_path, model="test-model")
    run2 = run_llm_decision_arm(days, (), cache2, None, nxt, arm="no_skills")
    assert run2.n_api_calls == 0 and run2.n_cache_hits == 2
    assert run2.series == run1.series
    assert len(calls) == 2

    cache3 = PromptCache(cache_path, model="test-model")
    with pytest.raises(RuntimeError):
        run_llm_decision_arm(days, (("up_calm", "some certified skill"),), cache3, None, nxt,
                           arm="with_skills")


def test_cache_rekeys_on_policy_change(tmp_path):
    """Review finding 7: rows recorded under a different policy must not replay."""
    path = tmp_path / "c.jsonl"
    old = PromptCache(path, model="m-old")
    old.put("prompt-x", '{"decision": "buy", "rationale": "old"}')
    fresh_same = PromptCache(path, model="m-old")
    assert fresh_same.get("prompt-x") is not None and fresh_same.n_stale_rows == 0
    fresh_other_model = PromptCache(path, model="m-new")
    assert fresh_other_model.get("prompt-x") is None
    assert fresh_other_model.n_stale_rows == 1
    fresh_other_system = PromptCache(path, model="m-old", system_prompt=SYSTEM_PROMPT + " v2")
    assert fresh_other_system.get("prompt-x") is None
    assert fresh_other_system.n_stale_rows == 1
    fresh_other_temperature = PromptCache(path, model="m-old", temperature=0.7)
    assert fresh_other_temperature.get("prompt-x") is None
    assert fresh_other_temperature.n_stale_rows == 1


def test_parse_errors_maintain_position(tmp_path):
    days = [
        _day(date="d1", tech="long", news="long", fund="long"),
        _day(date="d2", tech="long", news="long", fund="long"),
    ]
    nxt = {"d1": 0.02, "d2": 0.01}
    replies = iter(['{"decision": "buy", "rationale": "x"}', "garbled output"])

    def flaky_call(system_prompt, user_prompt):
        return next(replies)

    cache = PromptCache(tmp_path / "c.jsonl", model="m")
    run = run_llm_decision_arm(days, (), cache, flaky_call, nxt, arm="no_skills")
    assert run.n_parse_errors == 1
    assert [row[2] for row in run.series] == [1.0, 1.0]


# --- views export loader ----------------------------------------------------

def _write_export(path, rows, header=None):
    header = header or {"view_corpus_path": "c.jsonl", "prior_lessons_path": None,
                        "tickers": ["ECL"], "start": "2011-07-01",
                        "boundary": "2024-06-30", "end": "2025-12-31", "run_label": "X"}
    lines = [json.dumps({EXPORT_HEADER_KEY: header})]
    lines += [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_daily_views_requires_header_sorts_and_rejects_dupes(tmp_path):
    p = tmp_path / "views.jsonl"
    rows = [
        {"date": "2024-07-10", "ticker": "ECL", "market_state": "up_calm", "views": {}},
        {"date": "2024-07-09", "ticker": "ECL", "market_state": "up_calm", "views": {}},
    ]
    _write_export(p, rows)
    header, days = load_daily_views(p)
    assert header["prior_lessons_path"] is None
    assert [d["date"] for d in days] == ["2024-07-09", "2024-07-10"]

    p2 = tmp_path / "noheader.jsonl"
    p2.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_daily_views(p2)

    _write_export(p, rows + [rows[0]])
    with pytest.raises(ValueError):
        load_daily_views(p)

    _write_export(p, [rows[0], {**rows[1], "ticker": "TGT"}])
    with pytest.raises(ValueError):
        load_daily_views(p)
