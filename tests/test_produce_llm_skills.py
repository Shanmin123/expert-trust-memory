"""LLM skill producer: evidence hygiene, parsing, and output-format contract."""
from __future__ import annotations

import json

import pytest

from scripts.produce_llm_skills import (
    build_producer_prompt,
    evidence_cells,
    parse_producer_response,
)
from trading_agent.experiments.llm_decision_layer import load_producer_skills


_REPORT = {
    "manifest": {"tickers": ["ECL"], "attribution_horizon": 21},
    "verify_gate_report": {
        "global_memory": {"per_cell": {
            "global|fundamental": {"mean_m": 7.8e-03, "se": 2.1e-03, "n": 340,
                                   "p": 0.001, "p_holm": 0.016, "verified": True,
                                   "holm_rank": 1, "t": 3.7},
            "global|news": {"mean_m": 3.9e-04, "se": 2.0e-03, "n": 340,
                            "p": 0.8, "p_holm": 0.9, "verified": False},
        }},
        "market_state_memory": {"per_cell": {
            "down_stress|technical": {"mean_m": -3.5e-02, "se": 5.5e-03, "n": 44,
                                      "p": 0.01, "p_holm": 0.099, "verified": True},
        }},
    },
}


def test_evidence_cells_strip_verdict_fields_and_prompt_has_no_leak():
    cells = evidence_cells(_REPORT)
    assert len(cells) == 3
    for c in cells:
        assert set(c) == {"scope", "expert", "mean_m", "se", "n"}
    prompt = build_producer_prompt("ECL", 21, cells)
    assert prompt == build_producer_prompt("ECL", 21, cells)  # deterministic
    for banned in ("p_holm", "verified", "holm", "p=", "0.016", "0.099"):
        assert banned not in prompt
    assert "mean_m=+7.80e-03" in prompt and "n=340" in prompt
    assert "21 trading days" in prompt


def test_parse_maps_global_scope_and_validates_states():
    reply = json.dumps({"skills": [
        {"scope": "global", "text": "fundamental adds edge over peers."},
        {"scope": "down_stress", "text": "technical subtracts edge in down_stress."},
    ]})
    skills = parse_producer_response(reply, valid_states={"down_stress"})
    assert skills == ((None, "fundamental adds edge over peers."),
                      ("down_stress", "technical subtracts edge in down_stress."))
    assert parse_producer_response('{"skills": []}', valid_states=set()) == ()


def test_parse_is_loud_on_garbage_bad_scope_and_empty_text():
    with pytest.raises(json.JSONDecodeError):
        parse_producer_response("not json at all", valid_states=set())
    with pytest.raises(ValueError):
        parse_producer_response(json.dumps({"skills": [
            {"scope": "up_calm", "text": "x"}]}), valid_states={"down_stress"})
    with pytest.raises(ValueError):
        parse_producer_response(json.dumps({"skills": [
            {"scope": "global", "text": "  "}]}), valid_states=set())
    too_many = {"skills": [{"scope": "global", "text": f"s{i}"} for i in range(9)]}
    with pytest.raises(ValueError):
        parse_producer_response(json.dumps(too_many), valid_states=set())


def test_output_format_roundtrips_through_the_decision_layer_loader(tmp_path):
    payload = {
        "producer": {"kind": "llm", "model": "m", "temperature": 0.0,
                     "evidence": "per-cell peer-relative m mean/se/n"},
        "skills": [[None, "global skill"], ["down_stress", "state skill"]],
    }
    p = tmp_path / "ECL_llm_producer.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    producer, skills = load_producer_skills(p)
    assert producer["kind"] == "llm"
    assert skills == ((None, "global skill"), ("down_stress", "state skill"))
