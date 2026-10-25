"""LLM skill producer: the skill route's own learning step (decision-layer-free evidence).

Reads the SAME per-cell training evidence the statistical verifier consumed — the
peer-relative marginal record m per (market_state, expert) cell: mean per attribution
horizon, standard error, sample count — with the verifier's VERDICT FIELDS STRIPPED
(no p-values, no verified flags, no Holm ranks), and asks one temperature-0 LLM call to
judge which cells, if any, deserve to be written as skills. The comparison against the
certified transcription therefore isolates the JUDGE (bootstrap test vs LLM) with the
evidence held identical; both sides' evidence is decision-layer-free (expert calls +
market returns + peer average; no weights, no theta, no positions).

Discipline:
- Producer runs and its output is FROZEN before any decision-layer arm result exists
  (chain ordering), so skills cannot be selected on outcomes.
- Grid-invariance contract inherited from reflection: the prompt carries only raw
  per-cell statistics — no tau/theta quantities, no portfolio outcomes, no test data.
- Parse failure is LOUD (raises); a silent empty skill list would degrade the producer
  arm into the no-skill arm without anyone noticing.
- Calls cached with the same record-once/replay-free PromptCache as the decision layer.

Usage:
    PYTHONPATH=src python scripts/produce_llm_skills.py \
        --canonical-report reports/final/ECL__sealed.json \
        --cache reports/corpus/llm_skill_producer.jsonl \
        --out reports/lessons/ECL_llm_producer.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from trading_agent.experiments.llm_decision_layer import PromptCache, skills_digest

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_SKILLS = 6

PRODUCER_SYSTEM_PROMPT = (
    "You are reviewing the training-window track record of a three-expert stock "
    "committee (technical, news, fundamental). For each (market_state, expert) cell you "
    "get the PEER-RELATIVE marginal record of that expert's spoken calls: the mean m per "
    "attribution horizon (positive = the expert's calls added edge over the average of "
    "the other experts' calls), its standard error, and the sample count. Decide which "
    "cells, if any, represent reliable skill worth stating as prior context for a "
    "decision maker. Be skeptical: write NONE if the evidence looks like noise. "
    "Output strict JSON only."
)


def evidence_cells(report: dict) -> list[dict]:
    """Per-cell peer-relative evidence with the verifier's verdict fields stripped."""
    verification = report.get("verify_gate_report") or {}
    cells: list[dict] = []
    for family in ("global_memory", "market_state_memory"):
        per_cell = (verification.get(family) or {}).get("per_cell") or {}
        for cell, c in sorted(per_cell.items()):
            state, expert = cell.split("|", 1)
            cells.append({
                "scope": state,          # 'global' or a market_state
                "expert": expert,
                "mean_m": float(c["mean_m"]),
                "se": float(c["se"]),
                "n": int(c["n"]),
            })
    return cells


def build_producer_prompt(ticker: str, horizon: int, cells: list[dict],
                          max_skills: int = MAX_SKILLS) -> str:
    lines = [
        f"ticker: {ticker}",
        f"attribution horizon: {horizon} trading days (all mean_m values are per this horizon)",
        "",
        "Per-cell peer-relative training record:",
    ]
    for c in cells:
        lines.append(
            f"- scope={c['scope']} expert={c['expert']} "
            f"mean_m={c['mean_m']:+.2e} se={c['se']:.2e} n={c['n']}"
        )
    lines += [
        "",
        f"Write AT MOST {max_skills} skills; an empty list is a fully acceptable answer.",
        'Each skill: {"scope": "global" | "<market_state>", "text": "<one plain sentence '
        'stating the finding>"}. scope must be either "global" or exactly the market_state '
        "of the cell the skill is about.",
        'Your output must strictly conform to: {"skills": [ ... ]} with no additional content.',
    ]
    return "\n".join(lines)


def parse_producer_response(text: str, valid_states: set[str]) -> tuple[tuple[str | None, str], ...]:
    snippet = text.strip()
    start, end = snippet.find("{"), snippet.rfind("}")
    if start >= 0 and end > start:
        snippet = snippet[start:end + 1]
    payload = json.loads(snippet)  # loud on failure by design
    raw = payload["skills"]
    if not isinstance(raw, list) or len(raw) > MAX_SKILLS:
        raise ValueError(f"producer returned invalid skills payload: {raw!r}")
    out: list[tuple[str | None, str]] = []
    for item in raw:
        scope = item["scope"]
        text_ = str(item["text"]).strip()
        if not text_:
            raise ValueError("producer wrote an empty skill text")
        if scope == "global":
            out.append((None, text_))
        elif scope in valid_states:
            out.append((scope, text_))
        else:
            raise ValueError(f"producer used invalid scope {scope!r}")
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-report", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args(argv)

    try:
        from data_collection.config import load_env_file
        load_env_file()
    except ImportError:
        pass

    report = json.loads(Path(args.canonical_report).read_text(encoding="utf-8"))
    ticker = (report["manifest"].get("tickers") or ["?"])[0]
    horizon = int(report["manifest"].get("attribution_horizon") or 1)
    cells = evidence_cells(report)
    if not cells:
        raise SystemExit(f"{args.canonical_report}: no tested cells to judge")
    valid_states = {c["scope"] for c in cells if c["scope"] != "global"}

    prompt = build_producer_prompt(ticker, horizon, cells)
    for banned in ("p_holm", "verified", "holm", "p="):
        if banned in prompt:
            raise SystemExit(f"evidence leak: {banned!r} in producer prompt")

    cache = PromptCache(args.cache, args.model, system_prompt=PRODUCER_SYSTEM_PROMPT)
    response = cache.get(prompt)
    if response is None:
        if args.no_api:
            raise SystemExit("cache miss with API calls forbidden")
        from trading_agent.experts.backends import OpenAIChatBackend
        backend = OpenAIChatBackend(model=args.model, temperature=0.0)
        response = backend.call(PRODUCER_SYSTEM_PROMPT, prompt)
        cache.put(prompt, response)

    skills = parse_producer_response(response, valid_states)
    payload = {
        "producer": {
            "kind": "llm",
            "model": args.model,
            "temperature": 0.0,
            "evidence": "per-cell peer-relative m mean/se/n from "
                        f"{args.canonical_report} (verifier verdict fields stripped)",
            "canonical_report": args.canonical_report,
            "n_cells_shown": len(cells),
            "skills_sha": skills_digest(skills),
            "produced": _dt.date.today().isoformat(),
        },
        "skills": [list(s) for s in skills],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{ticker}: producer wrote {len(skills)} skills -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
