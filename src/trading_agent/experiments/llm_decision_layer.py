"""LLM decision layer comparison arm: certified skills enter at the decision-integration layer.

The main system's decision layer is numeric (certified weight tilts over ternary votes), so
a text-form certified skill has no native input there; feeding skills into EXPERT prompts
double-counts the same differential the weights already correct.
This module evaluates the other insertion point: a stateless LLM decision layer that reads the
three experts' stated views (direction + one-line reasoning) — optionally plus the frozen
certified skills — and emits the committee-level decision itself. No weight layer exists in
this arm; that absence is the comparison's content, not an omission.

Reference contracts (project rule: open-source reuse first, both already cited systems):
- TradingAgents role structure: analysts -> a trader who reads their reports and issues the
  final decision. The LLM decision layer is that trader role over this project's expert views.
- FinMem decision prompt (FINSABER-bundled: external/FINSABER/llm_traders/finmem/puppy/
  prompts.py): layered context injection, strict-JSON output, buy/sell/hold vocabulary with
  hold as the hard-to-decide fallback. Deliberate deviation: FinMem shows the model its
  current holdings; this LLM decision layer is position-blind so its information set matches the
  numeric layer it replaces (maintain semantics live in the downstream rule, as in the
  main system's strict boundaries: buy = long, sell = cash, hold = keep current position).

Discipline notes (hardened after the 2026-07-20 external review):
- Expert views come from the frozen corpus via the equal arm's daily-view export; the
  export carries a header binding it to its source run, and the runner refuses exports
  produced with prior lessons in the prompts.
- Certified skills are DERIVED from the canonical report's verification receipts inside
  this module (no free-text skill input for the certified arm), carry an explicit state
  scope, and are shown only in their certified market_state — mirroring the numeric
  certified-only retrieval, which deviates from equal only inside certified cells.
- The parity check must reproduce the canonical equal arm exactly AND validates the
  price-derived next-bar returns on every date any canonical arm was long, rejects NaN,
  duplicate dates, and any non-suffix date omission, and verifies price-calendar
  adjacency so an off-by-one return alignment cannot hide behind flat days.
- LLM decision layer calls are cached keyed by (model, temperature, system prompt, user prompt);
  cached rows are re-keyed on load so a policy change can never silently replay stale
  decisions.
- The LLM decision layer is stateless and reads only frozen training-window receipts, so a
  test-window evaluation is a fixed-policy replay, not test-window learning.
- Every series/metric in the comparison table is computed by THIS module's functions for
  both the LLM decision layer and the canonical arms (352 attributed-day same-basis, labeled as
  such; not interchangeable with the reports' 373-day headline metrics).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

TRADING_DAYS = 252

SYSTEM_PROMPT = (
    "You are the decision LLM decision layer of a three-expert stock committee. The experts have "
    "already spoken; your job is to integrate their stated views into one committee "
    "decision for the next trading bar. The 'date' field is today: treat it as now, and "
    "use only information available on or before it."
)

DECISION_INSTRUCTION = (
    "You should provide exactly one of the following investment decisions: buy or sell. "
    "When it is really hard to make a 'buy'-or-'sell' decision, you could go with 'hold'. "
    "buy = hold a long position for the next bar; sell = hold cash; hold = keep the "
    "current position unchanged. Your output should strictly conform to the following "
    "json format without any additional contents: "
    '{"decision": "buy" | "sell" | "hold", "rationale": "<one short sentence>"}'
)

_DIRECTION_LABEL = {
    "long": "LONG",
    "flat": "FLAT (bearish: prefers cash)",
    "neutral": "NEUTRAL (deliberate abstention: no view)",
}

EXPORT_HEADER_KEY = "__export_manifest__"


# --------------------------------------------------------------------------- inputs


def load_daily_views(path: str | Path) -> tuple[dict, list[dict]]:
    """Load the launcher's --export-daily-views JSONL -> (header, rows sorted by date).

    The first line must be the export manifest header (source binding); rows must be a
    single ticker with unique dates.
    """
    lines = [json.loads(line) for line in
             Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or EXPORT_HEADER_KEY not in lines[0]:
        raise ValueError(f"{path}: missing export manifest header line")
    header = lines[0][EXPORT_HEADER_KEY]
    rows = lines[1:]
    rows.sort(key=lambda r: r["date"])
    tickers = {r["ticker"] for r in rows}
    if len(tickers) != 1:
        raise ValueError(f"expected a single-name export, got tickers {sorted(tickers)}")
    dates = [r["date"] for r in rows]
    if len(set(dates)) != len(dates):
        dupes = sorted({d for d in dates if dates.count(d) > 1})[:3]
        raise ValueError(f"duplicate dates in export, e.g. {dupes}")
    return header, rows


def check_export_binding(header: Mapping, canonical_manifest: Mapping) -> None:
    """Refuse exports that do not come from the canonical run's exact configuration.

    Binds: view corpus path, tickers, window, and (critically) that the exporting run fed
    NO prior lessons into expert prompts — a lesson-fed export would silently change the
    reasoning text the LLM decision layer reads while leaving directions (and parity) intact.
    """
    if header.get("prior_lessons_path") is not None:
        raise AssertionError(
            f"export was produced with prior_lessons_path={header['prior_lessons_path']!r}; "
            "the LLM decision layer requires lesson-free expert prompts"
        )
    canon_corpus = (canonical_manifest.get("view_corpus") or {}).get("path")
    if header.get("view_corpus_path") != canon_corpus:
        raise AssertionError(
            f"export corpus {header.get('view_corpus_path')!r} != canonical run corpus "
            f"{canon_corpus!r}"
        )
    for key in ("tickers", "start", "boundary", "end",
                "expert_model", "expert_model_revision", "vote_policy"):
        if header.get(key) != canonical_manifest.get(key):
            raise AssertionError(
                f"export {key}={header.get(key)!r} != canonical {canonical_manifest.get(key)!r}"
            )


def derive_certified_skills(report: Mapping) -> tuple[tuple[str | None, str], ...]:
    """Transcribe the canonical report's VERIFIED cells into (state_scope, sentence) skills.

    The certified arm's skills are derived here, from the verification receipts, never
    accepted as free text — so a wrong-name or test-informed sentence cannot enter.
    State-family skills carry their state as scope and are only shown in that state,
    mirroring numeric certified-only retrieval. The magnitude is stated per attribution
    horizon (NOT per day; the h=21 marginal proxy is a monthly-horizon quantity).
    """
    horizon = int((report.get("manifest") or {}).get("attribution_horizon") or 1)
    verification = report.get("verify_gate_report") or {}
    out: list[tuple[str | None, str]] = []
    for family, scope_of in (("global_memory", lambda s: None),
                             ("market_state_memory", lambda s: s)):
        per_cell = (verification.get(family) or {}).get("per_cell") or {}
        for cell, c in sorted(per_cell.items()):
            if not c.get("verified"):
                continue
            state, expert = cell.split("|", 1)
            scope = scope_of(state)
            where = "across all market states" if scope is None else f"in the {scope} market state"
            direction = ("reliably ADDED edge over" if c["mean_m"] > 0
                         else "reliably SUBTRACTED edge versus")
            out.append((scope, (
                f"Training-window verified finding ({where}): the {expert} expert's "
                f"spoken judgments {direction} the average of the other experts' calls "
                f"(peer-relative marginal contribution {c['mean_m']:+.1e} per {horizon}-day attribution "
                f"horizon, Holm-adjusted p={c['p_holm']:.3f}). Treat this as certified "
                f"context, not an order."
            )))
    return tuple(out)


def skills_digest(skills: Sequence[tuple[str | None, str]]) -> str:
    payload = json.dumps(list(skills), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_producer_skills(path: str | Path) -> tuple[dict, tuple[tuple[str | None, str], ...]]:
    """Load an alternative-producer skill file (e.g. LLM-written skills).

    Free-text lists are refused: the file must be an object carrying non-empty
    'producer' provenance (who/what generated these, from which evidence, when)
    plus 'skills' as [[scope, text], ...]. Returns (producer manifest, skills).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or not isinstance(raw.get("producer"), dict)
            or not raw["producer"] or not isinstance(raw.get("skills"), list)):
        raise ValueError(
            f"{path}: producer skill files must be an object with non-empty 'producer' "
            "provenance and 'skills' [[scope, text], ...]; bare lists are refused"
        )
    skills = tuple((scope, str(text)) for scope, text in raw["skills"])
    return dict(raw["producer"]), skills


def paired_mbb(diffs: Sequence[float], block_len: int, n_boot: int = 5000,
               seed: int = 20260716) -> dict:
    """Centered, plus-one-corrected one-sided moving-block bootstrap on a paired
    daily-difference series (treated minus control net returns).

    H0: mean difference <= 0. The series is centered at its own mean before block
    resampling, and p = (1 + #{resampled mean >= observed mean}) / (n_boot + 1) — a
    calibrated bootstrap p-value, not an uncentered tail proportion (external review
    2026-07-20, new finding 4). Zero-heavy series (identical prompts -> identical
    cached decisions on most days) stay valid: zeros are resampled like any date and
    concentrate the null, they do not degenerate it.
    """
    import random as _random

    n = len(diffs)
    if n == 0:
        raise ValueError("empty difference series")
    if block_len < 1 or block_len > n:
        raise ValueError(f"block_len {block_len} out of range for n={n}")
    observed = sum(diffs) / n
    centered = [d - observed for d in diffs]
    rng = _random.Random(seed)
    n_blocks = (n + block_len - 1) // block_len
    max_start = n - block_len
    hits = 0
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n_blocks):
            start = rng.randint(0, max_start)
            sample.extend(centered[start:start + block_len])
        boot_mean = sum(sample[:n]) / n
        if boot_mean >= observed:
            hits += 1
    return {
        "mean_daily_diff": observed,
        "p_one_sided": (hits + 1) / (n_boot + 1),
        "block_len": block_len,
        "n_boot": n_boot,
        "n_dates": n,
        "n_nonzero_diffs": sum(1 for d in diffs if d != 0.0),
        "seed": seed,
    }


# --------------------------------------------------------------------------- prompt


def _view_line(expert_id: str, view: Mapping) -> str:
    direction = str(view.get("direction", "neutral"))
    label = _DIRECTION_LABEL.get(direction, direction.upper())
    if view.get("safe_default"):
        return f"- {expert_id}: NO VIEW (data unavailable or parse error; not a judgment)."
    reasoning = " ".join(str(view.get("reasoning", "")).split())
    if len(reasoning) > 280:
        reasoning = reasoning[:277] + "..."
    tail = f" Reasoning: {reasoning}" if reasoning else ""
    return f"- {expert_id}: {label}.{tail}"


def build_user_prompt(day: Mapping, skills: Sequence[tuple[str | None, str]] = ()) -> str:
    """Deterministic LLM decision layer prompt for one decision date.

    Expert order is fixed (technical, news, fundamental — the committee's canonical
    order) so the prompt is byte-stable for caching. A skill with a state scope appears
    ONLY on days in that market_state (numeric certified-only parity).
    """
    lines = [
        f"date: {day['date']}",
        f"ticker: {day['ticker']}",
        f"market_state: {day['market_state']} (causal trend-x-risk label from the market proxy)",
        "",
        "Expert views:",
    ]
    views = day.get("views", {})
    ordered = [e for e in ("technical", "news", "fundamental") if e in views]
    ordered += sorted(set(views) - set(ordered))
    for expert_id in ordered:
        lines.append(_view_line(expert_id, views[expert_id]))
    shown = [text for scope, text in skills
             if scope is None or scope == day.get("market_state")]
    if shown:
        lines.append("")
        lines.append(
            "Certified skills (verified on the training window by a statistical "
            "verification; context for integration, not orders):"
        )
        for text in shown:
            lines.append(f"- {text}")
    lines.append("")
    lines.append(DECISION_INSTRUCTION)
    return "\n".join(lines)


def parse_decision(text: str) -> tuple[str, bool]:
    """Parse the LLM decision layer's reply -> (decision, parse_error).

    Accepts strict JSON or a reply containing one JSON object. Unknown/missing decision
    falls back to 'hold' (keep current position) with parse_error=True — the LLM decision layer-
    level mirror of the main system's boundary rule that a non-judgment maintains.
    """
    snippet = text.strip()
    start, end = snippet.find("{"), snippet.rfind("}")
    if start >= 0 and end > start:
        snippet = snippet[start:end + 1]
    try:
        payload = json.loads(snippet)
        decision = str(payload.get("decision", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError, TypeError):
        decision = ""
    if decision in ("buy", "sell", "hold"):
        return decision, False
    return "hold", True


# --------------------------------------------------------------------------- positions


def exposures_from_decisions(decisions: Sequence[str], start_exposure: int = 0) -> list[int]:
    """buy -> 1, sell -> 0, hold -> previous (start flat, like every arm)."""
    out: list[int] = []
    position = int(start_exposure)
    for decision in decisions:
        if decision == "buy":
            position = 1
        elif decision == "sell":
            position = 0
        elif decision != "hold":
            raise ValueError(f"unknown decision {decision!r}")
        out.append(position)
    return out


def series_from_exposures(dates: Sequence[str], exposures: Sequence[int],
                          next_returns: Mapping[str, float]) -> list[list]:
    """[iso_date, gross, exposure] rows in the canonical daily_arm_series convention:
    exposure = action taken on the date, gross = the next-bar return that action earned
    (0.0 when flat)."""
    rows = []
    for date, exposure in zip(dates, exposures):
        r = next_returns.get(date)
        if r is None or not math.isfinite(r):
            raise KeyError(f"no finite next-bar return for decision date {date}")
        rows.append([date, float(exposure) * float(r), float(exposure)])
    return rows


def net_returns(series: Sequence[Sequence], cost_bps_per_side: float) -> list[float]:
    """Identical cost model to scripts/portfolio_arm_metrics.py: charge
    bps/1e4 * |exposure_t - exposure_{t-1}| against the gross return."""
    out = []
    prev_exposure = 0.0
    for _, gross, exposure in series:
        cost = cost_bps_per_side / 1e4 * abs(float(exposure) - prev_exposure)
        out.append(float(gross) - cost)
        prev_exposure = float(exposure)
    return out


def metrics_from_series(series: Sequence[Sequence], cost_bps_per_side: float = 10.0) -> dict:
    net = net_returns(series, cost_bps_per_side)
    n = len(net)
    if n == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "n_dates": 0}
    mean = sum(net) / n
    var = sum((x - mean) ** 2 for x in net) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    sharpe = (mean / std) * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    curve, peak, mdd = 1.0, 1.0, 0.0
    for x in net:
        curve *= (1.0 + x)
        peak = max(peak, curve)
        mdd = max(mdd, 1.0 - curve / peak)
    exposure_days = sum(1 for row in series if float(row[2]) > 0)
    prev = 0.0
    n_position_changes = 0
    for row in series:
        if float(row[2]) != prev:
            n_position_changes += 1
            prev = float(row[2])
    return {
        "sharpe": sharpe,
        "total_return": curve - 1.0,
        "max_drawdown": mdd,
        "n_dates": n,
        "participation_rate": exposure_days / n,
        "n_position_changes": n_position_changes,
        "position_changes_per_year": n_position_changes * TRADING_DAYS / n,
    }


# --------------------------------------------------------------------------- parity


def equal_committee_decisions(day: Mapping) -> str:
    """Reproduce the equal-weight committee's per-date action signal from exported views:
    A = mean of +/-1 over available (non-neutral, non-safe-default) experts; buy when
    A > 0, sell when A < 0, hold at exactly 0 (strict boundaries, theta = 0)."""
    votes = []
    for view in day.get("views", {}).values():
        if view.get("safe_default"):
            continue
        direction = view.get("direction")
        if direction == "long":
            votes.append(1.0)
        elif direction == "flat":
            votes.append(-1.0)
    if not votes:
        return "hold"
    score = sum(votes) / len(votes)
    if score > 0:
        return "buy"
    if score < 0:
        return "sell"
    return "hold"


def parity_check(days: Sequence[Mapping], canonical_series: Mapping[str, Sequence[Sequence]],
                 next_returns: Mapping[str, float], price_dates: Sequence[str] | None = None,
                 tolerance: float = 1e-9) -> dict:
    """Hard verification before any LLM decision layer series is trusted. All checks must pass:

    1. Unique, ordered dates on both sides; the canonical equal-series dates must be
       exactly a PREFIX of the export's dates (tail-only omission — a middle gap would
       let hold-maintain bridge dates the main system actually traded).
    2. Price-calendar adjacency: consecutive export dates must be consecutive price
       dates, so the next-bar return alignment is validated structurally, not just on
       days that happen to be long.
    3. Exposure identity: the equal committee rebuilt from exported views must match the
       canonical equal exposures exactly.
    4. Return identity: on EVERY canonical row of EVERY arm with exposure 1, the
       price-derived next-bar return must equal the recorded gross exactly; NaN anywhere
       fails; dates no arm was long must still have a finite, sane (|r| < 0.5) return.
    """
    failures: list[str] = []
    export_dates = [d["date"] for d in days]
    if len(set(export_dates)) != len(export_dates):
        failures.append("duplicate export dates")
    equal_rows = canonical_series["equal_weight"]
    canon_dates = [row[0] for row in equal_rows]
    if len(set(canon_dates)) != len(canon_dates):
        failures.append("duplicate canonical dates")
    for arm, rows in canonical_series.items():
        if [row[0] for row in rows] != canon_dates:
            failures.append(f"arm {arm} date vector differs from equal_weight")
    if export_dates[: len(canon_dates)] != canon_dates:
        failures.append("canonical dates are not a prefix of export dates (middle gap?)")

    if price_dates is not None:
        idx = {d: i for i, d in enumerate(price_dates)}
        for a, b in zip(export_dates, export_dates[1:]):
            ia, ib = idx.get(a), idx.get(b)
            if ia is None or ib is None or ib != ia + 1:
                failures.append(f"export dates {a}->{b} are not adjacent price dates")
                break

    rebuilt = exposures_from_decisions([equal_committee_decisions(d) for d in days])
    rebuilt_by_date = dict(zip(export_dates, rebuilt))
    long_checked = flat_checked = 0
    for arm, rows in canonical_series.items():
        for date, gross, exposure in rows:
            x = float(exposure)
            g = float(gross)
            if x not in (0.0, 1.0):
                failures.append(f"{arm} {date}: non-binary exposure {exposure!r}")
                continue
            if not math.isfinite(g):
                failures.append(f"{arm} {date}: non-finite gross {gross!r}")
                continue
            r = next_returns.get(date)
            if r is None or not math.isfinite(r):
                failures.append(f"{date}: missing/non-finite price return")
                continue
            if x == 1.0:
                if abs(r - g) > tolerance:
                    failures.append(f"{arm} {date}: price return {r!r} != recorded gross {g!r}")
                long_checked += 1
            elif g != 0.0:
                failures.append(f"{arm} {date}: flat row with nonzero gross {g!r}")
    for date in canon_dates:
        r = next_returns.get(date)
        if r is None or not math.isfinite(r) or abs(r) >= 0.5:
            failures.append(f"{date}: return {r!r} missing, non-finite, or implausible")
        else:
            flat_checked += 1
    for date, gross, exposure in equal_rows:
        if rebuilt_by_date.get(date) != int(exposure):
            failures.append(f"{date}: rebuilt equal exposure {rebuilt_by_date.get(date)} != {int(exposure)}")

    if failures:
        preview = "; ".join(failures[:6])
        raise AssertionError(
            f"parity check FAILED ({len(failures)} problems): {preview}"
        )
    return {
        "checked_days": len(canon_dates),
        "long_rows_return_checked": long_checked,
        "all_dates_return_sanity_checked": flat_checked,
        "mismatches": 0,
    }


# --------------------------------------------------------------------------- cache + runner


@dataclass
class LLMDecisionRun:
    """Result of one LLM decision layer arm over the test window."""

    arm: str
    series: list[list]
    metrics: dict
    decisions: dict = field(default_factory=dict)
    n_api_calls: int = 0
    n_cache_hits: int = 0
    n_parse_errors: int = 0
    n_stale_cache_rows: int = 0


class PromptCache:
    """Append-only JSONL cache; record once, replay free.

    The key covers the FULL decision policy: model, temperature, system prompt, and user
    prompt. Rows are re-keyed from their stored fields on load, so entries recorded under
    a different policy (old system prompt, different model alias/temperature) can never
    silently replay as current-policy decisions — they are counted stale and ignored.
    """

    def __init__(self, path: str | Path, model: str, temperature: float = 0.0,
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.path = Path(path)
        self.model = model
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self._store: dict[str, str] = {}
        self.n_stale_rows = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                expected = self.key(row.get("prompt", ""))
                if row.get("key") == expected:
                    self._store[expected] = row["response"]
                else:
                    self.n_stale_rows += 1

    def key(self, prompt: str) -> str:
        material = "\x00".join(
            [self.model, f"{self.temperature!r}", self.system_prompt, prompt]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]

    def get(self, prompt: str) -> str | None:
        return self._store.get(self.key(prompt))

    def put(self, prompt: str, response: str) -> None:
        key = self.key(prompt)
        self._store[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "prompt": prompt, "response": response},
                                sort_keys=True) + "\n")


def run_llm_decision_arm(
    days: Sequence[Mapping],
    skills: Sequence[tuple[str | None, str]],
    cache: PromptCache,
    call_fn: Callable[[str, str], str] | None,
    next_returns: Mapping[str, float],
    arm: str,
    cost_bps_per_side: float = 10.0,
) -> LLMDecisionRun:
    """Evaluate one LLM decision layer arm date by date. call_fn=None forbids API calls
    (pure replay; a cache miss then raises instead of spending)."""
    decisions: list[str] = []
    log: dict[str, dict] = {}
    n_calls = n_hits = n_parse_errors = 0
    for day in days:
        prompt = build_user_prompt(day, skills)
        response = cache.get(prompt)
        if response is None:
            if call_fn is None:
                raise RuntimeError(f"cache miss on {day['date']} with API calls forbidden")
            response = call_fn(cache.system_prompt, prompt)
            cache.put(prompt, response)
            n_calls += 1
        else:
            n_hits += 1
        decision, parse_error = parse_decision(response)
        n_parse_errors += int(parse_error)
        decisions.append(decision)
        log[day["date"]] = {"decision": decision, "parse_error": parse_error}
    exposures = exposures_from_decisions(decisions)
    series = series_from_exposures([d["date"] for d in days], exposures, next_returns)
    return LLMDecisionRun(
        arm=arm,
        series=series,
        metrics=metrics_from_series(series, cost_bps_per_side),
        decisions=log,
        n_api_calls=n_calls,
        n_cache_hits=n_hits,
        n_parse_errors=n_parse_errors,
        n_stale_cache_rows=cache.n_stale_rows,
    )
